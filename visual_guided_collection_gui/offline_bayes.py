from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass(frozen=True)
class LocalBOConfig:
    bounds: Sequence[tuple[float, float]]
    n_initial: int = 3
    max_trials: int = 12
    backend: str = "auto"
    xi: float = 0.01
    xi_boost: float = 0.1
    candidate_count: int = 4096
    stagnation_window: int = 3
    convergence_window: int = 7
    min_improvement: float = 1e-4
    random_state: int | None = None


class LocalBayesOptimizer:
    """Ask/tell Bayesian optimizer for local probe-pose minimization."""

    def __init__(self, config: LocalBOConfig) -> None:
        self.config = config
        self.bounds = np.asarray(config.bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (low, high) pairs")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("each bound high value must be greater than low value")
        if config.n_initial < 1:
            raise ValueError("n_initial must be at least 1")
        if config.max_trials < config.n_initial:
            raise ValueError("max_trials must be greater than or equal to n_initial")
        self.rng = np.random.default_rng(config.random_state)
        self._initial_points = self._make_initial_points(config.n_initial)
        self._next_initial = 0
        self.x_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []
        self._best_history: list[float] = []
        self._skopt_optimizer = self._make_skopt_optimizer(config.backend)
        self.last_ask_used_boost = False

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def n_observed(self) -> int:
        return len(self.y_observed)

    @property
    def best_x(self) -> np.ndarray | None:
        if not self.y_observed:
            return None
        return self.x_observed[int(np.argmin(self.y_observed))].copy()

    @property
    def best_y(self) -> float | None:
        if not self.y_observed:
            return None
        return float(np.min(self.y_observed))

    def ask(self) -> np.ndarray:
        self.last_ask_used_boost = False
        if self.should_boost_exploration():
            self.last_ask_used_boost = True
            return self._ask_by_expected_improvement(xi=float(self.config.xi_boost))
        if self._skopt_optimizer is not None:
            return np.asarray(self._skopt_optimizer.ask(), dtype=float)
        if self._next_initial < len(self._initial_points):
            point = self._initial_points[self._next_initial].copy()
            self._next_initial += 1
            return point
        if self.n_observed < self.config.n_initial:
            return self._sample_random_unique()
        return self._ask_by_expected_improvement(xi=float(self.config.xi))

    def tell(self, x: np.ndarray, y: float) -> None:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},), got {point.shape}")
        if np.any(point < self.bounds[:, 0]) or np.any(point > self.bounds[:, 1]):
            raise ValueError(f"x is outside bounds: {point}")
        value = float(y)
        if not np.isfinite(value):
            raise ValueError(f"objective value must be finite, got {y}")
        self.x_observed.append(point.copy())
        self.y_observed.append(value)
        self._best_history.append(float(np.min(self.y_observed)))
        if self._skopt_optimizer is not None:
            self._skopt_optimizer.tell(point.tolist(), value)

    def should_stop(self) -> bool:
        if self.n_observed >= self.config.max_trials:
            return True
        if self.n_observed <= self.config.n_initial:
            return False
        window = int(self.config.convergence_window)
        if self.n_observed < self.config.n_initial + window:
            return False
        if window <= 0 or len(self._best_history) < window + 1:
            return False
        return self._recent_ei_best_changes_are_small(window)

    def should_boost_exploration(self) -> bool:
        window = int(self.config.stagnation_window)
        if window <= 0:
            return False
        if self.n_observed < self.config.n_initial + window:
            return False
        return self._recent_ei_best_changes_are_small(window)

    def expected_improvement(self, candidates: np.ndarray, *, xi: float | None = None) -> np.ndarray:
        if self.n_observed < self.config.n_initial:
            raise RuntimeError("expected improvement requires initial observations")
        model = self._fit_model()
        candidates = np.asarray(candidates, dtype=float)
        mu, sigma = model.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        best = float(np.min(self.y_observed))
        improvement = best - mu - float(self.config.xi if xi is None else xi)
        z = improvement / sigma
        return improvement * norm.cdf(z) + sigma * norm.pdf(z)

    def posterior(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return GP posterior mean and standard deviation at candidate points."""
        if self.n_observed < self.config.n_initial:
            raise RuntimeError("posterior requires initial observations")
        points = np.asarray(candidates, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(f"candidates must have shape (N, {self.dim})")
        model = self._fit_model()
        mean, std = model.predict(points, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    def _ask_by_expected_improvement(self, *, xi: float) -> np.ndarray:
        candidates = self._sample_candidates(int(self.config.candidate_count))
        if not len(candidates):
            return self._sample_random_unique()
        ei = self.expected_improvement(candidates, xi=xi)
        return candidates[int(np.argmax(ei))].copy()

    def _recent_ei_best_changes_are_small(self, window: int) -> bool:
        recent = np.asarray(self._best_history[-(window + 1):], dtype=float)
        adjacent_improvements = np.abs(np.diff(recent))
        return bool(np.all(adjacent_improvements < self.config.min_improvement))

    def _fit_model(self) -> GaussianProcessRegressor:
        x = np.vstack(self.x_observed)
        y = np.asarray(self.y_observed, dtype=float)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.config.random_state,
            n_restarts_optimizer=2,
        )
        model.fit(x, y)
        return model

    def _make_initial_points(self, count: int) -> np.ndarray:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        if count == 1:
            return ((lows + highs) / 2.0).reshape(1, -1)
        unit = np.empty((count, self.dim), dtype=float)
        for dim in range(self.dim):
            perm = self.rng.permutation(count)
            unit[:, dim] = (perm + 0.5) / count
        return lows + unit * (highs - lows)

    def _make_skopt_optimizer(self, backend: str):
        backend = backend.lower()
        if backend not in {"auto", "skopt", "sklearn_ei"}:
            raise ValueError("backend must be one of: auto, skopt, sklearn_ei")
        if backend == "sklearn_ei":
            return None
        try:
            from skopt import Optimizer
            from skopt.space import Real
        except ImportError as exc:
            if backend == "skopt":
                raise ImportError(
                    "LocalBOConfig backend='skopt' requires scikit-optimize. "
                    "Install it with: pip install scikit-optimize"
                ) from exc
            return None
        dimensions = [Real(float(low), float(high)) for low, high in self.bounds]
        return Optimizer(
            dimensions=dimensions,
            base_estimator="GP",
            acq_func="EI",
            acq_func_kwargs={"xi": float(self.config.xi)},
            n_initial_points=int(self.config.n_initial),
            random_state=self.config.random_state,
        )

    def _sample_candidates(self, count: int) -> np.ndarray:
        candidates = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(count, self.dim))
        keep = [not self._is_observed(candidate) for candidate in candidates]
        return candidates[np.asarray(keep, dtype=bool)]

    def _sample_random_unique(self) -> np.ndarray:
        for _ in range(1000):
            candidate = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.dim)
            if not self._is_observed(candidate):
                return candidate
        return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.dim)

    def _is_observed(self, candidate: np.ndarray) -> bool:
        return any(np.allclose(candidate, observed, atol=1e-8, rtol=0.0) for observed in self.x_observed)
