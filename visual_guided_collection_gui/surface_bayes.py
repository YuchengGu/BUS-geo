from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from breast_path_planning.geometry import normalize_vector, rodrigues
from visual_guided_collection_gui.surface_teleop import matrix_to_rotvec
from visual_guided_collection_gui.ultrasound_quality import QualityScore, UltrasoundQualityScorer
from visual_guided_collection_gui.offline_bayes import LocalBayesOptimizer, LocalBOConfig


DEFAULT_LOCAL_BOUNDS: tuple[tuple[float, float], ...] = (
    (-0.05, 0.05),
    (-0.0873, 0.0873),
    (-0.0873, 0.0873),
    (-0.0873, 0.0873),
)
LOCAL_BOUND_NAMES = ("dn", "rx", "ry", "rz")


@dataclass(frozen=True)
class SurfaceBOConfig:
    bounds: tuple[tuple[float, float], ...] = DEFAULT_LOCAL_BOUNDS
    n_initial: int = 3
    n_ei: int = 12
    force_enabled: bool = True
    lambda_force: float = 0.05
    lambda_torque: float = 0.01
    force_epsilon: float = 0.2
    torque_epsilon: float = 0.02
    force_max: float = 15.0
    torque_max: float = 2.0
    large_penalty: float = 1_000.0
    settle_s: float = 0.1
    quality_max_size: int | None = None
    quality_speckle_max_size: int | None = None
    quality_confidence_max_size: int | None = 110
    quality_confidence_method: str = "fast"

    @property
    def max_trials(self) -> int:
        return int(self.n_initial) + int(self.n_ei)


@dataclass(frozen=True)
class ForceTorquePenalty:
    force_penalty: float
    torque_penalty: float
    force_valid: bool
    force_values: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))


@dataclass(frozen=True)
class OnlineObjective:
    F: float
    Q: float
    quality: QualityScore
    force_penalty: float
    torque_penalty: float
    force_valid: bool

    def meta(self) -> dict[str, Any]:
        return {
            "F": float(self.F),
            "Q": float(self.Q),
            "D": float(self.quality.features.D),
            "E": float(self.quality.features.E),
            "C": float(self.quality.features.C),
            "S": float(self.quality.features.S),
            "P_f": float(self.force_penalty),
            "P_tau": float(self.torque_penalty),
            "force_valid": bool(self.force_valid),
        }


@dataclass(frozen=True)
class SurfaceBOTrial:
    index: int
    phase: str
    x: np.ndarray
    target_tcp_pose: np.ndarray
    objective: OnlineObjective | None
    error: str | None = None


@dataclass(frozen=True)
class SurfaceBORunResult:
    trial_count: int
    cancelled: bool
    best_x: np.ndarray | None
    best_F: float | None
    trials: tuple[SurfaceBOTrial, ...]
    error: str | None = None


class SurfaceBOStopSignal:
    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event or threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    def should_stop(self) -> bool:
        return self._event.is_set()


class SurfaceBOCancelled(RuntimeError):
    pass


def make_quality_scorer(config: SurfaceBOConfig) -> UltrasoundQualityScorer:
    return UltrasoundQualityScorer(
        max_size=config.quality_max_size,
        speckle_max_size=config.quality_speckle_max_size,
        confidence_max_size=config.quality_confidence_max_size,
        confidence_method=config.quality_confidence_method,
    )


def parse_local_bounds(text: str) -> tuple[tuple[float, float], ...]:
    values: dict[str, tuple[float, float]] = {}
    for chunk in str(text).split(";"):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid BO bound item {item!r}; expected name=low,high")
        name, pair = item.split("=", 1)
        name = name.strip()
        if name not in LOCAL_BOUND_NAMES:
            raise ValueError(f"unknown BO bound name {name!r}; expected one of {LOCAL_BOUND_NAMES}")
        parts = [part.strip() for part in pair.split(",")]
        if len(parts) != 2:
            raise ValueError(f"invalid BO bound for {name!r}; expected low,high")
        low, high = float(parts[0]), float(parts[1])
        if high <= low:
            raise ValueError(f"invalid BO bound for {name!r}; high must be greater than low")
        values[name] = (low, high)
    missing = [name for name in LOCAL_BOUND_NAMES if name not in values]
    if missing:
        raise ValueError(f"missing BO bounds for: {', '.join(missing)}")
    return tuple(values[name] for name in LOCAL_BOUND_NAMES)


def compute_candidate_tcp_pose(
    reference_tcp_pose: np.ndarray,
    normal_base: np.ndarray,
    local_x: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_tcp_pose, dtype=float).reshape(6)
    x = np.asarray(local_x, dtype=float).reshape(4)
    normal = normalize_vector(np.asarray(normal_base, dtype=float).reshape(3))
    position = reference[:3] + float(x[0]) * normal
    rotation = rodrigues(reference[3:]) @ rodrigues(x[1:4])
    return np.concatenate([position, matrix_to_rotvec(rotation)])


def select_current_tcp_bo_reference(
    obs: dict[str, Any],
    enriched_obs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    reference_pose = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
    nearest = int(enriched_obs["path_nearest_index"])
    normals = np.asarray(enriched_obs["path_normals_base"], dtype=float)
    if normals.ndim == 1:
        normal = normals.reshape(3)
    else:
        normal = normals[0].reshape(3)
    return reference_pose, normalize_vector(normal), nearest


def compute_force_torque_penalty(
    force: np.ndarray | None,
    config: SurfaceBOConfig,
) -> ForceTorquePenalty:
    if not config.force_enabled or force is None:
        return ForceTorquePenalty(0.0, 0.0, False)
    values = np.asarray(force, dtype=float).reshape(-1)
    if values.shape[0] < 6 or not np.all(np.isfinite(values[:6])):
        return ForceTorquePenalty(0.0, 0.0, False)
    values = values[:6].copy()
    f_t = float(np.linalg.norm(values[:2]))
    f_n = abs(float(values[2]))
    tau_t = float(np.linalg.norm(values[3:5]))
    tau_n = abs(float(values[5]))
    force_over = max(0.0, f_n - float(config.force_max))
    torque_over = max(0.0, tau_n - float(config.torque_max))
    force_penalty = float(config.lambda_force) * (
        f_t / (f_n + float(config.force_epsilon))
        + (force_over / (float(config.force_max) + 1e-12)) ** 2
    )
    torque_penalty = float(config.lambda_torque) * (
        tau_t / (tau_n + float(config.torque_epsilon))
        + (torque_over / (float(config.torque_max) + 1e-12)) ** 2
    )
    return ForceTorquePenalty(force_penalty, torque_penalty, True, values)


def ultrasound_frame_from_obs(obs: dict[str, Any]) -> np.ndarray:
    frame = obs.get("Ultrasound_gray")
    if frame is None:
        frame = obs.get("Ultrasound_rgb")
    if frame is None:
        raise RuntimeError("Online BO requires obs['Ultrasound_gray'] or obs['Ultrasound_rgb']")
    return np.asarray(frame)


def compute_online_objective(
    obs: dict[str, Any],
    config: SurfaceBOConfig,
    *,
    scorer: UltrasoundQualityScorer | None = None,
) -> OnlineObjective:
    quality_scorer = scorer or make_quality_scorer(config)
    quality = quality_scorer.score_frame(ultrasound_frame_from_obs(obs))
    penalty = compute_force_torque_penalty(obs.get("force"), config)
    value = -float(quality.Q) + float(penalty.force_penalty) + float(penalty.torque_penalty)
    return OnlineObjective(
        F=value,
        Q=float(quality.Q),
        quality=quality,
        force_penalty=float(penalty.force_penalty),
        torque_penalty=float(penalty.torque_penalty),
        force_valid=bool(penalty.force_valid),
    )


def run_surface_bayes_optimization(
    *,
    devices,
    reference_tcp_pose: np.ndarray,
    normal_base: np.ndarray,
    recorder=None,
    config: SurfaceBOConfig | None = None,
    stop_signal: SurfaceBOStopSignal | None = None,
    scorer: UltrasoundQualityScorer | None = None,
    on_status: Callable[[str], None] | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
    candidate_allowed: Callable[[np.ndarray], bool] | None = None,
) -> SurfaceBORunResult:
    cfg = config or SurfaceBOConfig()
    signal = stop_signal or SurfaceBOStopSignal()
    quality_scorer = scorer or make_quality_scorer(cfg)
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=cfg.bounds,
            n_initial=cfg.n_initial,
            max_trials=cfg.max_trials,
            backend="sklearn_ei",
            random_state=None,
        )
    )
    trials: list[SurfaceBOTrial] = []
    target_history: list[np.ndarray] = []

    while not optimizer.should_stop():
        if signal.should_stop():
            return _surface_bo_result(optimizer, trials, cancelled=True)

        trial_index = optimizer.n_observed
        phase = "initial" if trial_index < cfg.n_initial else "EI"
        x = optimizer.ask()
        target = compute_candidate_tcp_pose(reference_tcp_pose, normal_base, x)
        target_history.append(target.copy())

        if candidate_allowed is not None and not candidate_allowed(target):
            optimizer.tell(x, float(cfg.large_penalty))
            trials.append(
                SurfaceBOTrial(
                    index=trial_index,
                    phase=phase,
                    x=x.copy(),
                    target_tcp_pose=target.copy(),
                    objective=None,
                    error="target_rejected_before_motion",
                )
            )
            continue

        if on_status is not None:
            on_status(f"Surface BO trial {trial_index + 1}/{cfg.max_trials}: {phase}")

        try:
            def waypoint_callback(_record: dict[str, Any]) -> None:
                if signal.should_stop():
                    raise SurfaceBOCancelled("Surface BO stopped by user")

            devices.move_tcp_pose_linear(
                target,
                max_position_step_m=0.001,
                max_rotation_step_rad=0.006,
                position_tolerance_m=0.002,
                rotation_tolerance_rad=0.03,
                timeout_s=60.0,
                waypoint_callback=waypoint_callback,
            )
            if cfg.settle_s > 0.0:
                time.sleep(float(cfg.settle_s))
            obs = devices.get_obs()
        except SurfaceBOCancelled as exc:
            trials.append(
                SurfaceBOTrial(
                    index=trial_index,
                    phase=phase,
                    x=x.copy(),
                    target_tcp_pose=target.copy(),
                    objective=None,
                    error=str(exc),
                )
            )
            return _surface_bo_result(optimizer, trials, cancelled=True)
        except Exception as exc:
            trials.append(
                SurfaceBOTrial(
                    index=trial_index,
                    phase=phase,
                    x=x.copy(),
                    target_tcp_pose=target.copy(),
                    objective=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return _surface_bo_result(optimizer, trials, cancelled=True, error=str(exc))

        objective = compute_online_objective(obs, cfg, scorer=quality_scorer)
        optimizer.tell(x, objective.F)
        meta = {
            "operation_mode": "auto",
            "auto_phase": "bo",
            "bo_trial_index": int(trial_index),
            "bo_phase": phase,
            "bo_x": x.tolist(),
            "bo_target_tcp_pose": target.tolist(),
            "best_F": optimizer.best_y,
            "best_x": None if optimizer.best_x is None else optimizer.best_x.tolist(),
            **objective.meta(),
        }
        if recorder is not None:
            recorder.save_sample(obs, target, meta=meta)
        if on_sample is not None:
            on_sample({"obs": obs, "action": target, "meta": meta})
        trials.append(
            SurfaceBOTrial(
                index=trial_index,
                phase=phase,
                x=x.copy(),
                target_tcp_pose=target.copy(),
                objective=objective,
            )
        )

        if signal.should_stop():
            return _surface_bo_result(optimizer, trials, cancelled=True)

    best_x = optimizer.best_x
    if best_x is not None and trials:
        best_target = compute_candidate_tcp_pose(reference_tcp_pose, normal_base, best_x)
        if not np.allclose(best_target, target_history[-1], atol=1e-8, rtol=0.0):
            try:
                devices.move_tcp_pose_linear(
                    best_target,
                    max_position_step_m=0.001,
                    max_rotation_step_rad=0.006,
                    position_tolerance_m=0.002,
                    rotation_tolerance_rad=0.03,
                    timeout_s=60.0,
                )
            except Exception as exc:
                return _surface_bo_result(optimizer, trials, cancelled=True, error=str(exc))

    return _surface_bo_result(optimizer, trials, cancelled=False)


def _surface_bo_result(
    optimizer: LocalBayesOptimizer,
    trials: list[SurfaceBOTrial],
    *,
    cancelled: bool,
    error: str | None = None,
) -> SurfaceBORunResult:
    return SurfaceBORunResult(
        trial_count=len(trials),
        cancelled=bool(cancelled),
        best_x=optimizer.best_x,
        best_F=optimizer.best_y,
        trials=tuple(trials),
        error=error,
    )
