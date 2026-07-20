from __future__ import annotations

import json
import threading
import time
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from breast_path_planning.geometry import normalize_vector, rodrigues
from gello.utils.control_utils import build_time_alignment_meta
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
SURFACE_BO_SEARCH_STRATEGIES = ("bo", "random", "lhs", "uniform")
SURFACE_BO_OBJECTIVE_VARIANTS = ("full", "no_penalty", "force_only", "torque_only")
DEFAULT_ULTRASOUND_CROP: tuple[int, int, int, int] = (99, 769, 542, 1524)


@dataclass(frozen=True)
class SurfaceBOConfig:
    bounds: tuple[tuple[float, float], ...] = DEFAULT_LOCAL_BOUNDS
    n_initial: int = 3
    n_ei: int = 12
    search_strategy: str = "bo"
    objective_variant: str = "full"
    random_state: int | None = None
    force_enabled: bool = True
    lambda_pressure: float = 0.08
    lambda_shear: float = 0.03
    lambda_torque: float = 0.02
    lambda_axial_torque: float = 0.005
    pressure_min: float = 2.0
    pressure_max: float = 8.0
    shear_max: float = 6.0
    torque_tangential_max: float = 0.8
    torque_axial_max: float = 0.5
    large_penalty: float = 1_000.0
    settle_s: float = 0.2
    quality_max_size: int | None = None
    quality_speckle_max_size: int | None = None
    quality_confidence_max_size: int | None = 110
    quality_confidence_method: str = "fast"
    ultrasound_crop: tuple[int, int, int, int] | None = DEFAULT_ULTRASOUND_CROP

    @property
    def max_trials(self) -> int:
        return int(self.n_initial) + int(self.n_ei)

    def __post_init__(self) -> None:
        if self.search_strategy not in SURFACE_BO_SEARCH_STRATEGIES:
            raise ValueError(
                f"search_strategy must be one of {SURFACE_BO_SEARCH_STRATEGIES}, "
                f"got {self.search_strategy!r}"
            )
        if self.objective_variant not in SURFACE_BO_OBJECTIVE_VARIANTS:
            raise ValueError(
                f"objective_variant must be one of {SURFACE_BO_OBJECTIVE_VARIANTS}, "
                f"got {self.objective_variant!r}"
            )


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
    before_objective: OnlineObjective | None = None
    verified_best_objective: OnlineObjective | None = None
    posterior_slices: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
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


def parse_ultrasound_crop(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None:
        return DEFAULT_ULTRASOUND_CROP
    item = str(text).strip().lower()
    if item in {"", "none", "full", "0"}:
        return None
    parts = [part.strip() for part in item.split(",")]
    if len(parts) != 4:
        raise ValueError("ultrasound crop must be row0,row1,col0,col1 or 'none'")
    row0, row1, col0, col1 = (int(float(part)) for part in parts)
    if row1 <= row0 or col1 <= col0:
        raise ValueError("ultrasound crop must satisfy row1 > row0 and col1 > col0")
    return (row0, row1, col0, col1)


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


def post_run_reset_tcp_targets(
    reference_tcp_pose: np.ndarray,
    normal_base: np.ndarray,
    *,
    retreat_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_tcp_pose, dtype=float).reshape(6)
    normal = normalize_vector(np.asarray(normal_base, dtype=float).reshape(3))
    retreat = reference.copy()
    retreat[:3] = reference[:3] + float(retreat_distance_m) * normal
    return retreat, reference.copy()


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
    pressure = max(0.0, -float(values[2]))
    tau_t = float(np.linalg.norm(values[3:5]))
    tau_z = abs(float(values[5]))
    pressure_low = max(0.0, float(config.pressure_min) - pressure)
    pressure_high = max(0.0, pressure - float(config.pressure_max))
    force_penalty = (
        float(config.lambda_pressure)
        * (
            (pressure_low / (float(config.pressure_min) + 1e-12)) ** 2
            + (pressure_high / (float(config.pressure_max) + 1e-12)) ** 2
        )
        + float(config.lambda_shear)
        * (f_t / (float(config.shear_max) + 1e-12)) ** 2
    )
    torque_penalty = (
        float(config.lambda_torque)
        * (tau_t / (float(config.torque_tangential_max) + 1e-12)) ** 2
        + float(config.lambda_axial_torque)
        * (tau_z / (float(config.torque_axial_max) + 1e-12)) ** 2
    )
    return ForceTorquePenalty(force_penalty, torque_penalty, True, values)


def crop_ultrasound_frame(
    frame: np.ndarray,
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    image = np.asarray(frame)
    if crop is None:
        return image
    if image.ndim < 2:
        return image
    height, width = image.shape[:2]
    row0, row1, col0, col1 = crop
    row0 = max(0, min(int(row0), height))
    row1 = max(0, min(int(row1), height))
    col0 = max(0, min(int(col0), width))
    col1 = max(0, min(int(col1), width))
    if row1 <= row0 or col1 <= col0:
        return image
    return image[row0:row1, col0:col1, ...]


def ultrasound_frame_from_obs(
    obs: dict[str, Any],
    *,
    crop: tuple[int, int, int, int] | None = DEFAULT_ULTRASOUND_CROP,
) -> np.ndarray:
    frame = obs.get("Ultrasound_gray")
    if frame is None:
        frame = obs.get("Ultrasound_rgb")
    if frame is None:
        raise RuntimeError("Online BO requires obs['Ultrasound_gray'] or obs['Ultrasound_rgb']")
    return crop_ultrasound_frame(np.asarray(frame), crop)


def compute_online_objective(
    obs: dict[str, Any],
    config: SurfaceBOConfig,
    *,
    scorer: UltrasoundQualityScorer | None = None,
) -> OnlineObjective:
    quality_scorer = scorer or make_quality_scorer(config)
    quality = quality_scorer.score_frame(ultrasound_frame_from_obs(obs, crop=config.ultrasound_crop))
    penalty = compute_force_torque_penalty(obs.get("force"), config)
    value = -float(quality.Q)
    if config.objective_variant in {"full", "force_only"}:
        value += float(penalty.force_penalty)
    if config.objective_variant in {"full", "torque_only"}:
        value += float(penalty.torque_penalty)
    return OnlineObjective(
        F=value,
        Q=float(quality.Q),
        quality=quality,
        force_penalty=float(penalty.force_penalty),
        torque_penalty=float(penalty.torque_penalty),
        force_valid=bool(penalty.force_valid),
    )


class _DirectSearchOptimizer:
    def __init__(self, config: LocalBOConfig, *, strategy: str) -> None:
        self.config = config
        self.strategy = strategy
        self.bounds = np.asarray(config.bounds, dtype=float)
        self.rng = np.random.default_rng(config.random_state)
        self.x_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []
        self.last_ask_used_boost = False
        self._lhs_points = self._make_lhs_points(int(config.max_trials))

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

    def should_stop(self) -> bool:
        return self.n_observed >= int(self.config.max_trials)

    def ask(self) -> np.ndarray:
        if self.strategy in {"lhs", "uniform"}:
            return self._lhs_points[self.n_observed].copy()
        return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])

    def tell(self, x: np.ndarray, y: float) -> None:
        point = np.asarray(x, dtype=float).reshape(-1)
        self.x_observed.append(point.copy())
        self.y_observed.append(float(y))

    def _make_lhs_points(self, count: int) -> np.ndarray:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        if count <= 0:
            return np.empty((0, self.bounds.shape[0]), dtype=float)
        unit = np.empty((count, self.bounds.shape[0]), dtype=float)
        for dim in range(self.bounds.shape[0]):
            strata = (np.arange(count, dtype=float) + self.rng.uniform(size=count)) / float(count)
            unit[:, dim] = strata[self.rng.permutation(count)]
        return lows + unit * (highs - lows)


def _make_search_optimizer(config: SurfaceBOConfig):
    local_config = LocalBOConfig(
        bounds=config.bounds,
        n_initial=config.n_initial,
        max_trials=config.max_trials,
        backend="sklearn_ei",
        random_state=config.random_state,
    )
    if config.search_strategy == "bo":
        return LocalBayesOptimizer(local_config)
    return _DirectSearchOptimizer(local_config, strategy=config.search_strategy)


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
    optimizer = _make_search_optimizer(cfg)
    trials: list[SurfaceBOTrial] = []
    target_history: list[np.ndarray] = []
    before_obs = devices.get_obs()
    before_objective = compute_online_objective(before_obs, cfg, scorer=quality_scorer)
    _record_bo_measurement(
        devices=devices,
        recorder=recorder,
        obs=before_obs,
        action=np.asarray(reference_tcp_pose, dtype=float),
        objective=before_objective,
        role="before",
        local_x=np.zeros(4, dtype=float),
        trial_index=None,
        phase="reference",
        best_F=None,
        best_x=None,
        config=cfg,
        on_sample=on_sample,
    )

    while not optimizer.should_stop():
        if signal.should_stop():
            return _surface_bo_result(
                optimizer,
                trials,
                cancelled=True,
                before_objective=before_objective,
            )

        trial_index = optimizer.n_observed
        if cfg.search_strategy == "bo":
            phase = "initial" if trial_index < cfg.n_initial else "EI"
        else:
            phase = cfg.search_strategy
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
            def waypoint_callback(record: dict[str, Any]) -> None:
                _record_bo_waypoint(
                    devices=devices,
                    recorder=recorder,
                    record=record,
                    trial_index=trial_index,
                    phase=phase,
                    stage="candidate",
                    on_sample=on_sample,
                )
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
        _record_bo_measurement(
            devices=devices,
            recorder=recorder,
            obs=obs,
            action=target,
            objective=objective,
            role="candidate",
            local_x=x,
            trial_index=trial_index,
            phase=phase,
            best_F=optimizer.best_y,
            best_x=optimizer.best_x,
            config=cfg,
            on_sample=on_sample,
        )
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
            return _surface_bo_result(
                optimizer,
                trials,
                cancelled=True,
                before_objective=before_objective,
            )

    best_x = optimizer.best_x
    verified_best_objective = None
    if best_x is not None and trials:
        best_target = compute_candidate_tcp_pose(reference_tcp_pose, normal_base, best_x)
        if not np.allclose(best_target, target_history[-1], atol=1e-8, rtol=0.0):
            try:
                def best_waypoint_callback(record: dict[str, Any]) -> None:
                    _record_bo_waypoint(
                        devices=devices,
                        recorder=recorder,
                        record=record,
                        trial_index=None,
                        phase="return_best",
                        stage="return_best",
                        on_sample=on_sample,
                    )
                    if signal.should_stop():
                        raise SurfaceBOCancelled("Surface BO stopped by user")

                devices.move_tcp_pose_linear(
                    best_target,
                    max_position_step_m=0.001,
                    max_rotation_step_rad=0.006,
                    position_tolerance_m=0.002,
                    rotation_tolerance_rad=0.03,
                    timeout_s=60.0,
                    waypoint_callback=best_waypoint_callback,
                )
            except Exception as exc:
                return _surface_bo_result(
                    optimizer,
                    trials,
                    cancelled=True,
                    before_objective=before_objective,
                    error=str(exc),
                )
        if cfg.settle_s > 0.0:
            time.sleep(float(cfg.settle_s))
        verified_obs = devices.get_obs()
        verified_best_objective = compute_online_objective(
            verified_obs,
            cfg,
            scorer=quality_scorer,
        )
        _record_bo_measurement(
            devices=devices,
            recorder=recorder,
            obs=verified_obs,
            action=best_target,
            objective=verified_best_objective,
            role="verified_best",
            local_x=best_x,
            trial_index=None,
            phase="verification",
            best_F=optimizer.best_y,
            best_x=best_x,
            config=cfg,
            on_sample=on_sample,
        )

    return _surface_bo_result(
        optimizer,
        trials,
        cancelled=False,
        before_objective=before_objective,
        verified_best_objective=verified_best_objective,
    )


def _surface_bo_result(
    optimizer,
    trials: list[SurfaceBOTrial],
    *,
    cancelled: bool,
    before_objective: OnlineObjective | None = None,
    verified_best_objective: OnlineObjective | None = None,
    error: str | None = None,
) -> SurfaceBORunResult:
    posterior_slices = _posterior_slices(optimizer)
    return SurfaceBORunResult(
        trial_count=len(trials),
        cancelled=bool(cancelled),
        best_x=optimizer.best_x,
        best_F=optimizer.best_y,
        trials=tuple(trials),
        before_objective=before_objective,
        verified_best_objective=verified_best_objective,
        posterior_slices=posterior_slices,
        error=error,
    )


def _record_bo_waypoint(
    *,
    devices,
    recorder,
    record: dict[str, Any],
    trial_index: int | None,
    phase: str,
    stage: str,
    on_sample: Callable[[dict[str, Any]], None] | None,
) -> None:
    if record.get("kind") != "waypoint" or recorder is None:
        return
    obs = devices.get_obs()
    action = np.asarray(record["target_tcp_pose"], dtype=float)
    timestamp = datetime.datetime.now()
    meta = _bo_time_alignment_meta(
        devices=devices,
        recorder=recorder,
        timestamp=timestamp,
        action_timing={},
    )
    meta.update({
        "auto_phase": "bo_move",
        "bo_is_measurement": False,
        "bo_counts_toward_budget": False,
        "bo_move_stage": stage,
        "bo_trial_index": trial_index,
        "bo_phase": phase,
        "bo_waypoint_index": int(record.get("index", 0)),
        "bo_waypoint_count": int(record.get("count", 0)),
    })
    recorder.save_sample(obs, action, meta=meta, timestamp=timestamp)
    if on_sample is not None:
        on_sample({"obs": obs, "action": action, "meta": meta})


def _record_bo_measurement(
    *,
    devices,
    recorder,
    obs: dict[str, Any],
    action: np.ndarray,
    objective: OnlineObjective,
    role: str,
    local_x: np.ndarray,
    trial_index: int | None,
    phase: str,
    best_F: float | None,
    best_x: np.ndarray | None,
    config: SurfaceBOConfig,
    on_sample: Callable[[dict[str, Any]], None] | None,
) -> None:
    timestamp = datetime.datetime.now()
    meta = _bo_time_alignment_meta(
        devices=devices,
        recorder=recorder,
        timestamp=timestamp,
        action_timing={},
    )
    meta.update({
        "auto_phase": "bo",
        "bo_is_measurement": True,
        "bo_counts_toward_budget": role == "candidate",
        "bo_measurement_role": role,
        "bo_trial_index": trial_index,
        "bo_phase": phase,
        "bo_search_strategy": config.search_strategy,
        "bo_objective_variant": config.objective_variant,
        "bo_x": np.asarray(local_x, dtype=float).tolist(),
        "bo_target_tcp_pose": np.asarray(action, dtype=float).tolist(),
        "best_F": best_F,
        "best_x": None if best_x is None else np.asarray(best_x, dtype=float).tolist(),
        **objective.meta(),
    })
    if recorder is not None:
        recorder.save_sample(obs, action, meta=meta, timestamp=timestamp)
    if on_sample is not None:
        on_sample({"obs": obs, "action": np.asarray(action), "meta": meta})


def _bo_time_alignment_meta(
    *,
    devices,
    recorder,
    timestamp: datetime.datetime,
    action_timing: dict[str, int],
) -> dict[str, Any]:
    env = getattr(devices, "env", None)
    obs_meta = dict(getattr(env, "last_obs_meta", {}) or {})
    sample_index = 0 if recorder is None else int(getattr(recorder, "sample_index", 0))
    episode_dir = None if recorder is None else getattr(recorder, "episode_dir", None)
    return build_time_alignment_meta(
        sample_index=sample_index,
        episode_id=None if episode_dir is None else episode_dir.name,
        control_loop_hz_config=getattr(env, "control_rate_hz", None),
        wall_time=timestamp,
        obs_meta=obs_meta,
        action_timing=action_timing,
        step_timing=dict(getattr(env, "last_step_timing", {}) or {}),
    )


def _posterior_slices(
    optimizer,
    *,
    points_per_dimension: int = 201,
) -> dict[str, dict[str, np.ndarray]]:
    if not hasattr(optimizer, "posterior") or not hasattr(optimizer, "expected_improvement"):
        return {}
    if optimizer.n_observed < optimizer.config.n_initial or optimizer.best_x is None:
        return {}
    slices: dict[str, dict[str, np.ndarray]] = {}
    for dimension, name in enumerate(LOCAL_BOUND_NAMES):
        grid = np.linspace(
            optimizer.bounds[dimension, 0],
            optimizer.bounds[dimension, 1],
            int(points_per_dimension),
        )
        candidates = np.tile(optimizer.best_x, (len(grid), 1))
        candidates[:, dimension] = grid
        mean, std = optimizer.posterior(candidates)
        ei = optimizer.expected_improvement(candidates)
        slices[name] = {
            "grid": grid,
            "mean": mean,
            "std": std,
            "ei": ei,
        }
    return slices


def save_surface_bo_run_artifacts(
    output_dir: str | Path,
    result: SurfaceBORunResult,
    *,
    config: SurfaceBOConfig,
    reference_tcp_pose: np.ndarray,
    normal_base: np.ndarray,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "surface_bo_run.json"
    posterior_path = output / "surface_bo_posterior.npz"
    trials = []
    for trial in result.trials:
        trials.append(
            {
                "index": trial.index,
                "phase": trial.phase,
                "x": trial.x.tolist(),
                "target_tcp_pose": trial.target_tcp_pose.tolist(),
                "objective": None if trial.objective is None else trial.objective.meta(),
                "error": trial.error,
            }
        )
    summary = {
        "schema_version": "surface_bo_run_v1",
        "reference_tcp_pose": np.asarray(reference_tcp_pose, dtype=float).tolist(),
        "normal_base": np.asarray(normal_base, dtype=float).tolist(),
        "bounds": [list(pair) for pair in config.bounds],
        "n_initial": config.n_initial,
        "n_ei": config.n_ei,
        "search_strategy": config.search_strategy,
        "objective_variant": config.objective_variant,
        "random_state": config.random_state,
        "penalty_model": "soft_pressure_shear_torque_v1",
        "pressure_min_N": config.pressure_min,
        "pressure_max_N": config.pressure_max,
        "shear_max_N": config.shear_max,
        "torque_tangential_max_Nm": config.torque_tangential_max,
        "torque_axial_max_Nm": config.torque_axial_max,
        "lambda_pressure": config.lambda_pressure,
        "lambda_shear": config.lambda_shear,
        "lambda_torque": config.lambda_torque,
        "lambda_axial_torque": config.lambda_axial_torque,
        "ultrasound_crop": None if config.ultrasound_crop is None else list(config.ultrasound_crop),
        "before": None if result.before_objective is None else result.before_objective.meta(),
        "verified_best": (
            None
            if result.verified_best_objective is None
            else result.verified_best_objective.meta()
        ),
        "best_x": None if result.best_x is None else result.best_x.tolist(),
        "best_F_observed": result.best_F,
        "cancelled": result.cancelled,
        "error": result.error,
        "trials": trials,
        "posterior_file": posterior_path.name,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    arrays: dict[str, np.ndarray] = {
        "observed_x": np.asarray([trial.x for trial in result.trials], dtype=float),
        "observed_F": np.asarray(
            [
                np.nan if trial.objective is None else trial.objective.F
                for trial in result.trials
            ],
            dtype=float,
        ),
    }
    for name, values in result.posterior_slices.items():
        for key, value in values.items():
            arrays[f"{name}_{key}"] = np.asarray(value, dtype=float)
    np.savez_compressed(posterior_path, **arrays)
    return summary_path, posterior_path
