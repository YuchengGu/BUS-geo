from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.surface_teleop import path_arclengths


COMPARISON_MODES = ("full_joint", "darboux")


@dataclass(frozen=True)
class ComparisonSegment:
    pair_id: str
    start_index: int
    end_index: int
    start_arclength_m: float
    end_arclength_m: float
    start_position_base: np.ndarray
    end_position_base: np.ndarray
    start_normal_base: np.ndarray
    end_normal_base: np.ndarray

    @property
    def length_m(self) -> float:
        return self.end_arclength_m - self.start_arclength_m

    def context(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "segment_start_index": int(self.start_index),
            "segment_end_index": int(self.end_index),
            "segment_start_arclength_m": float(self.start_arclength_m),
            "segment_end_arclength_m": float(self.end_arclength_m),
            "segment_length_m": float(self.length_m),
            "segment_start_position_base": self.start_position_base.tolist(),
            "segment_end_position_base": self.end_position_base.tolist(),
            "segment_start_normal_base": self.start_normal_base.tolist(),
            "segment_end_normal_base": self.end_normal_base.tolist(),
        }


@dataclass(frozen=True)
class ComparisonTrial:
    participant_id: str
    pair_id: str
    teleop_mode: str
    sequence_index: int


def generate_trial_pair(
    path: PlannedPath,
    *,
    pair_number: int,
    pair_id: str | None = None,
    length_m: float = 0.05,
    rng: np.random.Generator | None = None,
) -> ComparisonSegment:
    positions = np.asarray(path.positions_base, dtype=float)
    normals = np.asarray(path.normals_base, dtype=float)
    if len(positions) < 2:
        raise ValueError("comparison path must contain at least two points")
    length = float(length_m)
    if length <= 0.0:
        raise ValueError("comparison segment length_m must be positive")
    arclengths = path_arclengths(positions)
    valid = np.flatnonzero(arclengths <= arclengths[-1] - length + 1e-12)
    if valid.size == 0:
        raise ValueError(
            f"path length {arclengths[-1]:.6f} m is shorter than requested "
            f"comparison segment {length:.6f} m"
        )
    generator = rng if rng is not None else np.random.default_rng()
    start_index = int(generator.choice(valid))
    start_s = float(arclengths[start_index])
    end_s = start_s + length
    upper = int(np.searchsorted(arclengths, end_s, side="left"))
    upper = int(np.clip(upper, 1, len(positions) - 1))
    lower = upper - 1
    interval = float(arclengths[upper] - arclengths[lower])
    ratio = 0.0 if interval <= 1e-12 else (end_s - arclengths[lower]) / interval
    end_position = positions[lower] + ratio * (positions[upper] - positions[lower])
    end_normal = _normalize(
        normals[lower] + ratio * (normals[upper] - normals[lower]),
        fallback=normals[lower],
    )
    resolved_pair_id = str(pair_id).strip() if pair_id is not None else ""
    if not resolved_pair_id:
        resolved_pair_id = f"pair_{int(pair_number):03d}"
    return ComparisonSegment(
        pair_id=resolved_pair_id,
        start_index=start_index,
        end_index=upper,
        start_arclength_m=start_s,
        end_arclength_m=end_s,
        start_position_base=positions[start_index].copy(),
        end_position_base=end_position,
        start_normal_base=_normalize(
            normals[start_index],
            fallback=np.array([0.0, 0.0, 1.0]),
        ),
        end_normal_base=end_normal,
    )


class ComparisonExperiment:
    def __init__(
        self,
        *,
        endpoint_radius_m: float = 0.005,
        progress_tolerance_m: float = 0.003,
        timeout_s: float = 60.0,
    ) -> None:
        self.endpoint_radius_m = float(endpoint_radius_m)
        self.progress_tolerance_m = float(progress_tolerance_m)
        self.timeout_s = float(timeout_s)
        self.pair: ComparisonSegment | None = None
        self.participant_id: str | None = None
        self.completed_modes: set[str] = set()
        self.sequence_index = 0
        self.active_trial: ComparisonTrial | None = None
        self.phase: str | None = None
        self.end_reason: str | None = None
        self.approach_metrics: dict[str, float] = {}

    def set_pair(self, pair: ComparisonSegment) -> None:
        if self.active_trial is not None:
            raise RuntimeError("cannot replace comparison pair during an active trial")
        self.pair = pair
        self.participant_id = None
        self.completed_modes.clear()
        self.sequence_index = 0
        self.phase = None
        self.end_reason = None

    def rename_pair(self, pair_id: str) -> None:
        if self.active_trial is not None:
            raise RuntimeError("cannot rename comparison pair during an active trial")
        if self.completed_modes:
            raise RuntimeError("cannot rename comparison pair after trials are completed")
        if self.pair is None:
            raise RuntimeError("generate a comparison trial pair first")
        value = str(pair_id).strip()
        if not value:
            raise ValueError("pair ID must not be empty")
        self.pair = replace(self.pair, pair_id=value)

    def confirm_participant(self, participant_id: str) -> None:
        if self.pair is None:
            raise RuntimeError("generate a comparison trial pair first")
        value = str(participant_id).strip()
        if not value:
            raise ValueError("participant ID must not be empty")
        if self.active_trial is not None:
            raise RuntimeError("cannot change participant during an active trial")
        self.participant_id = value
        self.completed_modes.clear()
        self.sequence_index = 0
        self.phase = None
        self.end_reason = None
        self.approach_metrics = {}

    def finish_participant(self) -> None:
        if self.active_trial is not None:
            raise RuntimeError("cannot finish participant during an active trial")
        self.participant_id = None
        self.completed_modes.clear()
        self.sequence_index = 0
        self.phase = None
        self.end_reason = None
        self.approach_metrics = {}

    def clear_pair(self) -> None:
        if self.active_trial is not None:
            raise RuntimeError("cannot clear comparison pair during an active trial")
        self.pair = None
        self.finish_participant()

    def begin_trial(self, teleop_mode: str) -> ComparisonTrial:
        if self.pair is None:
            raise RuntimeError("generate a comparison trial pair first")
        if self.participant_id is None:
            raise RuntimeError("confirm a participant first")
        mode = str(teleop_mode)
        if mode not in COMPARISON_MODES:
            raise ValueError(f"unsupported comparison teleop mode: {mode}")
        if self.active_trial is not None:
            raise RuntimeError("a comparison trial is already active")
        if mode in self.completed_modes:
            raise RuntimeError(
                f"{mode} trial is already completed for participant "
                f"{self.participant_id}"
            )
        self.sequence_index += 1
        self.active_trial = ComparisonTrial(
            participant_id=self.participant_id,
            pair_id=self.pair.pair_id,
            teleop_mode=mode,
            sequence_index=self.sequence_index,
        )
        self.phase = "approach"
        self.end_reason = None
        self.approach_metrics = {}
        return self.active_trial

    def set_approach_metrics(
        self,
        *,
        duration_s: float,
        position_error_m: float,
        orientation_error_rad: float,
    ) -> None:
        if self.active_trial is None:
            raise RuntimeError("no active comparison trial")
        self.approach_metrics = {
            "approach_duration_s": float(duration_s),
            "start_position_error_m": float(position_error_m),
            "start_orientation_error_rad": float(orientation_error_rad),
        }

    def start_scan(self) -> None:
        if self.active_trial is None:
            raise RuntimeError("no active comparison trial")
        self.phase = "scan"

    def finish_trial(self, reason: str) -> ComparisonTrial:
        if self.active_trial is None:
            raise RuntimeError("no active comparison trial")
        value = str(reason)
        if value not in {"reached", "manual", "timeout"}:
            raise ValueError(f"unsupported comparison end reason: {value}")
        trial = self.active_trial
        self.completed_modes.add(trial.teleop_mode)
        self.end_reason = value
        self.active_trial = None
        self.phase = None
        return trial

    def abort_trial(self) -> None:
        self.active_trial = None
        self.phase = None
        self.end_reason = None
        self.approach_metrics = {}

    def endpoint_reached(
        self,
        probe_tip_position_base: np.ndarray,
        *,
        nearest_arclength_m: float,
        reference_height_m: float = 0.0,
    ) -> bool:
        if self.pair is None:
            return False
        normal = _normalize(
            self.pair.end_normal_base,
            fallback=np.array([0.0, 0.0, 1.0]),
        )
        error = (
            np.asarray(probe_tip_position_base, dtype=float).reshape(3)
            - (self.pair.end_position_base + float(reference_height_m) * normal)
        )
        tangent_error = error - float(np.dot(error, normal)) * normal
        distance = float(np.linalg.norm(tangent_error))
        progress_ok = (
            float(nearest_arclength_m)
            >= self.pair.end_arclength_m - self.progress_tolerance_m
        )
        return distance <= self.endpoint_radius_m and progress_ok

    def trial_context(self, *, action_mode: str) -> dict[str, object]:
        if self.active_trial is None or self.pair is None:
            raise RuntimeError("no active comparison trial")
        return {
            "operation_mode": "comparison",
            "participant_id": self.active_trial.participant_id,
            "trial_id": (
                f"{self.active_trial.participant_id}_"
                f"{self.active_trial.pair_id}_"
                f"{self.active_trial.teleop_mode}"
            ),
            "teleop_mode": self.active_trial.teleop_mode,
            "trial_sequence_index": int(self.active_trial.sequence_index),
            "trial_phase": self.phase,
            "action_mode": str(action_mode),
            **self.pair.context(),
            **self.approach_metrics,
        }


def _normalize(value: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    fallback_value = np.asarray(fallback, dtype=float).reshape(3)
    fallback_norm = float(np.linalg.norm(fallback_value))
    if fallback_norm <= 1e-12:
        raise ValueError("cannot normalize zero vector and zero fallback")
    return fallback_value / fallback_norm
