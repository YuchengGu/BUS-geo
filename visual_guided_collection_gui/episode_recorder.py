from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

from breast_path_planning.path_features import PathFeatureParams, compute_path_features
from breast_path_planning.path_io import PlannedPath
from gello.data_utils.format_obs import save_frame


def add_probe_tip_observation(obs: dict[str, Any], probe_tip_offset_m: float) -> dict[str, Any]:
    out = dict(obs)
    tcp_position = np.asarray(obs["tcp_position_base"], dtype=float).reshape(3)
    tcp_x = np.asarray(obs["tcp_x_axis_base"], dtype=float).reshape(3)
    tcp_y = np.asarray(obs["tcp_y_axis_base"], dtype=float).reshape(3)
    tcp_z = np.asarray(obs["tcp_z_axis_base"], dtype=float).reshape(3)

    out["probe_tip_position_base"] = tcp_position + float(probe_tip_offset_m) * tcp_z
    out["probe_x_axis_base"] = tcp_x
    out["probe_y_axis_base"] = tcp_y
    out["probe_z_axis_base"] = tcp_z
    return out


class EpisodeRecorder:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        agent_name: str,
        planned_path: PlannedPath,
        probe_tip_offset_m: float = 0.0,
        path_feature_params: PathFeatureParams | None = None,
        episode_context: dict[str, Any] | None = None,
        record_rgb_depth: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.agent_name = agent_name
        self.planned_path = planned_path
        self.probe_tip_offset_m = float(probe_tip_offset_m)
        self.path_feature_params = path_feature_params or PathFeatureParams()
        self.episode_context = dict(episode_context or {})
        self.record_rgb_depth = bool(record_rgb_depth)
        self.episode_dir: Path | None = None
        self.sample_index = 0
        self.last_path_index: int | None = None
        self.fine_scan_flag = 0

    def start(self, episode_id: str | None = None) -> Path:
        if episode_id is None:
            episode_id = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.episode_dir = self.data_dir / self.agent_name / episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.sample_index = 0
        self.last_path_index = None
        return self.episode_dir

    def stop(self) -> None:
        self.episode_dir = None

    def set_fine_scan_flag(self, enabled: bool) -> None:
        self.fine_scan_flag = 1 if enabled else 0

    def toggle_fine_scan_flag(self) -> int:
        self.fine_scan_flag = 0 if self.fine_scan_flag else 1
        return self.fine_scan_flag

    def enrich_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        enriched = add_probe_tip_observation(obs, self.probe_tip_offset_m)
        features = compute_path_features(
            self.planned_path,
            enriched["probe_tip_position_base"],
            last_index=self.last_path_index,
            params=self.path_feature_params,
        )
        self.last_path_index = int(features["path_nearest_index"])
        reference_tcp_positions = (
            np.asarray(features["path_target_positions_base"], dtype=float)
            + self.probe_tip_offset_m * np.asarray(features["path_normals_base"], dtype=float)
        )
        features["path_reference_tcp_positions_base"] = reference_tcp_positions
        features["path_reference_tcp_poses_base"] = np.concatenate(
            [
                reference_tcp_positions,
                np.asarray(features["path_reference_tcp_rotvecs_base"], dtype=float),
            ],
            axis=1,
        )
        enriched.update(features)
        enriched["fine_scan_flag"] = int(self.fine_scan_flag)
        return enriched

    def save_sample(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        if self.episode_dir is None:
            raise RuntimeError("EpisodeRecorder.start() must be called before saving")
        enriched = self.enrich_observation(obs)
        sample_meta = dict(meta or {})
        sample_meta.update(self.episode_context)
        sample_meta["sample_index"] = self.sample_index
        sample_meta["fine_scan_flag"] = int(self.fine_scan_flag)
        sample_meta["path_nearest_index"] = int(enriched["path_nearest_index"])
        sample_meta["path_progress"] = float(enriched["path_progress"])
        sample_meta["rgb_depth_recorded"] = bool(self.record_rgb_depth)
        saved_observation = (
            enriched
            if self.record_rgb_depth
            else {
                key: value
                for key, value in enriched.items()
                if not key.endswith(("_rgb", "_depth"))
            }
        )
        save_frame(
            self.episode_dir,
            timestamp or datetime.datetime.now(),
            saved_observation,
            np.asarray(action),
            meta=sample_meta,
        )
        self.sample_index += 1
        return enriched
