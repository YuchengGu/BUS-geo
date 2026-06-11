from __future__ import annotations

from typing import Any

import numpy as np

from breast_path_planning.geometry import rodrigues
from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.episode_recorder import add_probe_tip_observation


def obs_from_tcp_pose_rotvec(tcp_pose: np.ndarray) -> dict[str, np.ndarray]:
    pose = np.asarray(tcp_pose, dtype=float).reshape(6)
    rotation = rodrigues(pose[3:])
    return {
        "tcp_position_base": pose[:3].copy(),
        "tcp_x_axis_base": rotation[:, 0].copy(),
        "tcp_y_axis_base": rotation[:, 1].copy(),
        "tcp_z_axis_base": rotation[:, 2].copy(),
        "ee_pos_rotvec": pose.copy(),
    }


def probe_path_telemetry_lines(
    obs: dict[str, Any],
    path: PlannedPath | None,
    *,
    probe_tip_offset_m: float,
) -> tuple[list[str], int | None]:
    probe_obs = add_probe_tip_observation(obs, probe_tip_offset_m)
    tip = np.asarray(probe_obs["probe_tip_position_base"], dtype=float).reshape(3)
    x_axis = np.asarray(probe_obs["probe_x_axis_base"], dtype=float).reshape(3)
    y_axis = np.asarray(probe_obs["probe_y_axis_base"], dtype=float).reshape(3)
    z_axis = np.asarray(probe_obs["probe_z_axis_base"], dtype=float).reshape(3)

    lines = [
        f"Probe tip: {tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f} m",
        f"Probe x: {x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}",
        f"Probe y: {y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}",
        f"Probe z: {z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}",
    ]
    if path is None or len(path) == 0:
        return lines, None

    positions = np.asarray(path.positions_base, dtype=float)
    nearest = int(np.argmin(np.linalg.norm(positions - tip, axis=1)))
    distance_mm = float(np.linalg.norm(positions[nearest] - tip)) * 1000.0
    target = positions[nearest]
    lines.append(f"Nearest path point: {nearest}/{max(len(path) - 1, 0)}, distance: {distance_mm:.1f} mm")
    lines.append(f"Path point: {target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f} m")
    return lines, nearest
