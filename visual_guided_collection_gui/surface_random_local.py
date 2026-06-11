from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.surface_teleop import (
    SurfaceTeleopState,
    build_tcp_target,
    path_tangents,
)


@dataclass
class RandomLocalStartTarget:
    index: int
    tip_position_base: np.ndarray
    tcp_pose_base: np.ndarray
    normal_base: np.ndarray
    frame_axis_mode: str
    meta: dict[str, Any]


def random_local_start_target(
    path: PlannedPath,
    *,
    tip_height_m: float,
    probe_length_m: float,
    index: int | None = None,
    frame_axis_mode: str = "world-y",
    rng: np.random.Generator | None = None,
) -> RandomLocalStartTarget:
    positions = np.asarray(path.positions_base, dtype=float)
    normals = np.asarray(path.normals_base, dtype=float)
    if len(path) < 1:
        raise ValueError("path must contain at least one point")
    if index is None:
        generator = rng if rng is not None else np.random.default_rng()
        high_exclusive = max(1, len(path) - 1)
        index = int(generator.integers(0, high_exclusive))
    index = int(index)
    if index < 0 or index >= len(path):
        raise ValueError(f"random local start index {index} is outside path length {len(path)}")
    if len(path) >= 2 and index == len(path) - 1:
        raise ValueError("random local start must not use the final path point")

    tangents = path_tangents(positions)
    target = build_tcp_target(
        positions[index],
        tangents[index],
        normals[index],
        SurfaceTeleopState(normal_offset_m=float(tip_height_m)),
        probe_length_m=float(probe_length_m),
        frame_axis_mode=frame_axis_mode,
    )
    tcp_pose = target.tcp_pose_rotvec()
    meta = {
        "episode_mode": "random_local",
        "random_start_index": index,
        "random_start_position_base": positions[index].tolist(),
        "random_start_normal_base": normals[index].tolist(),
        "random_start_tcp_pose_base": tcp_pose.tolist(),
        "random_start_tip_position_base": target.probe_tip_position_base.tolist(),
        "random_start_tip_height_m": float(tip_height_m),
    }
    return RandomLocalStartTarget(
        index=index,
        tip_position_base=target.probe_tip_position_base,
        tcp_pose_base=tcp_pose,
        normal_base=normals[index],
        frame_axis_mode=target.frame_axis_mode,
        meta=meta,
    )
