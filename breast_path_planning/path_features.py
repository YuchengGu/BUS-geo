from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breast_path_planning.geometry import normalize_vector
from breast_path_planning.path_io import PlannedPath


@dataclass
class PathFeatureParams:
    lookahead: int = 8
    backtrack: int = 3
    forward: int = 7


def compute_path_features(
    path: PlannedPath,
    probe_tip_position_base: np.ndarray,
    last_index: int | None = None,
    params: PathFeatureParams | None = None,
) -> dict[str, np.ndarray | int | float]:
    if params is None:
        params = PathFeatureParams()
    if params.lookahead <= 0:
        raise ValueError("lookahead must be positive")
    probe = np.asarray(probe_tip_position_base, dtype=float).reshape(3)
    positions = path.positions_base

    if last_index is None:
        start = 0
        end = len(path)
    else:
        start = max(0, int(last_index) - params.backtrack)
        end = min(len(path), int(last_index) + params.forward + 1)
    local = positions[start:end]
    nearest_local = int(np.argmin(np.linalg.norm(local - probe, axis=1)))
    nearest = start + nearest_local

    indices = nearest + np.arange(params.lookahead)
    mask = indices < len(path)
    clipped = np.clip(indices, 0, len(path) - 1)
    target_positions = positions[clipped]
    target_normals = path.normals_base[clipped]
    target_tangents = _path_tangents(positions)[clipped]
    target_tcp_rotvecs = _tcp_reference_rotvecs(target_tangents, target_normals)
    residuals = target_positions - probe
    progress = 0.0 if len(path) <= 1 else float(nearest / (len(path) - 1))
    return {
        "path_nearest_index": int(nearest),
        "path_progress": progress,
        "path_indices": clipped.astype(np.int64),
        "path_lookahead_mask": mask.astype(bool),
        "path_target_positions_base": target_positions,
        "path_residuals_base": residuals,
        "path_normals_base": target_normals,
        "path_reference_tcp_rotvecs_base": target_tcp_rotvecs,
        "path_distance_to_nearest_m": float(np.linalg.norm(positions[nearest] - probe)),
    }


def _path_tangents(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if len(points) == 1:
        return np.tile(np.array([[1.0, 0.0, 0.0]]), (1, 1))
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    return np.vstack([normalize_vector(t, fallback=np.array([1.0, 0.0, 0.0])) for t in tangents])


def _tcp_reference_rotvecs(tangents: np.ndarray, normals: np.ndarray) -> np.ndarray:
    rotvecs = []
    for tangent, normal in zip(tangents, normals):
        n = normalize_vector(normal, fallback=np.array([0.0, 0.0, 1.0]))
        t = normalize_vector(tangent, fallback=np.array([1.0, 0.0, 0.0]))
        world_y = np.array([0.0, 1.0, 0.0], dtype=float)
        fallback = t - np.dot(t, n) * n
        if np.linalg.norm(fallback) < 1e-12:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            fallback = fallback - np.dot(fallback, n) * n
        x_axis = normalize_vector(world_y - np.dot(world_y, n) * n, fallback=fallback)
        z_axis = -n
        y_axis = normalize_vector(np.cross(z_axis, x_axis), fallback=np.array([1.0, 0.0, 0.0]))
        x_axis = normalize_vector(np.cross(y_axis, z_axis), fallback=x_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        rotvecs.append(_matrix_to_rotvec(rotation))
    return np.asarray(rotvecs, dtype=float)


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_angle = (float(np.trace(value)) - 1.0) / 2.0
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3, dtype=float)
    if np.pi - angle < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(value) + 1.0, 0.0) / 2.0)
        if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > 1e-12:
            axis[1] = value[0, 1] / (2.0 * axis[0])
            axis[2] = value[0, 2] / (2.0 * axis[0])
        elif axis[1] >= axis[2] and axis[1] > 1e-12:
            axis[0] = value[0, 1] / (2.0 * axis[1])
            axis[2] = value[1, 2] / (2.0 * axis[1])
        elif axis[2] > 1e-12:
            axis[0] = value[0, 2] / (2.0 * axis[2])
            axis[1] = value[1, 2] / (2.0 * axis[2])
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        return normalize_vector(axis, fallback=np.array([1.0, 0.0, 0.0])) * angle
    axis = np.array(
        [
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ],
        dtype=float,
    ) / (2.0 * np.sin(angle))
    return axis * angle
