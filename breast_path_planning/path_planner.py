from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from breast_path_planning.geometry import normalize_rows
from breast_path_planning.path_io import PlannedPath
from breast_path_planning.spatial import nearest_indices_2d
from breast_path_planning.surface_processing import constrain_normals_to_reference, estimate_normals


@dataclass
class PathPlannerParams:
    step_y_m: float = 0.02
    step_x_m: float = 0.007
    slice_tolerance_ratio: float = 0.65
    max_query_distance_m: float | None = None
    normal_k_neighbors: int = 20
    normal_reference_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    max_normal_angle_deg: float | None = None
    min_points_per_slice: int = 3
    use_geodesic_resample: bool = False


def _nearest_xy_indices(points_xy: np.ndarray, query_xy: np.ndarray, max_distance_m: float) -> np.ndarray:
    return nearest_indices_2d(points_xy, query_xy, max_distance_m)


def plan_serpentine_path(
    points_base: np.ndarray,
    normals_base: np.ndarray | None = None,
    params: PathPlannerParams | None = None,
    metadata: dict[str, Any] | None = None,
) -> PlannedPath:
    if params is None:
        params = PathPlannerParams()
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_base must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("Cannot plan a path from an empty point cloud")
    if params.step_y_m <= 0 or params.step_x_m <= 0:
        raise ValueError("step_y_m and step_x_m must be positive")

    if normals_base is None:
        normals = estimate_normals(
            points,
            k_neighbors=params.normal_k_neighbors,
            reference_direction=params.normal_reference_direction,
        )
    else:
        normals = np.asarray(normals_base, dtype=float)
        if normals.shape != points.shape:
            raise ValueError("normals_base must have the same shape as points_base")
    if params.max_normal_angle_deg is None:
        normals = normalize_rows(normals, params.normal_reference_direction)
    else:
        normals = constrain_normals_to_reference(
            normals,
            reference_direction=params.normal_reference_direction,
            max_angle_deg=params.max_normal_angle_deg,
        )

    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    center_y = 0.5 * (min_y + max_y)
    half_y = 0.5 * (max_y - min_y) * 0.7
    min_y = center_y - half_y
    max_y = center_y + half_y
    step_y = params.step_y_m
    step_x = params.step_x_m
    tolerance = float(step_y * params.slice_tolerance_ratio)
    max_query = params.max_query_distance_m
    if max_query is None:
        max_query = max(step_x, step_y) * 2.0

    rows: list[tuple[np.ndarray, np.ndarray]] = []
    row_index = 0
    y_values = np.arange(min_y, max_y + step_y * 0.5, step_y)
    points_xy = points[:, :2]
    for y in y_values:
        row_mask = np.abs(points[:, 1] - y) <= tolerance
        row_indices = np.flatnonzero(row_mask)
        if row_indices.shape[0] < params.min_points_per_slice:
            continue

        row_points = points[row_indices]
        x_min = float(np.min(row_points[:, 0]))
        x_max = float(np.max(row_points[:, 0]))
        center_x = 0.5 * (x_min + x_max)
        half_x = 0.5 * (x_max - x_min) * 0.7
        x_min = center_x - half_x
        x_max = center_x + half_x
        if x_max < x_min:
            continue
        num = max(2, int(np.floor((x_max - x_min) / step_x)) + 1)
        x_samples = np.linspace(x_min, x_max, num)
        query_xy = np.stack([x_samples, np.full_like(x_samples, y)], axis=1)
        nearest = _nearest_xy_indices(points_xy, query_xy, max_query)
        nearest = nearest[nearest >= 0]
        if nearest.shape[0] == 0:
            continue

        # Drop repeated nearest points while preserving order.
        unique = []
        seen = set()
        for idx in nearest.tolist():
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        row_positions = points[unique]
        row_normals = normals[unique]
        order = np.argsort(row_positions[:, 0])
        if row_index % 2 == 1:
            order = order[::-1]
        rows.append((row_positions[order], row_normals[order]))
        row_index += 1

    if not rows:
        raise RuntimeError("No valid path rows generated from segmented point cloud")

    positions = np.concatenate([row[0] for row in rows], axis=0)
    path_normals = np.concatenate([row[1] for row in rows], axis=0)
    corner_indices: list[int] = []
    offset = 0
    for row_index, (row_positions, _row_normals) in enumerate(rows):
        row_len = int(len(row_positions))
        if row_len > 0 and row_index > 0:
            corner_indices.append(offset)
        if row_len > 0 and row_index < len(rows) - 1:
            corner_indices.append(offset + row_len - 1)
        offset += row_len
    corner_indices = sorted(set(corner_indices))
    path_metadata: dict[str, Any] = {
        "planner": "adaptive_slice_serpentine_v1",
        "step_y_m": params.step_y_m,
        "step_x_m": params.step_x_m,
        "slice_tolerance_ratio": params.slice_tolerance_ratio,
        "max_normal_angle_deg": params.max_normal_angle_deg,
        "normal_constrain_enabled": params.max_normal_angle_deg is not None,
        "num_rows": len(rows),
        "geodesic_resample": False,
        "contains_tangent": False,
        "contains_reverse_y": False,
        "corner_indices": corner_indices,
    }
    if metadata:
        path_metadata.update(metadata)
    return PlannedPath(positions, path_normals, metadata=path_metadata)
