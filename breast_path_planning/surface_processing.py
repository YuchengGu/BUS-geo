from __future__ import annotations

import numpy as np

from breast_path_planning.geometry import normalize_rows, normalize_vector
from breast_path_planning.spatial import knn_indices


def estimate_normals(
    points_base: np.ndarray,
    *,
    k_neighbors: int = 20,
    reference_direction: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_base must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    if reference_direction is None:
        reference_direction = np.array([0.0, 0.0, 1.0], dtype=float)
    reference = normalize_vector(reference_direction)
    k = max(3, min(int(k_neighbors), len(points)))

    normals = np.zeros_like(points)
    neighbor_indices = knn_indices(points, k)

    for i, indices in enumerate(neighbor_indices):
        local = points[np.asarray(indices, dtype=int)]
        if local.shape[0] < 3:
            normals[i] = reference
            continue
        centered = local - np.mean(local, axis=0)
        cov = centered.T @ centered / max(local.shape[0] - 1, 1)
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        if float(np.dot(normal, reference)) < 0.0:
            normal = -normal
        normals[i] = normal
    return normalize_rows(normals, reference)


def constrain_normals_to_reference(
    normals: np.ndarray,
    *,
    reference_direction: np.ndarray | None = None,
    max_angle_deg: float = 30.0,
) -> np.ndarray:
    values = normalize_rows(normals)
    if reference_direction is None:
        reference_direction = np.array([0.0, 0.0, 1.0], dtype=float)
    reference = normalize_vector(reference_direction)
    max_angle_rad = float(np.deg2rad(max_angle_deg))
    out = np.zeros_like(values)

    for i, normal in enumerate(values):
        if float(np.dot(normal, reference)) < 0.0:
            normal = -normal
        dot = float(np.clip(np.dot(normal, reference), -1.0, 1.0))
        angle = float(np.arccos(dot))
        if angle <= max_angle_rad:
            out[i] = normal
            continue

        tangent = normal - dot * reference
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-12:
            out[i] = reference
            continue
        tangent = tangent / tangent_norm
        out[i] = np.cos(max_angle_rad) * reference + np.sin(max_angle_rad) * tangent
    return normalize_rows(out, reference)
