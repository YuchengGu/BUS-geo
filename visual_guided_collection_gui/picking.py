from __future__ import annotations

import numpy as np


def project_points_to_screen(
    points: np.ndarray,
    *,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=float)], axis=1)
    view = np.asarray(view_matrix, dtype=float).reshape(4, 4)
    projection = np.asarray(projection_matrix, dtype=float).reshape(4, 4)
    clip = (projection @ view @ homog.T).T
    w = clip[:, 3]

    valid = np.isfinite(clip).all(axis=1) & (np.abs(w) > 1e-12)
    ndc = np.zeros((pts.shape[0], 3), dtype=float)
    ndc[valid] = clip[valid, :3] / w[valid, None]
    valid &= (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
    valid &= (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    valid &= (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)

    screen = np.empty((pts.shape[0], 2), dtype=float)
    screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * float(width)
    screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * float(height)
    return screen, valid


def pick_nearest_projected_point(
    points: np.ndarray,
    *,
    click_xy: tuple[float, float],
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    width: int,
    height: int,
    max_pixel_distance: float = 12.0,
) -> int | None:
    screen, valid = project_points_to_screen(
        points,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        width=width,
        height=height,
    )
    if not bool(np.any(valid)):
        return None

    click = np.asarray(click_xy, dtype=float).reshape(2)
    distances = np.linalg.norm(screen - click, axis=1)
    distances[~valid] = np.inf
    index = int(np.argmin(distances))
    if float(distances[index]) > float(max_pixel_distance):
        return None
    return index

