from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breast_path_planning.path_io import PlannedPath


@dataclass
class PathFeatureParams:
    lookahead: int = 16
    backtrack: int = 5
    forward: int = 50


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
        "path_distance_to_nearest_m": float(np.linalg.norm(positions[nearest] - probe)),
    }

