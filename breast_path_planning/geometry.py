from __future__ import annotations

import numpy as np


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return normalize_vector(fallback)
    return value / norm


def normalize_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {array.shape}")
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=float)
    out = array.copy()
    norms = np.linalg.norm(out, axis=1)
    good = norms > 1e-12
    out[good] = out[good] / norms[good, None]
    out[~good] = normalize_vector(fallback)
    return out


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3, dtype=float)

    axis = rv / angle
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def rotvec_pose_to_transform(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=float).reshape(-1)
    if value.shape[0] < 6:
        raise ValueError(f"Expected at least 6 pose values, got {value.shape}")
    return make_transform(rodrigues(value[3:6]), value[:3])


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {pts.shape}")
    tf = np.asarray(transform, dtype=float).reshape(4, 4)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=float)], axis=1)
    return (tf @ homog.T).T[:, :3]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))

