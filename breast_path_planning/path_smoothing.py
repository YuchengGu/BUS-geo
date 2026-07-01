from __future__ import annotations

import warnings

import numpy as np


def moving_average_smooth_path(points: np.ndarray, *, window: int = 5, passes: int = 1) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    if len(values) < 3:
        return values.copy()
    window_size = int(window)
    if window_size < 3:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    out = values.copy()
    for _ in range(max(1, int(passes))):
        previous = out.copy()
        for index in range(1, len(values) - 1):
            start = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            out[index] = np.mean(previous[start:end], axis=0)
        out[0] = values[0]
        out[-1] = values[-1]
    return out


def b_spline_smooth_path(points: np.ndarray, *, smoothing_factor: float = 0.0007830520230250682) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    if len(values) < 4:
        return values.copy()
    try:
        from scipy.interpolate import splev, splprep
    except ImportError:
        return moving_average_smooth_path(values, window=5, passes=2)

    u = normalized_chord_parameters(values)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tck, _ = splprep(values.T, u=u, s=float(smoothing_factor), k=min(3, len(values) - 1))
        out = np.asarray(splev(u, tck), dtype=float).T
    except Exception:
        return values.copy()
    out[0] = values[0]
    out[-1] = values[-1]
    return out


def normalized_chord_parameters(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(values))
    return cumulative / total
