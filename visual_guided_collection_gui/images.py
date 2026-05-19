from __future__ import annotations

import numpy as np


def depth_to_display_rgb(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    if value.ndim != 2:
        raise ValueError(f"depth must have shape (H, W) or (H, W, 1), got {value.shape}")

    depth_f = value.astype(float, copy=False)
    valid = np.isfinite(depth_f) & (depth_f > 0)
    if np.any(valid):
        high = float(np.percentile(depth_f[valid], 95.0))
        low = float(np.percentile(depth_f[valid], 2.0))
        if high <= low:
            high = float(np.max(depth_f[valid]))
            low = float(np.min(depth_f[valid]))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0

    norm = np.clip((depth_f - low) / (high - low), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    try:
        import cv2

        color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        color_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    color_rgb[~valid] = 0
    return color_rgb.astype(np.uint8, copy=False)

