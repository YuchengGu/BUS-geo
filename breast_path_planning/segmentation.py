from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from breast_path_planning.pointcloud_from_d405 import PointCloud
from breast_path_planning.spatial import RadiusSearchIndex


@dataclass
class SegmentationParams:
    spatial_radius_m: float = 0.015
    hue_threshold_deg: float = 30.0
    saturation_threshold: float = 0.35
    value_threshold: float = 0.60
    max_distance_from_seed_m: float = 0.12
    max_points: int | None = None


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=float) / 255.0
    r = values[:, 0]
    g = values[:, 1]
    b = values[:, 2]
    maxc = np.max(values, axis=1)
    minc = np.min(values, axis=1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-12
    r_mask = nonzero & (maxc == r)
    g_mask = nonzero & (maxc == g)
    b_mask = nonzero & (maxc == b)
    hue[r_mask] = (60.0 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360.0) % 360.0
    hue[g_mask] = 60.0 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120.0
    hue[b_mask] = 60.0 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240.0

    saturation = np.zeros_like(maxc)
    saturation[maxc > 1e-12] = delta[maxc > 1e-12] / maxc[maxc > 1e-12]
    return np.stack([hue, saturation, maxc], axis=1)


def seed_pixels_to_indices(cloud: PointCloud, seed_pixels: Sequence[tuple[int, int]]) -> list[int]:
    if cloud.pixels_uv is None:
        raise ValueError("PointCloud must include pixels_uv to use seed pixels")
    if not seed_pixels:
        raise ValueError("At least one seed pixel is required")
    pixels = np.asarray(cloud.pixels_uv, dtype=float)
    indices = []
    for seed_u, seed_v in seed_pixels:
        seed = np.array([seed_u, seed_v], dtype=float)
        idx = int(np.argmin(np.sum((pixels - seed) ** 2, axis=1)))
        indices.append(idx)
    return indices


def _query_neighbors(points: np.ndarray, radius: float):
    index = RadiusSearchIndex(points, radius)
    return index.query_radius


def segment_region_from_seed_indices(
    cloud: PointCloud,
    seed_indices: Sequence[int],
    params: SegmentationParams | None = None,
) -> tuple[PointCloud, np.ndarray]:
    if params is None:
        params = SegmentationParams()
    if cloud.colors_rgb is None:
        raise ValueError("PointCloud must include colors_rgb for color region growing")
    if len(cloud) == 0:
        raise ValueError("Cannot segment an empty point cloud")
    seeds = [int(i) for i in seed_indices]
    if any(i < 0 or i >= len(cloud) for i in seeds):
        raise ValueError("seed index out of bounds")

    points = cloud.points_base
    hsv = rgb_to_hsv(cloud.colors_rgb)
    seed_hsv = hsv[seeds]
    ref_hue = float(np.angle(np.mean(np.exp(1j * np.deg2rad(seed_hsv[:, 0]))), deg=True) % 360.0)
    ref_sat = float(np.mean(seed_hsv[:, 1]))
    ref_val = float(np.mean(seed_hsv[:, 2]))
    seed_center = np.mean(points[seeds], axis=0)

    visited = np.zeros(len(cloud), dtype=bool)
    in_region = np.zeros(len(cloud), dtype=bool)
    queue = list(seeds)
    for idx in seeds:
        visited[idx] = True
        in_region[idx] = True

    neighbors_for = _query_neighbors(points, params.spatial_radius_m)
    while queue:
        current = queue.pop(0)
        for neighbor in neighbors_for(current):
            if visited[neighbor]:
                continue
            visited[neighbor] = True
            if np.linalg.norm(points[neighbor] - seed_center) > params.max_distance_from_seed_m:
                continue
            dh = abs(float(hsv[neighbor, 0]) - ref_hue)
            dh = min(dh, 360.0 - dh)
            ds = abs(float(hsv[neighbor, 1]) - ref_sat)
            dv = abs(float(hsv[neighbor, 2]) - ref_val)
            if (
                dh <= params.hue_threshold_deg
                and ds <= params.saturation_threshold
                and dv <= params.value_threshold
            ):
                in_region[neighbor] = True
                queue.append(neighbor)
                if params.max_points is not None and int(np.sum(in_region)) >= params.max_points:
                    return cloud.subset(in_region), in_region

    return cloud.subset(in_region), in_region


def segment_region_from_seed_pixels(
    cloud: PointCloud,
    seed_pixels: Sequence[tuple[int, int]],
    params: SegmentationParams | None = None,
) -> tuple[PointCloud, np.ndarray]:
    return segment_region_from_seed_indices(cloud, seed_pixels_to_indices(cloud, seed_pixels), params)
