from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


class RadiusSearchIndex:
    def __init__(self, points: np.ndarray, radius: float):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError(f"points must have shape (N, D), got {self.points.shape}")
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self._cells: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, point in enumerate(self.points):
            self._cells[self._cell(point)].append(index)

    def _cell(self, point: np.ndarray) -> tuple[int, ...]:
        return tuple(np.floor(point / self.radius).astype(int).tolist())

    def _neighbor_cells(self, cell: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
        ranges = [range(c - 1, c + 2) for c in cell]
        if len(cell) == 2:
            for x in ranges[0]:
                for y in ranges[1]:
                    yield (x, y)
        elif len(cell) == 3:
            for x in ranges[0]:
                for y in ranges[1]:
                    for z in ranges[2]:
                        yield (x, y, z)
        else:
            raise ValueError("RadiusSearchIndex only supports 2D or 3D points")

    def query_radius(self, index: int) -> list[int]:
        point = self.points[int(index)]
        candidates: list[int] = []
        for cell in self._neighbor_cells(self._cell(point)):
            candidates.extend(self._cells.get(cell, []))
        if not candidates:
            return []
        candidate_array = np.asarray(candidates, dtype=int)
        delta = self.points[candidate_array] - point
        keep = np.sum(delta * delta, axis=1) <= self.radius * self.radius
        return candidate_array[keep].tolist()


def nearest_indices_2d(points_xy: np.ndarray, query_xy: np.ndarray, max_distance_m: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    queries = np.asarray(query_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2), got {points.shape}")
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError(f"query_xy must have shape (M, 2), got {queries.shape}")
    if len(points) == 0 or len(queries) == 0:
        return np.zeros(0, dtype=int)

    index = RadiusSearchIndex(points, max_distance_m)
    out = []
    for query in queries:
        candidates: list[int] = []
        for cell in index._neighbor_cells(index._cell(query)):
            candidates.extend(index._cells.get(cell, []))
        if not candidates:
            out.append(-1)
            continue
        candidate_array = np.asarray(candidates, dtype=int)
        delta = points[candidate_array] - query
        dist2 = np.sum(delta * delta, axis=1)
        best_local = int(np.argmin(dist2))
        best_dist = float(np.sqrt(dist2[best_local]))
        out.append(int(candidate_array[best_local]) if best_dist <= max_distance_m else -1)
    return np.asarray(out, dtype=int)


def knn_indices(points: np.ndarray, k: int, batch_size: int = 512) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"points must have shape (N, D), got {values.shape}")
    if len(values) == 0:
        return np.zeros((0, 0), dtype=int)
    neighbors = max(1, min(int(k), len(values)))
    result = np.empty((len(values), neighbors), dtype=int)
    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        delta = values[start:end, None, :] - values[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        result[start:end] = np.argsort(dist2, axis=1)[:, :neighbors]
    return result

