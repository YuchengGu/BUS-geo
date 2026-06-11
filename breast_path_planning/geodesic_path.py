from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from breast_path_planning.geometry import normalize_rows
from breast_path_planning.path_io import PlannedPath
from breast_path_planning.spatial import knn_indices
from breast_path_planning.surface_processing import estimate_normals


@dataclass
class GeodesicPathParams:
    k_neighbors: int = 12


def _nearest_index(points: np.ndarray, query: np.ndarray) -> int:
    delta = points - np.asarray(query, dtype=float).reshape(1, 3)
    return int(np.argmin(np.sum(delta * delta, axis=1)))


def _build_knn_graph(points: np.ndarray, k_neighbors: int) -> list[list[tuple[int, float]]]:
    if k_neighbors < 2:
        raise ValueError("k_neighbors must be at least 2 because the nearest neighbor list includes self")
    neighbors = knn_indices(points, k_neighbors)
    graph: list[dict[int, float]] = [dict() for _ in range(len(points))]
    for src, row in enumerate(neighbors):
        for dst in row.tolist():
            if src == dst:
                continue
            weight = float(np.linalg.norm(points[src] - points[dst]))
            previous = graph[src].get(dst)
            if previous is None or weight < previous:
                graph[src][dst] = weight
                graph[dst][src] = weight
    return [list(edges.items()) for edges in graph]


def _shortest_path_indices(graph: list[list[tuple[int, float]]], source: int, sink: int) -> list[int]:
    if source == sink:
        return [source]

    dist = np.full(len(graph), np.inf, dtype=float)
    prev = np.full(len(graph), -1, dtype=int)
    dist[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, int(source))]
    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist > dist[node]:
            continue
        if node == sink:
            break
        for neighbor, weight in graph[node]:
            candidate = current_dist + weight
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                prev[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    if not np.isfinite(dist[sink]):
        raise RuntimeError("No connected surface graph path found between adjacent planned path waypoints")

    out = [int(sink)]
    node = int(sink)
    while node != source:
        node = int(prev[node])
        if node < 0:
            raise RuntimeError("Broken predecessor chain while reconstructing geodesic path")
        out.append(node)
    out.reverse()
    return out


def resample_path_with_surface_geodesics(
    source_path: PlannedPath,
    surface_points_base: np.ndarray,
    *,
    surface_normals_base: np.ndarray | None = None,
    params: GeodesicPathParams | None = None,
    metadata: dict[str, Any] | None = None,
) -> PlannedPath:
    if params is None:
        params = GeodesicPathParams()
    points = np.asarray(surface_points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"surface_points_base must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("surface_points_base must not be empty")

    if surface_normals_base is None:
        normals = estimate_normals(points)
    else:
        normals = normalize_rows(np.asarray(surface_normals_base, dtype=float))
        if normals.shape != points.shape:
            raise ValueError("surface_normals_base must have the same shape as surface_points_base")

    graph = _build_knn_graph(points, params.k_neighbors)
    waypoint_indices = [_nearest_index(points, position) for position in source_path.positions_base]

    path_indices: list[int] = []
    for start, end in zip(waypoint_indices[:-1], waypoint_indices[1:]):
        segment = _shortest_path_indices(graph, start, end)
        if path_indices and segment and path_indices[-1] == segment[0]:
            segment = segment[1:]
        path_indices.extend(segment)
    if not path_indices:
        path_indices = [waypoint_indices[0]]

    path_metadata = dict(source_path.metadata)
    path_metadata.update(
        {
            "planner": "surface_geodesic_knn_v1",
            "geodesic_resample": True,
            "geodesic_k_neighbors": int(params.k_neighbors),
            "geodesic_source_path_points": len(source_path),
            "geodesic_output_points": len(path_indices),
        }
    )
    if metadata:
        path_metadata.update(metadata)

    return PlannedPath(
        points[np.asarray(path_indices, dtype=int)],
        normals[np.asarray(path_indices, dtype=int)],
        metadata=path_metadata,
        frame=source_path.frame,
    )

