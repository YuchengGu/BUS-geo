from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from breast_path_planning.geometry import normalize_rows
from breast_path_planning.path_io import PlannedPath
from breast_path_planning.surface_processing import estimate_normals


@dataclass(frozen=True)
class GeodesicEnergy:
    total: float
    curvature: float
    fidelity: float


@dataclass
class GeodesicPathParams:
    k_neighbors: int = 12
    max_iterations: int = 2000
    fidelity_weight: float = 1e-3
    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    perturbation_radius_m: float = 0.01
    max_candidate_step_m: float | None = None
    corner_perturbation_scale: float = 0.01
    min_segment_length_m: float = 3e-3
    random_seed: int | None = None
    energy_record_interval: int = 50


class _SurfaceProjector:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.points)
        except Exception:
            self._tree = None

    def nearest_index(self, query: np.ndarray) -> int:
        value = np.asarray(query, dtype=float).reshape(3)
        if self._tree is not None:
            _dist, index = self._tree.query(value, k=1)
            return int(index)
        delta = self.points - value.reshape(1, 3)
        return int(np.argmin(np.sum(delta * delta, axis=1)))


def discrete_geodesic_curvatures(positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=float)
    normal_values = normalize_rows(np.asarray(normals, dtype=float))
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {values.shape}")
    if normal_values.shape != values.shape:
        raise ValueError("normals must have the same shape as positions")
    if len(values) < 3:
        return np.zeros(0, dtype=float)

    out = np.zeros(len(values) - 2, dtype=float)
    eps = 1e-12
    for local_index, i in enumerate(range(1, len(values) - 1)):
        previous_point = values[i - 1]
        point = values[i]
        next_point = values[i + 1]
        s_minus = float(np.linalg.norm(point - previous_point))
        s_plus = float(np.linalg.norm(next_point - point))
        if s_minus < eps or s_plus < eps or s_minus + s_plus < eps:
            continue

        first = (next_point - previous_point) / (s_minus + s_plus)
        second = (2.0 / (s_minus + s_plus)) * (
            (next_point - point) / s_plus - (point - previous_point) / s_minus
        )
        speed = float(np.linalg.norm(first))
        if speed < eps:
            continue
        out[local_index] = float(np.dot(normal_values[i], np.cross(first, second)) / (speed**3))
    return out


def geodesic_path_energy(
    positions: np.ndarray,
    normals: np.ndarray,
    initial_positions: np.ndarray,
    *,
    fidelity_weight: float,
) -> GeodesicEnergy:
    values = np.asarray(positions, dtype=float)
    initial = np.asarray(initial_positions, dtype=float)
    if initial.shape != values.shape:
        raise ValueError("initial_positions must have the same shape as positions")
    kg = discrete_geodesic_curvatures(values, normals)
    curvature = float(np.sum(kg * kg))
    if len(values) < 3:
        fidelity = 0.0
    else:
        delta = values[1:-1] - initial[1:-1]
        fidelity = float(fidelity_weight) * float(np.sum(delta * delta))
    return GeodesicEnergy(total=curvature + fidelity, curvature=curvature, fidelity=fidelity)


def resample_path_with_surface_geodesics(
    source_path: PlannedPath,
    surface_points_base: np.ndarray,
    *,
    surface_normals_base: np.ndarray | None = None,
    params: GeodesicPathParams | None = None,
    metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
    path_snapshot_callback: Callable[[dict[str, float | int], np.ndarray], None] | None = None,
) -> PlannedPath:
    cfg = params or GeodesicPathParams()
    points = np.asarray(surface_points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"surface_points_base must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("surface_points_base must not be empty")
    if cfg.max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if cfg.initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive")
    if not (0.0 < cfg.cooling_rate <= 1.0):
        raise ValueError("cooling_rate must be in (0, 1]")
    if cfg.perturbation_radius_m <= 0.0:
        raise ValueError("perturbation_radius_m must be positive")
    if cfg.max_candidate_step_m is not None and cfg.max_candidate_step_m <= 0.0:
        raise ValueError("max_candidate_step_m must be positive when set")
    if cfg.corner_perturbation_scale <= 0.0:
        raise ValueError("corner_perturbation_scale must be positive")
    if cfg.min_segment_length_m < 0.0:
        raise ValueError("min_segment_length_m must be non-negative")

    if surface_normals_base is None:
        normals = estimate_normals(points)
    else:
        normals = normalize_rows(np.asarray(surface_normals_base, dtype=float))
        if normals.shape != points.shape:
            raise ValueError("surface_normals_base must have the same shape as surface_points_base")

    projector = _SurfaceProjector(points)
    initial_indices = np.asarray(
        [projector.nearest_index(position) for position in source_path.positions_base],
        dtype=int,
    )
    current_positions = points[initial_indices].copy()
    current_normals = normals[initial_indices].copy()
    initial_positions = current_positions.copy()
    current_energy = geodesic_path_energy(
        current_positions,
        current_normals,
        initial_positions,
        fidelity_weight=cfg.fidelity_weight,
    )
    initial_energy = current_energy

    rng = np.random.default_rng(cfg.random_seed)
    temperature = float(cfg.initial_temperature)
    accepted = 0
    rejected_large_steps = 0
    corner_indices = {
        int(index)
        for index in source_path.metadata.get("corner_indices", [])
        if 0 <= int(index) < len(source_path)
    }
    history: list[dict[str, float | int]] = [_energy_record(0, current_energy, temperature, accepted)]
    _emit_progress(history[-1], current_positions, progress_callback, path_snapshot_callback)

    if len(current_positions) >= 3:
        for iteration in range(1, int(cfg.max_iterations) + 1):
            point_index = int(rng.integers(1, len(current_positions) - 1))
            perturbation_radius = float(cfg.perturbation_radius_m)
            if point_index in corner_indices:
                perturbation_radius *= float(cfg.corner_perturbation_scale)
            proposal = current_positions[point_index] + _sample_bounded_perturbation(rng, perturbation_radius)
            surface_index = projector.nearest_index(proposal)
            candidate_step = float(np.linalg.norm(points[surface_index] - current_positions[point_index]))
            if cfg.max_candidate_step_m is not None and candidate_step > float(cfg.max_candidate_step_m):
                rejected_large_steps += 1
                temperature *= float(cfg.cooling_rate)
                if cfg.energy_record_interval > 0 and iteration % int(cfg.energy_record_interval) == 0:
                    history.append(_energy_record(iteration, current_energy, temperature, accepted))
                    _emit_progress(history[-1], current_positions, progress_callback, path_snapshot_callback)
                continue

            candidate_positions = current_positions.copy()
            candidate_normals = current_normals.copy()
            candidate_positions[point_index] = points[surface_index]
            candidate_normals[point_index] = normals[surface_index]
            if not _has_valid_adjacent_segments(candidate_positions, point_index, cfg.min_segment_length_m):
                temperature *= float(cfg.cooling_rate)
                if cfg.energy_record_interval > 0 and iteration % int(cfg.energy_record_interval) == 0:
                    history.append(_energy_record(iteration, current_energy, temperature, accepted))
                    _emit_progress(history[-1], current_positions, progress_callback, path_snapshot_callback)
                continue
            candidate_energy = geodesic_path_energy(
                candidate_positions,
                candidate_normals,
                initial_positions,
                fidelity_weight=cfg.fidelity_weight,
            )

            delta = candidate_energy.total - current_energy.total
            if delta <= 0.0 or rng.random() < float(np.exp(-delta / max(temperature, 1e-12))):
                current_positions = candidate_positions
                current_normals = candidate_normals
                current_energy = candidate_energy
                accepted += 1

            temperature *= float(cfg.cooling_rate)
            if cfg.energy_record_interval > 0 and iteration % int(cfg.energy_record_interval) == 0:
                history.append(_energy_record(iteration, current_energy, temperature, accepted))
                _emit_progress(history[-1], current_positions, progress_callback, path_snapshot_callback)

    if not history or history[-1]["iteration"] != int(cfg.max_iterations):
        history.append(_energy_record(int(cfg.max_iterations), current_energy, temperature, accepted))
        _emit_progress(history[-1], current_positions, progress_callback, path_snapshot_callback)

    path_metadata = dict(source_path.metadata)
    path_metadata.update(
        {
            "planner": "geodesic_energy_sa_v1",
            "geodesic_resample": True,
            "geodesic_source_path_points": len(source_path),
            "geodesic_output_points": len(current_positions),
            "geodesic_energy_initial": float(initial_energy.total),
            "geodesic_curvature_initial": float(initial_energy.curvature),
            "geodesic_fidelity_initial": float(initial_energy.fidelity),
            "geodesic_energy_final": float(current_energy.total),
            "geodesic_curvature_final": float(current_energy.curvature),
            "geodesic_fidelity_final": float(current_energy.fidelity),
            "geodesic_sa_iterations": int(cfg.max_iterations),
            "geodesic_sa_accepted_moves": int(accepted),
            "geodesic_sa_initial_temperature": float(cfg.initial_temperature),
            "geodesic_sa_cooling_rate": float(cfg.cooling_rate),
            "geodesic_fidelity_weight": float(cfg.fidelity_weight),
            "geodesic_perturbation_radius_m": float(cfg.perturbation_radius_m),
            "geodesic_max_candidate_step_m": (
                None if cfg.max_candidate_step_m is None else float(cfg.max_candidate_step_m)
            ),
            "geodesic_corner_perturbation_scale": float(cfg.corner_perturbation_scale),
            "geodesic_rejected_large_steps": int(rejected_large_steps),
            "geodesic_min_segment_length_m": float(cfg.min_segment_length_m),
            "geodesic_energy_history": history,
        }
    )
    if metadata:
        path_metadata.update(metadata)

    return PlannedPath(
        current_positions,
        current_normals,
        metadata=path_metadata,
        frame=source_path.frame,
    )


def _has_valid_adjacent_segments(positions: np.ndarray, point_index: int, min_length: float) -> bool:
    if min_length <= 0.0:
        return True
    values = np.asarray(positions, dtype=float)
    i = int(point_index)
    if i > 0 and float(np.linalg.norm(values[i] - values[i - 1])) < float(min_length):
        return False
    if i < len(values) - 1 and float(np.linalg.norm(values[i + 1] - values[i])) < float(min_length):
        return False
    return True


def _emit_progress(
    record: dict[str, float | int],
    positions: np.ndarray,
    progress_callback: Callable[[dict[str, float | int]], None] | None,
    path_snapshot_callback: Callable[[dict[str, float | int], np.ndarray], None] | None,
) -> None:
    if progress_callback is not None:
        progress_callback(dict(record))
    if path_snapshot_callback is not None:
        path_snapshot_callback(dict(record), np.asarray(positions, dtype=float).copy())


def _sample_bounded_perturbation(rng: np.random.Generator, radius: float) -> np.ndarray:
    value = float(radius)
    if value <= 0.0:
        raise ValueError("radius must be positive")
    direction = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.zeros(3, dtype=float)
    # Uniform in a 3-D ball, so radius is a hard upper bound rather than a Gaussian sigma.
    length = value * float(rng.random() ** (1.0 / 3.0))
    return direction / norm * length


def _energy_record(
    iteration: int,
    energy: GeodesicEnergy,
    temperature: float,
    accepted: int,
) -> dict[str, float | int]:
    return {
        "iteration": int(iteration),
        "temperature": float(temperature),
        "accepted_moves": int(accepted),
        "total": float(energy.total),
        "curvature": float(energy.curvature),
        "fidelity": float(energy.fidelity),
    }
