from __future__ import annotations

import numpy as np

from breast_path_planning.geodesic_path import GeodesicPathParams, resample_path_with_surface_geodesics
from breast_path_planning.path_io import PlannedPath
from breast_path_planning.surface_processing import estimate_normals


GUI_GEODESIC_SURFACE_NORMAL_K_NEIGHBORS = 20

GUI_GEODESIC_PARAMS = GeodesicPathParams(
    max_iterations=5000,
    fidelity_weight=50000000,
    initial_temperature=1,
    cooling_rate=0.995,
    perturbation_radius_m=0.01,
    max_candidate_step_m=0.0075,
    corner_perturbation_scale=0.1,
    random_seed=0,
    energy_record_interval=10,
)


def optimize_gui_planned_path_geodesic(
    source_path: PlannedPath,
    surface_points_base: np.ndarray,
    *,
    surface_normals_base: np.ndarray | None = None,
    params: GeodesicPathParams = GUI_GEODESIC_PARAMS,
) -> PlannedPath:
    points = np.asarray(surface_points_base, dtype=float)
    if surface_normals_base is None:
        normals = estimate_normals(
            points,
            k_neighbors=GUI_GEODESIC_SURFACE_NORMAL_K_NEIGHBORS,
        )
    else:
        normals = np.asarray(surface_normals_base, dtype=float)
    return resample_path_with_surface_geodesics(
        source_path,
        points,
        surface_normals_base=normals,
        params=params,
        metadata={"geodesic_trigger": "gui_optimize_geodesic"},
    )
