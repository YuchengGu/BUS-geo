from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from breast_path_planning.path_io import PlannedPath, load_planned_path
from breast_path_planning.path_smoothing import b_spline_smooth_path, moving_average_smooth_path


MOVING_AVERAGE_WINDOW = 5
MOVING_AVERAGE_PASSES = 2
B_SPLINE_SMOOTHING_FACTOR = 0.0007830520230250682


def path_variant_method(path: PlannedPath) -> str:
    metadata = dict(path.metadata or {})
    explicit = metadata.get("path_variant_method")
    if explicit:
        return str(explicit)
    if metadata.get("geodesic_trigger") == "gui_optimize_geodesic" or metadata.get("geodesic_resample"):
        return "geodesic"
    return "original"


def path_variant_context(path: PlannedPath) -> dict[str, Any]:
    metadata = dict(path.metadata or {})
    method = path_variant_method(path)
    context: dict[str, Any] = {
        "path_variant_method": method,
        "path_variant_point_count": int(len(path)),
    }
    for key in (
        "moving_average_window",
        "moving_average_passes",
        "b_spline_smoothing_factor",
        "geodesic_energy_initial",
        "geodesic_energy_final",
        "geodesic_curvature_initial",
        "geodesic_curvature_final",
        "geodesic_fidelity_final",
        "geodesic_sa_accepted_moves",
        "geodesic_rejected_large_steps",
    ):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                context[f"path_variant_{key}"] = value
    return context


def original_path_for_variant(planning_session: Any) -> PlannedPath:
    output_dir = getattr(planning_session, "output_dir", None)
    if output_dir is not None:
        backup = Path(output_dir) / "planned_path_before_geodesic.json"
        if backup.exists():
            return load_planned_path(backup)
    path = planning_session.planned_path
    if path is None:
        raise RuntimeError("No planned path exists")
    return path


def apply_original_variant(source_path: PlannedPath) -> PlannedPath:
    return PlannedPath(
        positions_base=np.asarray(source_path.positions_base, dtype=float).copy(),
        normals_base=np.asarray(source_path.normals_base, dtype=float).copy(),
        metadata={**dict(source_path.metadata), "path_variant_method": "original"},
        frame=source_path.frame,
    )


def apply_moving_average_variant(
    source_path: PlannedPath,
    *,
    window: int = MOVING_AVERAGE_WINDOW,
    passes: int = MOVING_AVERAGE_PASSES,
) -> PlannedPath:
    return PlannedPath(
        positions_base=moving_average_smooth_path(source_path.positions_base, window=window, passes=passes),
        normals_base=np.asarray(source_path.normals_base, dtype=float).copy(),
        metadata={
            **dict(source_path.metadata),
            "path_variant_method": "moving_average",
            "moving_average_window": int(window),
            "moving_average_passes": int(passes),
        },
        frame=source_path.frame,
    )


def apply_b_spline_variant(
    source_path: PlannedPath,
    *,
    smoothing_factor: float = B_SPLINE_SMOOTHING_FACTOR,
) -> PlannedPath:
    return PlannedPath(
        positions_base=b_spline_smooth_path(source_path.positions_base, smoothing_factor=smoothing_factor),
        normals_base=np.asarray(source_path.normals_base, dtype=float).copy(),
        metadata={
            **dict(source_path.metadata),
            "path_variant_method": "b_spline",
            "b_spline_smoothing_factor": float(smoothing_factor),
        },
        frame=source_path.frame,
    )
