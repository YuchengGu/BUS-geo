#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EVOLUTIONS_DIR = SCRIPT_DIR / "geodesic_evolutions"
OUTPUT_DIR = SCRIPT_DIR / "geodesic_boxplots"
BASE_SUMMARY_PATH = OUTPUT_DIR / "geodesic_boxplot_summary.json"
SMOOTHING_WINDOW = 5
SMOOTHING_PASSES = 2
ORDINARY_SECOND_DERIVATIVE_FACTOR = 2.0
B_SPLINE_GEODESIC_SECOND_DERIVATIVE_FACTOR = 2.0
TARGET_B_SPLINE_MEDIAN_DISPLACEMENT_MM = 3.0


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    enhanced, b_spline_smoothing_factor = b_spline_augmented_metrics(Path(EVOLUTIONS_DIR), Path(BASE_SUMMARY_PATH))
    if not enhanced:
        raise FileNotFoundError(f"No path_evolution_snapshots.json files found under {EVOLUTIONS_DIR}")

    all_metrics_path = output_dir / "geodesic_boxplot_all_metrics_summary.json"
    write_json(
        all_metrics_path,
        {
            "ordinary_second_derivative_factor": ORDINARY_SECOND_DERIVATIVE_FACTOR,
            "added_method_geodesic_second_derivative_factor": B_SPLINE_GEODESIC_SECOND_DERIVATIVE_FACTOR,
            "normal_source": (
                "fixed same-index normals from planned_path_geodesic.json; copied dataset "
                "does not include planned_path_before_geodesic.json initial normals"
            ),
            "target_b_spline_median_displacement_mm": TARGET_B_SPLINE_MEDIAN_DISPLACEMENT_MM,
            "b_spline_smoothing_factor": b_spline_smoothing_factor,
            "cases": enhanced,
        },
    )

    print(f"Saved all boxplot metrics: {all_metrics_path}")


def b_spline_augmented_metrics(evolutions_dir: Path, base_summary_path: Path) -> tuple[list[dict[str, Any]], float]:
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    base_cases = {case["case"]: case for case in base_summary["cases"]}
    loaded_cases = []
    for case_dir in sorted(evolutions_dir.glob("live_gui_*")):
        if case_dir.name not in base_cases:
            continue
        snapshots = load_snapshots(case_dir)
        normals = load_fixed_normals(case_dir)
        if snapshots is None or normals is None:
            continue
        original = np.asarray(snapshots[0]["positions_base"], dtype=float)
        geodesic = np.asarray(snapshots[-1]["positions_base"], dtype=float)
        validate_same_shape(case_dir, original, geodesic)
        validate_same_shape(case_dir, original, normals)
        loaded_cases.append((case_dir, original, geodesic, normals))

    smoothing_factor = median_displacement_matched_b_spline_factor(
        [original for _case_dir, original, _geodesic, _normals in loaded_cases],
        TARGET_B_SPLINE_MEDIAN_DISPLACEMENT_MM,
    )

    cases = []
    for case_dir, original, geodesic, normals in loaded_cases:
        moving_average = moving_average_positions(original, window=SMOOTHING_WINDOW, passes=SMOOTHING_PASSES)
        b_spline = b_spline_smooth_positions(original, smoothing_factor)
        base_methods = base_cases[case_dir.name]["methods"]
        positions_by_method = {
            "original": original,
            "moving_average": moving_average,
            "b_spline": b_spline,
            "geodesic": geodesic,
        }
        methods = {
            "original": geodesic_curvature_metrics(original, normals, original, fallback=base_methods["original"]),
            "moving_average": geodesic_curvature_metrics(moving_average, normals, original, fallback=base_methods["moving_average"]),
            "b_spline": geodesic_curvature_metrics(b_spline, normals, original),
            "geodesic": geodesic_curvature_metrics(geodesic, normals, original, fallback=base_methods["geodesic"]),
        }
        for method, positions in positions_by_method.items():
            methods[method].update(ordinary_curvature_metrics(positions, original))
        cases.append(
            {
                "case": case_dir.name,
                "fixed_normals_source": str(case_dir / "planned_path_geodesic.json"),
                "methods": methods,
            }
        )
    return cases, smoothing_factor


def ordinary_curvature_metrics(positions: np.ndarray, reference_positions: np.ndarray) -> dict[str, float]:
    curvature_vectors = discrete_second_arclength_derivatives(
        positions,
        second_derivative_factor=ORDINARY_SECOND_DERIVATIVE_FACTOR,
    )
    if curvature_vectors.size == 0:
        mean_curvature = 0.0
        max_curvature = 0.0
        squared_mean = 0.0
    else:
        curvature_magnitudes = np.linalg.norm(curvature_vectors, axis=1)
        mean_curvature = float(np.mean(curvature_magnitudes))
        max_curvature = float(np.max(curvature_magnitudes))
        squared_mean = float(np.mean(curvature_magnitudes * curvature_magnitudes))
    return {
        "average_ordinary_curvature": mean_curvature,
        "mean_squared_ordinary_curvature": squared_mean,
        "max_ordinary_curvature": max_curvature,
        "mean_displacement_mm": mean_displacement_mm(positions, reference_positions),
        "max_displacement_mm": max_displacement_mm(positions, reference_positions),
    }


def geodesic_curvature_metrics(
    positions: np.ndarray,
    normals: np.ndarray,
    reference_positions: np.ndarray,
    fallback: dict[str, Any] | None = None,
) -> dict[str, float]:
    kg = discrete_geodesic_curvatures_assumed_normals(positions, normals)
    if kg.size:
        kg_squared_sum = float(np.sum(kg * kg))
        max_abs_kg = float(np.max(np.abs(kg)))
        mean_abs_kg = float(np.mean(np.abs(kg)))
    else:
        kg_squared_sum = 0.0
        max_abs_kg = 0.0
        mean_abs_kg = 0.0
    if fallback is not None:
        kg_squared_sum = float(fallback["kg_squared_sum"])
        max_abs_kg = float(fallback["max_abs_kg"])
    return {
        "kg_squared_sum": kg_squared_sum,
        "mean_abs_kg": mean_abs_kg,
        "max_abs_kg": max_abs_kg,
        "mean_displacement_mm": mean_displacement_mm(positions, reference_positions),
        "max_displacement_mm": max_displacement_mm(positions, reference_positions),
    }


def discrete_geodesic_curvatures_assumed_normals(positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=float)
    normal_values = normalize_rows(np.asarray(normals, dtype=float))
    if len(values) < 3:
        return np.empty(0, dtype=float)
    first_derivatives, second_derivatives, valid = finite_difference_derivatives(
        values,
        second_derivative_factor=B_SPLINE_GEODESIC_SECOND_DERIVATIVE_FACTOR,
    )
    if first_derivatives.size == 0:
        return np.empty(0, dtype=float)
    internal_normals = normal_values[1:-1][valid]
    cross_values = np.cross(first_derivatives, second_derivatives)
    numerator = np.einsum("ij,ij->i", internal_normals, cross_values)
    denominator = np.linalg.norm(first_derivatives, axis=1) ** 3
    valid_denominator = denominator > 1e-12
    return numerator[valid_denominator] / denominator[valid_denominator]


def finite_difference_derivatives(
    positions: np.ndarray,
    *,
    second_derivative_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(positions, dtype=float)
    previous_steps = np.linalg.norm(values[1:-1] - values[:-2], axis=1)
    next_steps = np.linalg.norm(values[2:] - values[1:-1], axis=1)
    valid = (previous_steps > 1e-12) & (next_steps > 1e-12)
    if not np.any(valid):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float), valid

    first = (values[2:][valid] - values[:-2][valid]) / (previous_steps[valid, None] + next_steps[valid, None])
    previous_tangents = (values[1:-1][valid] - values[:-2][valid]) / previous_steps[valid, None]
    next_tangents = (values[2:][valid] - values[1:-1][valid]) / next_steps[valid, None]
    second = second_derivative_factor * (next_tangents - previous_tangents) / (
        previous_steps[valid, None] + next_steps[valid, None]
    )
    return first, second, valid


def discrete_second_arclength_derivatives(
    positions: np.ndarray,
    *,
    second_derivative_factor: float,
) -> np.ndarray:
    _first, second, _valid = finite_difference_derivatives(
        positions,
        second_derivative_factor=second_derivative_factor,
    )
    return second


def moving_average_positions(positions: np.ndarray, *, window: int, passes: int) -> np.ndarray:
    values = np.asarray(positions, dtype=float).copy()
    if window <= 1 or len(values) < 3:
        return values
    radius = int(window) // 2
    for _ in range(int(passes)):
        smoothed = values.copy()
        for index in range(1, len(values) - 1):
            start = max(0, index - radius)
            stop = min(len(values), index + radius + 1)
            smoothed[index] = np.mean(values[start:stop], axis=0)
        values = smoothed
    return values


def median_displacement_matched_b_spline_factor(
    paths: list[np.ndarray],
    target_median_displacement_mm: float,
) -> float:
    if not paths or target_median_displacement_mm <= 1e-9:
        return 0.0
    low = 0.0
    high = 1e-10
    for _ in range(50):
        median_displacement = median_b_spline_displacement(paths, high)
        if median_displacement >= target_median_displacement_mm or high > 10.0:
            break
        low = high
        high *= 2.0

    best = low
    for _ in range(50):
        mid = 0.5 * (low + high)
        median_displacement = median_b_spline_displacement(paths, mid)
        if median_displacement <= target_median_displacement_mm:
            best = mid
            low = mid
        else:
            high = mid
    return best


def median_b_spline_displacement(paths: list[np.ndarray], smoothing_factor: float) -> float:
    displacements = [
        mean_displacement_mm(b_spline_smooth_positions(path, smoothing_factor), path)
        for path in paths
    ]
    return float(np.median(displacements))


def b_spline_smooth_positions(positions: np.ndarray, smoothing_factor: float) -> np.ndarray:
    try:
        from scipy.interpolate import splev, splprep
    except ImportError:
        return moving_average_positions(positions, window=SMOOTHING_WINDOW, passes=SMOOTHING_PASSES)

    values = np.asarray(positions, dtype=float)
    u = normalized_chord_parameters(values)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tck, _ = splprep(values.T, u=u, s=float(smoothing_factor), k=min(3, len(values) - 1))
        return np.asarray(splev(u, tck), dtype=float).T
    except Exception:
        return values.copy()


def normalized_chord_parameters(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=float)
    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if cumulative[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(values))
    return cumulative / cumulative[-1]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    normalized = values.copy()
    valid = norms > 1e-12
    normalized[valid] = normalized[valid] / norms[valid, None]
    return normalized


def mean_displacement_mm(positions: np.ndarray, reference_positions: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(positions) - np.asarray(reference_positions), axis=1)) * 1000.0)


def max_displacement_mm(positions: np.ndarray, reference_positions: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(np.asarray(positions) - np.asarray(reference_positions), axis=1)) * 1000.0)


def load_snapshots(case_dir: Path) -> list[dict[str, Any]] | None:
    path = case_dir / "path_evolution_snapshots.json"
    if not path.exists():
        return None
    snapshots = json.loads(path.read_text(encoding="utf-8"))
    return snapshots if snapshots else None


def load_fixed_normals(case_dir: Path) -> np.ndarray | None:
    path = case_dir / "planned_path_geodesic.json"
    if not path.exists():
        return None
    planned = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray([point["normal_base"] for point in planned["points"]], dtype=float)


def validate_same_shape(case_dir: Path, first: np.ndarray, second: np.ndarray) -> None:
    if first.shape != second.shape:
        raise ValueError(f"Point count mismatch for {case_dir}: {first.shape} vs {second.shape}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
