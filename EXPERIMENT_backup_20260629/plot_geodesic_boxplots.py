#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.geodesic_path import (  # noqa: E402
    GeodesicPathParams,
    discrete_geodesic_curvatures,
    resample_path_with_surface_geodesics,
)
from breast_path_planning.path_io import PlannedPath, load_planned_path  # noqa: E402
from breast_path_planning.path_smoothing import moving_average_smooth_path  # noqa: E402
from breast_path_planning.pointcloud_from_d405 import load_point_cloud_ply  # noqa: E402
from breast_path_planning.surface_processing import estimate_normals  # noqa: E402


# 直接改这里。默认读取 0615 批量路径。
INPUT_DIRS: list[str] = []
INPUT_GLOB = "breast_path_planning/results/live_gui_0615_*"
OUTPUT_DIR = "EXPERIMENT/geodesic_boxplots/fucking_boxplots"
RERUN_ANALYSIS = False
MEASURE_GEODESIC_RUNTIME = True

SMOOTHING_WINDOW = 5
SMOOTHING_PASSES = 2
NORMAL_K_NEIGHBORS = 20
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

METHOD_LABELS = {
    "original": "Original",
    "moving_average": "Moving average",
    "geodesic": "Geodesic",
}
METHOD_COLORS = {
    "original": "#D95F5F",
    "moving_average": "#8C8C8C",
    "geodesic": "#4DAF7C",
}
RUNTIME_METHOD_LABELS = {
    "moving_average": "Moving average",
    "geodesic": "Geodesic",
}
AXIS_COLOR = "#222222"
RUNTIME_LABEL_FONT_SIZE = 7.8
METRIC_LABEL_FONT_SIZE = 10.5


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "geodesic_boxplot_summary.json"
    if RERUN_ANALYSIS or not summary_path.exists():
        input_dirs = resolve_input_dirs(INPUT_DIRS, INPUT_GLOB)
        if not input_dirs:
            raise FileNotFoundError("No valid path directories found. Set INPUT_DIRS or INPUT_GLOB.")
        print(f"Found {len(input_dirs)} cases.")
        summary = {
            "params": analysis_params(),
            "cases": [analyze_case(path, index, len(input_dirs)) for index, path in enumerate(input_dirs, start=1)],
        }
        write_json(summary_path, summary)
    else:
        print(f"Loading existing summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    plot_all_boxplots(summary["cases"], output_dir)
    print(f"Saved boxplots under: {output_dir}")


def resolve_input_dirs(configured_dirs: list[str], input_glob: str) -> list[Path]:
    if configured_dirs:
        candidates = [Path(value).expanduser() for value in configured_dirs]
    else:
        candidates = [Path(value) for value in sorted(glob.glob(input_glob))]
    return [
        path
        for path in candidates
        if path.is_dir()
        and (path / "planned_path_before_geodesic.json").exists()
        and (path / "planned_path.json").exists()
    ]


def analyze_case(input_dir: Path, case_index: int, case_count: int) -> dict[str, Any]:
    print(f"Analyzing case {case_index}/{case_count}: {input_dir}", flush=True)
    original = load_planned_path(input_dir / "planned_path_before_geodesic.json")
    geodesic = load_planned_path(input_dir / "planned_path.json")
    validate_matching_paths(original, geodesic)

    t0 = time.perf_counter()
    smooth_positions = moving_average_smooth_path(
        original.positions_base,
        window=SMOOTHING_WINDOW,
        passes=SMOOTHING_PASSES,
    )
    smooth_runtime_s = time.perf_counter() - t0

    methods = {
        "original": compute_method_metrics(
            original.positions_base,
            original.normals_base,
            original.positions_base,
        ),
        "moving_average": compute_method_metrics(
            smooth_positions,
            original.normals_base,
            original.positions_base,
        ),
        "geodesic": compute_method_metrics(
            geodesic.positions_base,
            geodesic.normals_base,
            original.positions_base,
        ),
    }

    runtimes = {"moving_average": float(smooth_runtime_s)}
    if MEASURE_GEODESIC_RUNTIME and (input_dir / "segmented_breast.ply").exists():
        cloud = load_point_cloud_ply(input_dir / "segmented_breast.ply")
        surface_normals = estimate_normals(cloud.points_base, k_neighbors=NORMAL_K_NEIGHBORS)
        t0 = time.perf_counter()
        resample_path_with_surface_geodesics(
            original,
            cloud.points_base,
            surface_normals_base=surface_normals,
            params=GUI_GEODESIC_PARAMS,
        )
        runtimes["geodesic"] = float(time.perf_counter() - t0)
    else:
        runtimes["geodesic"] = None

    return {
        "case": input_dir.name,
        "input_dir": str(input_dir),
        "methods": methods,
        "runtime_s": runtimes,
    }


def validate_matching_paths(original: PlannedPath, geodesic: PlannedPath) -> None:
    if original.positions_base.shape != geodesic.positions_base.shape:
        raise ValueError(
            f"Original/geodesic path point counts differ: "
            f"{original.positions_base.shape} vs {geodesic.positions_base.shape}"
        )


def compute_method_metrics(
    positions: np.ndarray,
    normals: np.ndarray,
    reference_positions: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(positions, dtype=float)
    normal_values = np.asarray(normals, dtype=float)
    reference = np.asarray(reference_positions, dtype=float)
    kg = discrete_geodesic_curvatures(values, normal_values)
    if kg.size:
        kg_squared_sum = float(np.sum(kg * kg))
        max_abs_kg = float(np.max(np.abs(kg)))
    else:
        kg_squared_sum = 0.0
        max_abs_kg = 0.0
    displacement = float(np.mean(np.linalg.norm(values - reference, axis=1)) * 1000.0)
    return {
        "kg_squared_sum": kg_squared_sum,
        "max_abs_kg": max_abs_kg,
        "mean_displacement_mm": displacement,
    }


def plot_all_boxplots(cases: list[dict[str, Any]], output_dir: Path) -> None:
    stems = boxplot_output_stems(output_dir)
    plot_method_metric_boxplot(
        cases,
        metric_key="kg_squared_sum",
        ylabel=r"$\Sigma \kappa_g^2$",
        output_stem=stems["kg_squared_sum"],
        scientific_y=True,
    )
    plot_method_metric_boxplot(
        cases,
        metric_key="max_abs_kg",
        ylabel="$\\max$\n$|\\kappa_g|$",
        output_stem=stems["max_abs_kg"],
    )
    plot_method_metric_boxplot(
        cases,
        metric_key="mean_displacement_mm",
        ylabel="$\\bar{d}$\n$(\\mathrm{mm})$",
        output_stem=stems["mean_displacement_mm"],
        method_order=["moving_average", "geodesic"],
    )
    plot_runtime_boxplot(cases, output_stem=stems["runtime_s"])


def boxplot_output_stems(output_dir: Path) -> dict[str, Path]:
    base = Path(output_dir)
    return {
        "kg_squared_sum": base / "kg_squared_sum_boxplot",
        "max_abs_kg": base / "max_abs_kg_boxplot",
        "mean_displacement_mm": base / "mean_displacement_boxplot",
        "runtime_s": base / "runtime_boxplot",
    }


def plot_method_metric_boxplot(
    cases: list[dict[str, Any]],
    *,
    metric_key: str,
    ylabel: str,
    output_stem: Path,
    scientific_y: bool = False,
    method_order: list[str] | None = None,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    if method_order is None:
        method_order = ["original", "moving_average", "geodesic"]
    values = [
        [float(case["methods"][method][metric_key]) for case in cases]
        for method in method_order
    ]
    fig, axis = plt.subplots(figsize=(3.3, 2.55))
    draw_boxplot(axis, values, [METHOD_LABELS[method] for method in method_order], [METHOD_COLORS[m] for m in method_order])
    axis.set_ylabel(ylabel, rotation=0, labelpad=24, fontsize=METRIC_LABEL_FONT_SIZE)
    axis.yaxis.set_label_coords(-0.12, 0.5)
    if scientific_y:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.yaxis.get_offset_text().set_size(6.5)
    style_box_axis(axis)
    save_figure_all_formats(fig, output_stem)


def plot_runtime_boxplot(cases: list[dict[str, Any]], *, output_stem: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    method_order = ["moving_average", "geodesic"]
    values = []
    labels = []
    colors = []
    for method in method_order:
        method_values = [
            float(case["runtime_s"][method])
            for case in cases
            if case["runtime_s"].get(method) is not None
        ]
        if method_values:
            values.append(method_values)
            labels.append(RUNTIME_METHOD_LABELS[method])
            colors.append(METHOD_COLORS[method])
    fig, axis = plt.subplots(figsize=(2.7, 2.55))
    draw_boxplot(axis, values, labels, colors)
    axis.set_ylabel("Time\n$(\\mathrm{s})$", rotation=0, labelpad=18, fontsize=RUNTIME_LABEL_FONT_SIZE)
    axis.yaxis.set_label_coords(-0.16, 0.5)
    style_box_axis(axis)
    save_figure_all_formats(fig, output_stem)


def draw_boxplot(axis, values: list[list[float]], labels: list[str], colors: list[str]) -> None:
    positions = np.arange(1, len(values) + 1)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": AXIS_COLOR, "linewidth": 1.0},
        whiskerprops={"color": AXIS_COLOR, "linewidth": 0.8},
        capprops={"color": AXIS_COLOR, "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": "#FFFFFF", "markeredgecolor": AXIS_COLOR},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(AXIS_COLOR)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for index, (series, color) in enumerate(zip(values, colors, strict=True), start=1):
        jitter = rng.normal(0.0, 0.035, size=len(series))
        axis.scatter(
            np.full(len(series), index, dtype=float) + jitter,
            series,
            s=8,
            color=color,
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")


def style_box_axis(axis) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=7, width=0.8, length=3.0, colors=AXIS_COLOR)
    axis.spines["left"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_color(AXIS_COLOR)


def configure_matplotlib(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure_all_formats(fig, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.9)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def analysis_params() -> dict[str, Any]:
    return {
        "smoothing_window": int(SMOOTHING_WINDOW),
        "smoothing_passes": int(SMOOTHING_PASSES),
        "measure_geodesic_runtime": bool(MEASURE_GEODESIC_RUNTIME),
        "geodesic": {
            "max_iterations": int(GUI_GEODESIC_PARAMS.max_iterations),
            "fidelity_weight": float(GUI_GEODESIC_PARAMS.fidelity_weight),
            "initial_temperature": float(GUI_GEODESIC_PARAMS.initial_temperature),
            "cooling_rate": float(GUI_GEODESIC_PARAMS.cooling_rate),
            "perturbation_radius_m": float(GUI_GEODESIC_PARAMS.perturbation_radius_m),
            "max_candidate_step_m": GUI_GEODESIC_PARAMS.max_candidate_step_m,
            "corner_perturbation_scale": float(GUI_GEODESIC_PARAMS.corner_perturbation_scale),
            "random_seed": GUI_GEODESIC_PARAMS.random_seed,
            "normal_k_neighbors": int(NORMAL_K_NEIGHBORS),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
