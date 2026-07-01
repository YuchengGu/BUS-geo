#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache_geodesic_boxplots"))

import numpy as np


# 直接改这里。默认读取 0615 批量路径。
INPUT_DIRS: list[str] = []
INPUT_GLOB = "breast_path_planning/results/live_gui_0615_*"
# RERUN_ANALYSIS=False 时，只需要这里已有的 geodesic_boxplot_summary.json。
OUTPUT_DIR = "geodesic_boxplots"
RERUN_ANALYSIS = False
MEASURE_GEODESIC_RUNTIME = True

SMOOTHING_WINDOW = 5
SMOOTHING_PASSES = 2
NORMAL_K_NEIGHBORS = 20
GUI_GEODESIC_PARAM_VALUES = {
    "max_iterations": 5000,
    "fidelity_weight": 50000000,
    "initial_temperature": 1,
    "cooling_rate": 0.995,
    "perturbation_radius_m": 0.01,
    "max_candidate_step_m": 0.0075,
    "corner_perturbation_scale": 0.1,
    "random_seed": 0,
    "energy_record_interval": 10,
}

METHOD_LABELS = {
    "original": "Original\n(Clinical)",
    "moving_average": "Moving\naverage",
    "b_spline": "B-spline",
    "geodesic": "Geodesic\n(Ours)",
}
METHOD_COLORS = {
    "original": "#D95F5F",
    "moving_average": "#8C8C8C",
    "b_spline": "#4C78A8",
    "geodesic": "#4DAF7C",
}
RUNTIME_METHOD_LABELS = {
    "moving_average": "Moving average",
    "geodesic": "Geodesic\n(Ours)",
}
AXIS_COLOR = "#222222"
RUNTIME_LABEL_FONT_SIZE = 7.8
METRIC_LABEL_FONT_SIZE = 9
FIGSIZE_4_METHODS = (3.25, 2.55)
FIGSIZE_3_METHODS = (3.25, 2.55)
YLABEL_X = -0.17
BOX_WIDTH = 0.7
JITTER_STD = 0.03
XTICK_LABEL_ROTATION = 25
XTICK_LABEL_HA = "center"


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

    all_metrics_path = output_dir / "geodesic_boxplot_all_metrics_summary.json"
    if all_metrics_path.exists():
        print(f"Loading all metrics summary: {all_metrics_path}")
        plot_summary = json.loads(all_metrics_path.read_text(encoding="utf-8"))
    else:
        plot_summary = summary

    plot_all_boxplots(plot_summary["cases"], output_dir, runtime_cases=summary["cases"])
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
    from breast_path_planning.geodesic_path import resample_path_with_surface_geodesics
    from breast_path_planning.path_io import load_planned_path
    from breast_path_planning.path_smoothing import moving_average_smooth_path
    from breast_path_planning.pointcloud_from_d405 import load_point_cloud_ply
    from breast_path_planning.surface_processing import estimate_normals

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
            params=geodesic_params(),
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


def validate_matching_paths(original: Any, geodesic: Any) -> None:
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
    from breast_path_planning.geodesic_path import discrete_geodesic_curvatures

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


def plot_all_boxplots(
    cases: list[dict[str, Any]],
    output_dir: Path,
    *,
    runtime_cases: list[dict[str, Any]],
) -> None:
    stems = boxplot_output_stems(output_dir)
    plot_method_metric_boxplot(
        cases,
        metric_key="mean_abs_kg",
        ylabel="$\\overline{|\\kappa_g|}$\n$(\\mathrm{m}^{-1})$",
        output_stem=stems["mean_abs_kg"],
    )
    plot_method_metric_boxplot(
        cases,
        metric_key="max_abs_kg",
        ylabel="$\\max$\n$|\\kappa_g|$\n$(\\mathrm{m}^{-1})$",
        output_stem=stems["max_abs_kg"],
        scientific_y=True,
    )
    plot_method_metric_boxplot(
        cases,
        metric_key="mean_displacement_mm",
        ylabel="$\\bar{d}$\n$(\\mathrm{mm})$",
        output_stem=stems["mean_displacement_mm"],
        method_order=available_methods(cases, ["moving_average", "b_spline", "geodesic"]),
    )
    plot_ordinary_curvature_boxplots(cases, output_dir)


def boxplot_output_stems(output_dir: Path) -> dict[str, Path]:
    base = Path(output_dir)
    return {
        "mean_abs_kg": base / "mean_abs_kg_boxplot",
        "max_abs_kg": base / "max_abs_kg_boxplot",
        "mean_displacement_mm": base / "mean_displacement_boxplot",
    }


def plot_method_metric_boxplot(
    cases: list[dict[str, Any]],
    *,
    metric_key: str,
    ylabel: str,
    output_stem: Path,
    scientific_y: bool = False,
    method_order: list[str] | None = None,
    ylabel_x: float = YLABEL_X,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    if method_order is None:
        method_order = available_methods(cases, ["original", "moving_average", "b_spline", "geodesic"])
    values = [
        [float(case["methods"][method][metric_key]) for case in cases]
        for method in method_order
    ]
    figsize = FIGSIZE_4_METHODS if len(method_order) >= 4 else FIGSIZE_3_METHODS
    fig, axis = plt.subplots(figsize=figsize)
    draw_boxplot(axis, values, [METHOD_LABELS[method] for method in method_order], [METHOD_COLORS[m] for m in method_order])
    axis.set_ylabel(ylabel, rotation=0, labelpad=24, fontsize=METRIC_LABEL_FONT_SIZE)
    axis.yaxis.set_label_coords(ylabel_x, 0.5)
    if scientific_y:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.yaxis.get_offset_text().set_size(6.5)
    style_box_axis(axis)
    emphasize_ours_tick(axis)
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


def plot_ordinary_curvature_boxplots(cases: list[dict[str, Any]], output_dir: Path) -> None:
    if not metric_available(cases, "average_ordinary_curvature"):
        print("Skipping ordinary curvature boxplots: average_ordinary_curvature is missing from summary.")
        return
    plot_method_metric_boxplot(
        cases,
        metric_key="average_ordinary_curvature",
        ylabel=r"$\overline{\kappa}$" "\n" r"$(\mathrm{m}^{-1})$",
        output_stem=output_dir / "average_ordinary_curvature_boxplot",
        method_order=["original", "moving_average", "b_spline", "geodesic"],
    )


def plot_curvature_reduction_boxplot(
    cases: list[dict[str, Any]],
    *,
    output_stem: Path,
    method_order: list[str] | None = None,
    figsize: tuple[float, float] = (3.0, 2.55),
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    if method_order is None:
        method_order = available_methods(cases, ["moving_average", "b_spline", "geodesic"])
    else:
        method_order = available_methods(cases, method_order)
    values = [
        [
            curvature_reduction_percent(case, method)
            for case in cases
        ]
        for method in method_order
    ]
    fig, axis = plt.subplots(figsize=figsize)
    draw_boxplot(axis, values, [METHOD_LABELS[method] for method in method_order], [METHOD_COLORS[m] for m in method_order])
    axis.axhline(0.0, color=AXIS_COLOR, linewidth=0.8, linestyle="--", alpha=0.55)
    axis.set_ylabel("$\\Delta\\Sigma\\kappa_g^2$\n$(\\%)$", rotation=0, labelpad=23, fontsize=METRIC_LABEL_FONT_SIZE)
    axis.yaxis.set_label_coords(-0.14, 0.5)
    style_box_axis(axis)
    save_figure_all_formats(fig, output_stem)


def plot_curvature_reduction_efficiency_boxplot(
    cases: list[dict[str, Any]],
    *,
    output_stem: Path,
    method_order: list[str] | None = None,
    figsize: tuple[float, float] = (3.0, 2.55),
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    if method_order is None:
        method_order = available_methods(cases, ["moving_average", "b_spline", "geodesic"])
    else:
        method_order = available_methods(cases, method_order)
    values = [
        [
            curvature_reduction_percent(case, method) / max(float(case["methods"][method]["mean_displacement_mm"]), 1e-12)
            for case in cases
        ]
        for method in method_order
    ]
    fig, axis = plt.subplots(figsize=figsize)
    draw_boxplot(axis, values, [METHOD_LABELS[method] for method in method_order], [METHOD_COLORS[m] for m in method_order])
    axis.axhline(0.0, color=AXIS_COLOR, linewidth=0.8, linestyle="--", alpha=0.55)
    axis.set_ylabel("$\\Delta\\Sigma\\kappa_g^2$\n" "$(\\%/\\mathrm{mm})$", rotation=0, labelpad=26, fontsize=METRIC_LABEL_FONT_SIZE)
    axis.yaxis.set_label_coords(-0.17, 0.5)
    style_box_axis(axis)
    save_figure_all_formats(fig, output_stem)


def curvature_reduction_percent(case: dict[str, Any], method: str) -> float:
    original = float(case["methods"]["original"]["kg_squared_sum"])
    if abs(original) < 1e-12:
        return 0.0
    value = float(case["methods"][method]["kg_squared_sum"])
    return 100.0 * (1.0 - value / original)


def available_methods(cases: list[dict[str, Any]], preferred_order: list[str]) -> list[str]:
    if not cases:
        return []
    available = set(cases[0]["methods"])
    return [method for method in preferred_order if method in available]


def metric_available(cases: list[dict[str, Any]], metric_key: str) -> bool:
    if not cases:
        return False
    return all(metric_key in metrics for metrics in cases[0]["methods"].values())


def draw_boxplot(axis, values: list[list[float]], labels: list[str], colors: list[str]) -> None:
    positions = np.arange(1, len(values) + 1)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": AXIS_COLOR, "linewidth": 1.0},
        whiskerprops={"color": AXIS_COLOR, "linewidth": 0.8},
        capprops={"color": AXIS_COLOR, "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(AXIS_COLOR)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for index, (series, color) in enumerate(zip(values, colors, strict=True), start=1):
        jitter = rng.normal(0.0, JITTER_STD, size=len(series))
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
    axis.set_xticklabels(labels, rotation=XTICK_LABEL_ROTATION, ha=XTICK_LABEL_HA)


def emphasize_ours_tick(axis) -> None:
    for tick in axis.get_xticklabels():
        if "(Ours)" in tick.get_text():
            tick.set_fontweight("bold")


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
            "max_iterations": int(GUI_GEODESIC_PARAM_VALUES["max_iterations"]),
            "fidelity_weight": float(GUI_GEODESIC_PARAM_VALUES["fidelity_weight"]),
            "initial_temperature": float(GUI_GEODESIC_PARAM_VALUES["initial_temperature"]),
            "cooling_rate": float(GUI_GEODESIC_PARAM_VALUES["cooling_rate"]),
            "perturbation_radius_m": float(GUI_GEODESIC_PARAM_VALUES["perturbation_radius_m"]),
            "max_candidate_step_m": GUI_GEODESIC_PARAM_VALUES["max_candidate_step_m"],
            "corner_perturbation_scale": float(GUI_GEODESIC_PARAM_VALUES["corner_perturbation_scale"]),
            "random_seed": GUI_GEODESIC_PARAM_VALUES["random_seed"],
            "normal_k_neighbors": int(NORMAL_K_NEIGHBORS),
        },
    }


def geodesic_params() -> Any:
    from breast_path_planning.geodesic_path import GeodesicPathParams

    return GeodesicPathParams(**GUI_GEODESIC_PARAM_VALUES)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
