#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
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
from breast_path_planning.path_io import load_planned_path  # noqa: E402
from breast_path_planning.pointcloud_from_d405 import load_point_cloud_ply  # noqa: E402
from breast_path_planning.surface_processing import estimate_normals  # noqa: E402


# 直接改这里。默认读取你 0615 采集的 19 组优化前/后路径。
INPUT_DIRS: list[str] = []
INPUT_GLOB = "breast_path_planning/results/live_gui_0615_*"
OUTPUT_DIR = "EXPERIMENT/geodesic_progress_figure"
# False means: if geodesic_progress_metrics.json already exists, only redraw figures.
RERUN_OPTIMIZATION = False

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
NORMAL_K_NEIGHBORS = 20
COMMON_PROGRESS_POINTS = 501
SAVE_SEPARATE_PANELS = True

INDIVIDUAL_COLOR = "#B8B8B8"
MEAN_COLOR = "#B2182B"
BAND_COLOR = "#F4A3A8"
BASELINE_COLOR = "#555555"
AXIS_TICK_LABEL_SIZE = 6.5
AXIS_ARROW_COLOR = "#222222"


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "geodesic_progress_metrics.json"

    if RERUN_OPTIMIZATION or not metrics_path.exists():
        input_dirs = resolve_input_dirs()
        if not input_dirs:
            raise FileNotFoundError(
                "No valid input directories found. Set INPUT_DIRS or INPUT_GLOB at the top of this file."
            )
        print(f"Found {len(input_dirs)} path directories.")
        runs = [compute_progress_metrics(input_dir) for input_dir in input_dirs]
        write_json(metrics_path, {"params": params_dict(GUI_GEODESIC_PARAMS), "runs": runs})
    else:
        print(f"Loading existing metrics: {metrics_path}")
        runs = json.loads(metrics_path.read_text(encoding="utf-8"))["runs"]

    output_stem = output_dir / "geodesic_progress_figure"
    save_separate_panels(runs, output_stem)
    print(f"Saved individual panels under: {output_stem.parent / 'panels'}")


def resolve_input_dirs() -> list[Path]:
    if INPUT_DIRS:
        candidates = [Path(value).expanduser() for value in INPUT_DIRS]
    else:
        candidates = sorted(Path().glob(INPUT_GLOB))
    return [
        path
        for path in candidates
        if (path / "segmented_breast.ply").exists()
        and (path / "planned_path_before_geodesic.json").exists()
    ]


def compute_progress_metrics(input_dir: Path) -> dict[str, Any]:
    print(f"Processing {input_dir} ...", flush=True)
    cloud = load_point_cloud_ply(input_dir / "segmented_breast.ply")
    initial_path = load_planned_path(input_dir / "planned_path_before_geodesic.json")
    surface_normals = estimate_normals(cloud.points_base, k_neighbors=NORMAL_K_NEIGHBORS)

    records: list[dict[str, float | int]] = []
    snapshots: list[np.ndarray] = []

    def on_progress(record: dict[str, float | int]) -> None:
        records.append(dict(record))

    def on_snapshot(_record: dict[str, float | int], positions: np.ndarray) -> None:
        snapshots.append(np.asarray(positions, dtype=float).copy())

    optimized = resample_path_with_surface_geodesics(
        initial_path,
        cloud.points_base,
        surface_normals_base=surface_normals,
        params=GUI_GEODESIC_PARAMS,
        metadata={"geodesic_trigger": "batch_progress_figure"},
        progress_callback=on_progress,
        path_snapshot_callback=on_snapshot,
    )
    if len(records) != len(snapshots):
        raise RuntimeError(f"Progress/snapshot count mismatch for {input_dir}")

    initial_positions = snapshots[0]
    normal_lookup = SurfaceNormalLookup(cloud.points_base, surface_normals)
    kg0 = discrete_geodesic_curvatures(initial_positions, normal_lookup.normals_at(initial_positions))
    max_kg0 = max(float(np.max(np.abs(kg0))) if kg0.size else 0.0, 1e-12)

    progress = []
    total_ratio = []
    curvature_ratio = []
    max_kg_ratio = []
    mean_displacement_mm = []
    for record, positions in zip(records, snapshots, strict=True):
        progress.append(float(record["iteration"]) / float(GUI_GEODESIC_PARAMS.max_iterations))
        total_ratio.append(float(record["total"]))
        curvature_ratio.append(float(record["curvature"]))
        normals = normal_lookup.normals_at(positions)
        kg = discrete_geodesic_curvatures(positions, normals)
        max_kg = float(np.max(np.abs(kg))) if kg.size else 0.0
        max_kg_ratio.append(max_kg / max_kg0)
        mean_displacement_mm.append(float(np.mean(np.linalg.norm(positions - initial_positions, axis=1))) * 1000.0)

    total_ratio = normalize_to_initial(np.asarray(total_ratio, dtype=float)).tolist()
    curvature_ratio = normalize_to_initial(np.asarray(curvature_ratio, dtype=float)).tolist()
    print(
        "  E ratio {:.3f} -> {:.3f}, curvature ratio {:.3f} -> {:.3f}, "
        "max|kg| ratio {:.3f} -> {:.3f}, mean displacement {:.2f} mm".format(
            total_ratio[0],
            total_ratio[-1],
            curvature_ratio[0],
            curvature_ratio[-1],
            max_kg_ratio[0],
            max_kg_ratio[-1],
            mean_displacement_mm[-1],
        ),
        flush=True,
    )

    return {
        "input_dir": str(input_dir),
        "progress": progress,
        "E_over_E0": total_ratio,
        "curvature_energy_over_initial": curvature_ratio,
        "max_abs_kg_over_initial": max_kg_ratio,
        "mean_displacement_mm": mean_displacement_mm,
        "accepted_moves": int(optimized.metadata["geodesic_sa_accepted_moves"]),
        "rejected_large_steps": int(optimized.metadata["geodesic_rejected_large_steps"]),
    }


class SurfaceNormalLookup:
    def __init__(self, points: np.ndarray, normals: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.normals = np.asarray(normals, dtype=float)
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.points)
        except Exception:
            self._tree = None

    def normals_at(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if self._tree is not None:
            _dist, indices = self._tree.query(values, k=1)
            return self.normals[np.asarray(indices, dtype=int)]
        delta = values[:, None, :] - self.points[None, :, :]
        indices = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        return self.normals[indices]


def normalize_to_initial(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=float)
    baseline = float(finite[0])
    if abs(baseline) < 1e-12:
        baseline = 1.0
    return array / baseline


def aggregate_metric_curves(
    runs: list[dict[str, Any]],
    metric_key: str,
    *,
    num_grid: int = COMMON_PROGRESS_POINTS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    grid = np.linspace(0.0, 1.0, int(num_grid))
    curves = []
    for run in runs:
        progress = np.asarray(run["progress"], dtype=float)
        values = np.asarray(run[metric_key], dtype=float)
        if progress.shape != values.shape:
            raise ValueError(f"Progress and {metric_key} shapes differ for {run.get('input_dir')}")
        order = np.argsort(progress)
        curves.append(np.interp(grid, progress[order], values[order]))
    stack = np.vstack(curves)
    stats = {
        "mean": np.mean(stack, axis=0),
        "q25": np.percentile(stack, 25, axis=0),
        "q75": np.percentile(stack, 75, axis=0),
    }
    return grid, stack, stats


def plot_progress_figure(runs: list[dict[str, Any]], output_stem: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)

    panels = [
        ("E_over_E0", r"$E/E_0$", "A"),
        ("curvature_energy_over_initial", r"$\sum_i \kappa_{g,i}^2 / \sum_i \kappa_{g,i,0}^2$", "B"),
        ("max_abs_kg_over_initial", r"$\max_i|\kappa_{g,i}| / \max_i|\kappa_{g,i,0}|$", "C"),
        ("mean_displacement_mm", r"$\bar{d}$ (mm)", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)

    for axis, (metric_key, ylabel, label) in zip(axes.ravel(), panels, strict=True):
        grid, stack, stats = aggregate_metric_curves(runs, metric_key)
        axis.fill_between(grid, stats["q25"], stats["q75"], color=BAND_COLOR, alpha=0.35, linewidth=0)
        for curve in stack:
            axis.plot(grid, curve, color=INDIVIDUAL_COLOR, alpha=0.28, linewidth=0.75)
        axis.plot(grid, stats["mean"], color=MEAN_COLOR, linewidth=2.2)
        baseline = 0.0 if metric_key == "mean_displacement_mm" else 1.0
        axis.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=0.8, alpha=0.7)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylabel(ylabel)
        axis.text(
            -0.12,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
        axis.grid(True, color="#E6E6E6", linewidth=0.5, alpha=0.8)

    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized optimization progress")

    axes[0, 0].plot([], [], color=INDIVIDUAL_COLOR, alpha=0.8, linewidth=1.0, label="Individual paths")
    axes[0, 0].plot([], [], color=MEAN_COLOR, linewidth=2.2, label="Mean")
    axes[0, 0].fill_between([], [], [], color=BAND_COLOR, alpha=0.35, label="IQR")
    axes[0, 0].legend(loc="best", fontsize=7)

    fig.tight_layout(pad=1.0)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    if SAVE_SEPARATE_PANELS:
        for metric_key, ylabel, label in panels:
            save_individual_panel(runs, metric_key, ylabel, label, individual_panel_stem(output_stem, metric_key))


def configure_matplotlib(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def individual_panel_stem(output_stem: Path, metric_key: str) -> Path:
    return output_stem.parent / "panels" / metric_key


def separate_panel_stems(output_stem: Path) -> dict[str, Path]:
    return {
        "E_over_E0": individual_panel_stem(output_stem, "E_over_E0"),
        "curvature_energy_over_initial": individual_panel_stem(
            output_stem,
            "curvature_energy_over_initial",
        ),
        "max_abs_kg_over_initial": individual_panel_stem(output_stem, "max_abs_kg_over_initial"),
        "mean_displacement_mm": individual_panel_stem(output_stem, "mean_displacement_mm"),
    }


def save_separate_panels(runs: list[dict[str, Any]], output_stem: Path) -> None:
    import matplotlib as mpl

    configure_matplotlib(mpl)
    panels = [
        ("E_over_E0", r"$\frac{E}{E_0}$", ""),
        (
            "curvature_energy_over_initial",
            r"$\frac{\sum_i \kappa_{g,i}^{2}}{\sum_i \kappa_{g,i,0}^{2}}$",
            "",
        ),
        (
            "max_abs_kg_over_initial",
            r"$\frac{\max_i |\kappa_{g,i}|}{\max_i |\kappa_{g,i,0}|}$",
            "",
        ),
        ("mean_displacement_mm", "$\\bar{d}$\n$(\\mathrm{mm})$", ""),
    ]
    stems = separate_panel_stems(output_stem)
    for metric_key, ylabel, label in panels:
        save_individual_panel(runs, metric_key, ylabel, label, stems[metric_key])


def save_individual_panel(
    runs: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    label: str,
    output_stem: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    grid, stack, stats = aggregate_metric_curves(runs, metric_key)

    fig, axis = plt.subplots(figsize=(3.5, 2.25))
    axis.fill_between(grid, stats["q25"], stats["q75"], color=BAND_COLOR, alpha=0.35, linewidth=0)
    for curve in stack:
        axis.plot(grid, curve, color=INDIVIDUAL_COLOR, alpha=0.28, linewidth=0.75)
    axis.plot(grid, stats["mean"], color=MEAN_COLOR, linewidth=2.2)
    baseline = 0.0 if metric_key == "mean_displacement_mm" else 1.0
    axis.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=0.8, alpha=0.7)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Normalized optimization progress")
    axis.set_ylabel("")
    style_axes_with_arrows(axis)
    label_x, label_y = metric_label_position(metric_key)
    axis.text(
        label_x,
        label_y,
        ylabel,
        transform=axis.transAxes,
        fontsize=metric_label_fontsize(metric_key),
        va="center",
        ha="center",
        clip_on=False,
    )
    if label:
        axis.text(
            -0.13,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
    axis.grid(False)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metric_label_position(metric_key: str) -> tuple[float, float]:
    if metric_key == "E_over_E0":
        return -0.12, 0.82
    return -0.18, 0.82


def style_axes_with_arrows(axis) -> None:
    axis.tick_params(axis="both", which="major", labelsize=AXIS_TICK_LABEL_SIZE, width=0.7, length=3.0)
    axis.spines["bottom"].set_visible(True)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_color(AXIS_ARROW_COLOR)
    axis.spines["left"].set_color(AXIS_ARROW_COLOR)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.spines["left"].set_linewidth(0.8)


def metric_label_fontsize(metric_key: str) -> float:
    if metric_key == "E_over_E0":
        return 15.0
    if metric_key in {"curvature_energy_over_initial", "max_abs_kg_over_initial"}:
        return 11.5
    return 10.0


def params_dict(params: GeodesicPathParams) -> dict[str, float | int | None]:
    return {
        "max_iterations": int(params.max_iterations),
        "fidelity_weight": float(params.fidelity_weight),
        "initial_temperature": float(params.initial_temperature),
        "cooling_rate": float(params.cooling_rate),
        "perturbation_radius_m": float(params.perturbation_radius_m),
        "max_candidate_step_m": None
        if params.max_candidate_step_m is None
        else float(params.max_candidate_step_m),
        "corner_perturbation_scale": float(params.corner_perturbation_scale),
        "random_seed": params.random_seed,
        "energy_record_interval": int(params.energy_record_interval),
        "normal_k_neighbors": int(NORMAL_K_NEIGHBORS),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
