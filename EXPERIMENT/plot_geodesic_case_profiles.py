#!/usr/bin/env python3
from __future__ import annotations

import sys
import glob
import json
import os
import tempfile
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache_geodesic_case_profiles"))

try:
    from .compute_curvature_metrics import (  # type: ignore[import-not-found]
        b_spline_smooth_positions,
        discrete_geodesic_curvatures_assumed_normals,
        moving_average_positions,
    )
except ImportError:
    from compute_curvature_metrics import (  # noqa: E402
        b_spline_smooth_positions,
        discrete_geodesic_curvatures_assumed_normals,
        moving_average_positions,
    )

SCRIPT_DIR = Path(__file__).resolve().parent

# 直接改这里。默认读取当前复制过来的 geodesic_evolutions 数据。
# INPUT_DIRS 非空时按列表循环；否则按 INPUT_GLOB 自动找一批目录。
INPUT_DIRS: list[str] = []
INPUT_GLOB = str(SCRIPT_DIR / "geodesic_evolutions" / "live_gui_*")
INPUT_DIR: str | None = None
OUTPUT_DIR = SCRIPT_DIR / "geodesic_case_profiles"
B_SPLINE_METRICS_PATH = SCRIPT_DIR / "geodesic_boxplots" / "geodesic_boxplot_all_metrics_summary.json"

BEFORE_COLOR = "#C73E3A"
AFTER_COLOR = "#4DAF7C"
SMOOTH_COLOR = "#6C6C6C"
B_SPLINE_COLOR = "#4C78A8"
AXIS_COLOR = "#222222"
SMOOTHING_WINDOW = 5
SMOOTHING_PASSES = 2
LEGEND_FONT_SIZE = 5.0


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dirs = resolve_input_dirs(
        configured_dir=INPUT_DIR,
        configured_dirs=INPUT_DIRS,
        input_glob=INPUT_GLOB,
    )
    if not input_dirs:
        raise FileNotFoundError(
            "No valid input directories found. Set INPUT_DIRS, INPUT_DIR, or INPUT_GLOB at the top."
        )
    b_spline_smoothing_factor = load_b_spline_smoothing_factor(Path(B_SPLINE_METRICS_PATH))
    print(f"Found {len(input_dirs)} cases.")
    for index, input_dir in enumerate(input_dirs, start=1):
        print(f"Plotting case {index}/{len(input_dirs)}: {input_dir}")
        case_data = load_evolution_case(input_dir)
        plot_case_profiles(case_data, b_spline_smoothing_factor, output_dir, input_dir.name)
        stems = case_profile_output_stems(output_dir, input_dir.name)
        for stem in stems.values():
            print(f"Saved: {stem}.pdf")
            print(f"Saved: {stem}.svg")
            print(f"Saved: {stem}.png")


def resolve_input_dirs(
    *,
    configured_dir: str | None,
    configured_dirs: list[str],
    input_glob: str,
) -> list[Path]:
    if configured_dirs:
        candidates = [Path(value).expanduser() for value in configured_dirs]
    elif configured_dir and configured_dir.strip():
        candidates = [Path(configured_dir).expanduser()]
    else:
        candidates = [Path(value) for value in sorted(glob.glob(input_glob))]
    return [path for path in candidates if _is_valid_input_dir(path)]


def _is_valid_input_dir(path: Path) -> bool:
    evolution_format = (
        (path / "path_evolution_snapshots.json").exists()
        and (path / "planned_path_geodesic.json").exists()
    )
    planned_path_format = (
        (path / "planned_path_before_geodesic.json").exists()
        and (path / "planned_path.json").exists()
    )
    return path.is_dir() and (evolution_format or planned_path_format)


def plot_case_profiles(
    case_data: dict[str, np.ndarray],
    b_spline_smoothing_factor: float,
    output_dir: Path,
    case_name: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    original_positions = case_data["original_positions"]
    geodesic_positions = case_data["geodesic_positions"]
    normals = case_data["normals"]
    validate_matching_arrays(original_positions, geodesic_positions)
    validate_matching_arrays(original_positions, normals)

    s_before = normalized_arclength(original_positions)
    s_after = normalized_arclength(geodesic_positions)
    smoothed_positions = moving_average_positions(
        original_positions,
        window=SMOOTHING_WINDOW,
        passes=SMOOTHING_PASSES,
    )
    b_spline_positions = b_spline_smooth_positions(original_positions, b_spline_smoothing_factor)
    s_smooth = normalized_arclength(smoothed_positions)
    s_b_spline = normalized_arclength(b_spline_positions)
    kg_before = np.abs(discrete_geodesic_curvatures_assumed_normals(original_positions, normals))
    kg_smooth = np.abs(discrete_geodesic_curvatures_assumed_normals(smoothed_positions, normals))
    kg_b_spline = np.abs(discrete_geodesic_curvatures_assumed_normals(b_spline_positions, normals))
    kg_after = np.abs(discrete_geodesic_curvatures_assumed_normals(geodesic_positions, normals))
    displacement_smooth = pointwise_displacement_mm(original_positions, smoothed_positions)
    displacement_b_spline = pointwise_displacement_mm(original_positions, b_spline_positions)
    displacement_after = pointwise_displacement_mm(original_positions, geodesic_positions)
    stems = case_profile_output_stems(output_dir, case_name)

    fig, axis = plt.subplots(figsize=(5.8, 2.35))

    axis.plot(
        s_before[1:-1],
        kg_before,
        color=BEFORE_COLOR,
        linewidth=1.55,
        label="Original",
    )
    axis.plot(
        s_smooth[1:-1],
        kg_smooth,
        color=SMOOTH_COLOR,
        linewidth=1.35,
        label="Moving-average smooth",
    )
    axis.plot(
        s_b_spline[1:-1],
        kg_b_spline,
        color=B_SPLINE_COLOR,
        linewidth=1.35,
        label="B-spline",
    )
    axis.plot(
        s_after[1:-1],
        kg_after,
        color=AFTER_COLOR,
        linewidth=1.75,
        label="Geodesic optimized",
    )
    axis.set_ylabel(r"$|\kappa_g|$", rotation=0, labelpad=20, fontsize=13)
    axis.yaxis.set_label_coords(-0.11, 0.5)
    axis.set_xlabel("Normalized arc length")
    style_profile_axis(axis)
    axis.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, ncol=1, frameon=True)
    save_figure_all_formats(fig, stems["kg"])

    fig, axis = plt.subplots(figsize=(5.8, 2.35))
    axis.plot(
        s_before,
        displacement_smooth,
        color=SMOOTH_COLOR,
        linewidth=1.25,
        linestyle="--",
        label="Moving-average smooth",
    )
    axis.plot(
        s_before,
        displacement_b_spline,
        color=B_SPLINE_COLOR,
        linewidth=1.35,
        linestyle="--",
        label="B-spline",
    )
    axis.plot(
        s_before,
        displacement_after,
        color=AFTER_COLOR,
        linewidth=1.35,
        linestyle="--",
        label="Geodesic optimized",
    )
    axis.set_ylabel("$d$\n$(\\mathrm{mm})$", rotation=0, labelpad=22, fontsize=12)
    axis.yaxis.set_label_coords(-0.11, 0.5)
    axis.set_xlabel("Normalized arc length")
    style_profile_axis(axis)
    axis.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, ncol=1, frameon=True)
    save_figure_all_formats(fig, stems["displacement"])


def load_b_spline_smoothing_factor(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run compute_curvature_metrics.py first so case profiles use the same B-spline setting."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["b_spline_smoothing_factor"])


def load_evolution_case(case_dir: Path) -> dict[str, np.ndarray]:
    if not (case_dir / "path_evolution_snapshots.json").exists():
        from breast_path_planning.path_io import load_planned_path

        original = load_planned_path(case_dir / "planned_path_before_geodesic.json")
        geodesic = load_planned_path(case_dir / "planned_path.json")
        return {
            "original_positions": np.asarray(original.positions_base, dtype=float),
            "geodesic_positions": np.asarray(geodesic.positions_base, dtype=float),
            "normals": np.asarray(geodesic.normals_base, dtype=float),
        }
    snapshots = json.loads((case_dir / "path_evolution_snapshots.json").read_text(encoding="utf-8"))
    planned_geodesic = json.loads((case_dir / "planned_path_geodesic.json").read_text(encoding="utf-8"))
    if not snapshots:
        raise ValueError(f"No snapshots in {case_dir / 'path_evolution_snapshots.json'}")
    original_positions = np.asarray(snapshots[0]["positions_base"], dtype=float)
    geodesic_positions = np.asarray(snapshots[-1]["positions_base"], dtype=float)
    normals = np.asarray(
        [point["normal_base"] for point in planned_geodesic["points"]],
        dtype=float,
    )
    return {
        "original_positions": original_positions,
        "geodesic_positions": geodesic_positions,
        "normals": normals,
    }


def case_profile_output_stems(output_dir: Path, case_name: str) -> dict[str, Path]:
    case_dir = Path(output_dir) / str(case_name)
    return {
        "kg": case_dir / "kg",
        "displacement": case_dir / "displacement",
    }


def style_profile_axis(axis) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=7, width=0.8, length=3.0, colors=AXIS_COLOR)
    axis.spines["left"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_color(AXIS_COLOR)


def save_figure_all_formats(fig, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.9)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


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


def validate_matching_arrays(first: np.ndarray, second: np.ndarray) -> None:
    if first.shape != second.shape:
        raise ValueError(
            "Profile arrays must have matching point counts, "
            f"got {first.shape} and {second.shape}"
        )


def normalized_arclength(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    if len(values) == 0:
        return np.zeros(0, dtype=float)
    if len(values) == 1:
        return np.zeros(1, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.zeros(len(values), dtype=float)
    return cumulative / total


def pointwise_displacement_mm(before_points: np.ndarray, after_points: np.ndarray) -> np.ndarray:
    before = np.asarray(before_points, dtype=float)
    after = np.asarray(after_points, dtype=float)
    if before.shape != after.shape:
        raise ValueError(f"before/after point shapes must match, got {before.shape} and {after.shape}")
    return np.linalg.norm(after - before, axis=1) * 1000.0


if __name__ == "__main__":
    main()
