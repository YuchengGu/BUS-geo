#!/usr/bin/env python3
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.geodesic_path import discrete_geodesic_curvatures  # noqa: E402
from breast_path_planning.path_io import PlannedPath, load_planned_path  # noqa: E402
from breast_path_planning.path_smoothing import moving_average_smooth_path  # noqa: E402


# 直接改这里。目录里需要有 planned_path_before_geodesic.json 和 planned_path.json。
# INPUT_DIRS 非空时按列表循环；否则按 INPUT_GLOB 自动找一批目录。
INPUT_DIRS: list[str] = []
INPUT_GLOB = "breast_path_planning/results/live_gui_0615_*"
INPUT_DIR: str | None = None
OUTPUT_DIR = "EXPERIMENT/geodesic_case_profiles"

BEFORE_COLOR = "#C73E3A"
AFTER_COLOR = "#1B8A3A"
SMOOTH_COLOR = "#6C6C6C"
DISPLACEMENT_COLOR = "#2C5AA0"
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
    print(f"Found {len(input_dirs)} cases.")
    for index, input_dir in enumerate(input_dirs, start=1):
        print(f"Plotting case {index}/{len(input_dirs)}: {input_dir}")
        before_path = load_planned_path(input_dir / "planned_path_before_geodesic.json")
        after_path = load_planned_path(input_dir / "planned_path.json")
        plot_case_profiles(before_path, after_path, output_dir, input_dir.name)
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
    return (
        path.is_dir()
        and (path / "planned_path_before_geodesic.json").exists()
        and (path / "planned_path.json").exists()
    )


def plot_case_profiles(
    before_path: PlannedPath,
    after_path: PlannedPath,
    output_dir: Path,
    case_name: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    configure_matplotlib(mpl)
    validate_matching_paths(before_path, after_path)

    s_before = normalized_arclength(before_path.positions_base)
    s_after = normalized_arclength(after_path.positions_base)
    smoothed_positions = moving_average_smooth_path(
        before_path.positions_base,
        window=SMOOTHING_WINDOW,
        passes=SMOOTHING_PASSES,
    )
    s_smooth = normalized_arclength(smoothed_positions)
    kg_before = np.abs(discrete_geodesic_curvatures(before_path.positions_base, before_path.normals_base))
    kg_smooth = np.abs(discrete_geodesic_curvatures(smoothed_positions, before_path.normals_base))
    kg_after = np.abs(discrete_geodesic_curvatures(after_path.positions_base, after_path.normals_base))
    displacement_smooth = pointwise_displacement_mm(before_path.positions_base, smoothed_positions)
    displacement_after = pointwise_displacement_mm(before_path.positions_base, after_path.positions_base)
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
        linestyle="--",
        label="Moving-average smooth",
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
        displacement_after,
        color=DISPLACEMENT_COLOR,
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


def draw_split_side_legends(axis) -> None:
    draw_split_legend_item(axis, x0=-0.19, x1=-0.12, y=0.96, color=BEFORE_COLOR, text="Before")
    draw_split_legend_item(axis, x0=-0.19, x1=-0.12, y=0.77, color=AFTER_COLOR, text="After")
    draw_split_legend_item(
        axis,
        x0=1.13,
        x1=1.22,
        y=0.96,
        color=DISPLACEMENT_COLOR,
        text="Displace-\nment",
        linestyle="--",
    )


def draw_split_legend_item(
    axis,
    *,
    x0: float,
    x1: float,
    y: float,
    color: str,
    text: str,
    linestyle: str = "-",
) -> None:
    x_mid = (float(x0) + float(x1)) * 0.5
    axis.plot(
        [x0, x1],
        [y, y],
        transform=axis.transAxes,
        color=color,
        linewidth=1.5,
        linestyle=linestyle,
        clip_on=False,
    )


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


def validate_matching_paths(before_path: PlannedPath, after_path: PlannedPath) -> None:
    if before_path.positions_base.shape != after_path.positions_base.shape:
        raise ValueError(
            "Before/after paths must have matching point counts for pointwise displacement, "
            f"got {before_path.positions_base.shape} and {after_path.positions_base.shape}"
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
