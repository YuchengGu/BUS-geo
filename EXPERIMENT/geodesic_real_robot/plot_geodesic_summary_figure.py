#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_geodesic_real_robot_summary"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from EXPERIMENT.geodesic_real_robot.force_analysis import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    METHOD_ORDER,
    TrialSignals,
    compute_force_metrics,
    load_group,
    resample_by_progress,
)


BEST_GROUP = 7
COHORT_GROUPS = tuple(range(2, 9))
PROCESS_METHODS = ("original", "geodesic")
BOX_METHODS = ("moving_average", "b_spline", "geodesic")
PROCESS_POINTS = 501

BOX_FIGSIZE = (3.25, 2.55)
PROCESS_FIGSIZE = (6.50, 2.55)
COMBINED_FIGSIZE = (13.00, 5.10)

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
AXIS_COLOR = "#222222"
BOX_WIDTH = 0.7
BOX_Y_PADDING_RATIO = 0.25
JITTER_STD = 0.03


def plot_geodesic_summary_figure(
    best_trials: dict[str, TrialSignals],
    cohort_trials: dict[int, dict[str, TrialSignals]],
    output_dir: str | Path,
    *,
    best_group: int = BEST_GROUP,
) -> dict[str, tuple[Path, Path, Path]]:
    _configure_matplotlib()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    metrics = _cohort_metrics(cohort_trials)
    box_specs = _box_specs()
    outputs: dict[str, tuple[Path, Path, Path]] = {}

    combined = plt.figure(figsize=COMBINED_FIGSIZE)
    grid = combined.add_gridspec(2, 4)
    process_axes = (
        combined.add_subplot(grid[0, 0:2]),
        combined.add_subplot(grid[0, 2:4]),
    )
    box_axes = tuple(combined.add_subplot(grid[1, index]) for index in range(4))
    _draw_tangential_force(process_axes[0], best_trials, best_group=best_group, show_legend=True)
    _draw_tangential_torque(process_axes[1], best_trials, show_legend=False)
    for axis, spec in zip(box_axes, box_specs):
        _draw_metric_boxplot(axis, metrics, **spec)
    combined.tight_layout(pad=0.9, h_pad=1.2, w_pad=1.0)
    outputs["combined"] = _save_figure(
        combined,
        destination / "geodesic_real_robot_summary",
    )

    process_specs: tuple[
        tuple[str, Callable[[plt.Axes, dict[str, TrialSignals]], None]], ...
    ] = (
        (
            "process_tangential_force",
            lambda axis, trials: _draw_tangential_force(
                axis,
                trials,
                best_group=best_group,
                show_legend=True,
            ),
        ),
        (
            "process_tangential_torque",
            lambda axis, trials: _draw_tangential_torque(
                axis,
                trials,
                show_legend=True,
            ),
        ),
    )
    for name, draw in process_specs:
        fig, axis = plt.subplots(figsize=PROCESS_FIGSIZE)
        draw(axis, best_trials)
        fig.tight_layout(pad=0.9)
        outputs[name] = _save_figure(fig, destination / name)

    for spec in box_specs:
        fig, axis = plt.subplots(figsize=BOX_FIGSIZE)
        _draw_metric_boxplot(axis, metrics, **spec)
        fig.tight_layout(pad=0.9)
        outputs[str(spec["name"])] = _save_figure(
            fig,
            destination / str(spec["name"]),
        )

    return outputs


def _draw_tangential_force(
    axis: plt.Axes,
    trials: dict[str, TrialSignals],
    *,
    best_group: int,
    show_legend: bool,
) -> None:
    for method in PROCESS_METHODS:
        trial = trials[method]
        progress, raw = resample_by_progress(
            trial.progress,
            trial.tangential_force_n,
            points=PROCESS_POINTS,
        )
        color = METHOD_COLORS[method]
        axis.plot(progress, raw, color=color, linewidth=0.45, alpha=0.20)
        axis.plot(
            progress,
            _smooth_signal(raw),
            color=color,
            linewidth=1.35,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
    axis.set_xlabel("Normalized scan progress")
    _set_horizontal_ylabel(axis, "$F_t$\n$(\\mathrm{N})$", x=-0.07)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    _style_axis(axis)
    if show_legend:
        axis.legend(loc="upper right", fontsize=7, ncol=2)


def _draw_tangential_torque(
    axis: plt.Axes,
    trials: dict[str, TrialSignals],
    *,
    show_legend: bool,
) -> None:
    for method in PROCESS_METHODS:
        trial = trials[method]
        progress, raw = resample_by_progress(
            trial.progress,
            trial.tangential_torque_nm,
            points=PROCESS_POINTS,
        )
        color = METHOD_COLORS[method]
        axis.plot(progress, raw, color=color, linewidth=0.45, alpha=0.20)
        axis.plot(
            progress,
            _smooth_signal(raw),
            color=color,
            linewidth=1.35,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
    axis.set_xlabel("Normalized scan progress")
    _set_horizontal_ylabel(
        axis,
        "$\\tau_t$\n$(\\mathrm{N\\cdot m})$",
        x=-0.08,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    _style_axis(axis)
    if show_legend:
        axis.legend(loc="upper left", fontsize=7, ncol=2)


def _draw_metric_boxplot(
    axis: plt.Axes,
    metrics: dict[int, dict[str, dict[str, float | int | bool]]],
    *,
    name: str,
    key: str,
    ylabel: str,
) -> None:
    groups = sorted(metrics)
    values = [
        [float(metrics[group][method][key]) for group in groups]
        for method in BOX_METHODS
    ]
    positions = np.arange(1, len(BOX_METHODS) + 1, dtype=float)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": AXIS_COLOR, "linewidth": 1.0},
        whiskerprops={"color": AXIS_COLOR, "linewidth": 0.8},
        capprops={"color": AXIS_COLOR, "linewidth": 0.8},
    )
    for patch, method in zip(boxes["boxes"], BOX_METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.45)
        patch.set_edgecolor(AXIS_COLOR)
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for position, method, series in zip(positions, BOX_METHODS, values):
        jitter = rng.normal(0.0, JITTER_STD, size=len(series))
        axis.scatter(
            np.full(len(series), position) + jitter,
            series,
            s=8,
            color=METHOD_COLORS[method],
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [METHOD_LABELS[method] for method in BOX_METHODS],
        rotation=25,
        ha="center",
    )
    _set_horizontal_ylabel(axis, ylabel, x=-0.19)
    _set_centered_boxplot_ylim(axis, values)
    _style_axis(axis)
    _add_geodesic_median_reduction(axis, values)
    for tick in axis.get_xticklabels():
        if "(Ours)" in tick.get_text():
            tick.set_fontweight("bold")


def _add_geodesic_median_reduction(
    axis: plt.Axes,
    values: list[list[float]],
) -> None:
    if "geodesic" not in BOX_METHODS:
        return
    geodesic_index = BOX_METHODS.index("geodesic")
    geodesic_median = _finite_median(values[geodesic_index])
    if not np.isfinite(geodesic_median):
        return
    lines: list[str] = []
    short_labels = {
        "moving_average": "MA",
        "b_spline": "B-spline",
    }
    for method, series in zip(BOX_METHODS, values, strict=True):
        if method == "geodesic" or method not in short_labels:
            continue
        baseline_median = _finite_median(series)
        if not np.isfinite(baseline_median) or abs(baseline_median) < 1e-12:
            continue
        reduction = 100.0 * (baseline_median - geodesic_median) / baseline_median
        arrow = r"$\downarrow$" if reduction >= 0 else r"$\uparrow$"
        lines.append(f"vs {short_labels[method]} {arrow}{abs(reduction):.1f}%")
    if not lines:
        return
    axis.text(
        0.98,
        0.96,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=METHOD_COLORS["geodesic"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )


def _finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _cohort_metrics(
    cohort_trials: dict[int, dict[str, TrialSignals]],
) -> dict[int, dict[str, dict[str, float | int | bool]]]:
    return {
        group: {
            method: compute_force_metrics(trial)
            for method, trial in trials.items()
        }
        for group, trials in cohort_trials.items()
    }


def _box_specs() -> tuple[dict[str, str], ...]:
    return (
        {
            "name": "box_offset_variation",
            "key": "force_offset_variation_mm",
            "ylabel": "$\\mathrm{TV}(o)$\n$(\\mathrm{mm})$",
        },
        {
            "name": "box_outward_correction",
            "key": "force_offset_outward_motion_mm",
            "ylabel": "$D_{\\mathrm{lift}}$\n$(\\mathrm{mm})$",
        },
        {
            "name": "box_tangential_force_p95",
            "key": "tangential_force_p95_n",
            "ylabel": "$F_{t,95}$\n$(\\mathrm{N})$",
        },
        {
            "name": "box_tangential_torque_p95",
            "key": "torque_tangential_p95_nm",
            "ylabel": "$\\tau_{t,95}$\n$(\\mathrm{N\\,m})$",
        },
    )


def _smooth_signal(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size < 5:
        return data.copy()
    window = min(41, data.size if data.size % 2 == 1 else data.size - 1)
    if window < 5:
        return data.copy()
    return savgol_filter(data, window_length=window, polyorder=2, mode="interp")


def _set_centered_boxplot_ylim(
    axis: plt.Axes,
    values: list[list[float]],
) -> None:
    data = np.concatenate([np.asarray(series, dtype=float) for series in values])
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    span = maximum - minimum
    if span <= np.finfo(float).eps:
        span = max(abs(minimum) * 0.2, 1e-6)
    padding = BOX_Y_PADDING_RATIO * span
    axis.set_ylim(minimum - padding, maximum + padding)


def _set_horizontal_ylabel(axis: plt.Axes, text: str, *, x: float) -> None:
    axis.set_ylabel(text, rotation=0, labelpad=20, fontsize=9)
    axis.yaxis.set_label_coords(x, 0.5)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(
        axis="both",
        labelsize=7,
        width=0.8,
        length=3.0,
        colors=AXIS_COLOR,
    )
    axis.spines["left"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_color(AXIS_COLOR)


def _configure_matplotlib() -> None:
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


def _save_figure(
    fig: plt.Figure,
    output_stem: Path,
) -> tuple[Path, Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = (
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".svg"),
        output_stem.with_suffix(".png"),
    )
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = _parser().parse_args()
    best_trials = load_group(
        args.best_group,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    cohort = {
        group: load_group(
            group,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
        )
        for group in args.groups
    }
    output_dir = Path(args.output_dir) / "geodesic_real_robot_summary"
    outputs = plot_geodesic_summary_figure(
        best_trials,
        cohort,
        output_dir,
        best_group=args.best_group,
    )
    for name, paths in outputs.items():
        print(f"{name}:")
        for path in paths:
            print(f"  Saved: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot combined and standalone geodesic real-robot figures."
    )
    parser.add_argument("--best-group", type=int, default=BEST_GROUP)
    parser.add_argument("--groups", type=int, nargs="+", default=list(COHORT_GROUPS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


if __name__ == "__main__":
    main()
