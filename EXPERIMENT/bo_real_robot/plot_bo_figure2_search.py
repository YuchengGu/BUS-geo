from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EXPERIMENT.bo_real_robot.plot_bo_figure1 import (
    Measurement,
    candidate_measurements,
    clear_directory,
    configure_matplotlib,
    jittered_scatter,
    load_measurements,
    save_figure,
    soften_axis,
)


DATA_ROOT = Path("/home/ubuntu22/bc_data/gello")
OUTPUT_DIR = Path("EXPERIMENT/bo_real_robot/results/figure2_search")
METHODS = (
    ("bo_full", "BO", "#20E236"),
    ("random_full", "Random", "#FFD21F"),
    ("lhs_full", "LHS", "#19D3C5"),
)
PANEL_FIGSIZE = (4.5, 3.0)
CONTACT_PANEL_FIGSIZE = (4.5, 3.0)


@dataclass(frozen=True)
class RunBest:
    run_name: str
    condition: str
    method_label: str
    best_f: float
    q_at_best_f: float
    best_q: float
    pressure_at_best_f: float
    ft_at_best_f: float
    tau_t_at_best_f: float
    best_f_trial: int
    best_q_trial: int
    n_trials: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BO Figure 2: search strategy comparison.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    configure_matplotlib()
    output_dir = args.output_dir
    panel_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(panel_dir)

    results = load_search_results(args.data_root.expanduser().resolve())
    fig = make_summary_figure(results)
    save_figure(fig, output_dir / "figure2_search_comparison", png=True)
    plt.close(fig)
    save_individual_panels(results, panel_dir)
    write_summary(results, output_dir)
    print(f"Saved Figure 2 outputs under: {output_dir}")
    print(f"Saved individual panel PDFs under: {panel_dir}")


def load_search_results(data_root: Path) -> list[RunBest]:
    out: list[RunBest] = []
    method_by_condition = {condition: (label, color) for condition, label, color in METHODS}
    for run_json in sorted(data_root.glob("bo_*/surface_bo_run.json"), key=lambda p: p.stat().st_mtime):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        condition = f"{run.get('search_strategy')}_{run.get('objective_variant')}"
        if condition not in method_by_condition:
            continue
        measurements = candidate_measurements(load_measurements(run_json.parent, crop=None, with_images=False))
        if not measurements:
            continue
        best_f = min(measurements, key=lambda m: m.F)
        best_q = max(measurements, key=lambda m: m.Q)
        label, _color = method_by_condition[condition]
        out.append(
            RunBest(
                run_name=run_json.parent.name,
                condition=condition,
                method_label=label,
                best_f=float(best_f.F),
                q_at_best_f=float(best_f.Q),
                best_q=float(best_q.Q),
                pressure_at_best_f=float(best_f.pressure),
                ft_at_best_f=float(best_f.force_tangential),
                tau_t_at_best_f=float(best_f.torque_tangential),
                best_f_trial=int(best_f.trial if best_f.trial is not None else -1),
                best_q_trial=int(best_q.trial if best_q.trial is not None else -1),
                n_trials=len(measurements),
            )
        )
    expected_n = len({item.run_name for item in out if item.condition == METHODS[0][0]})
    for condition, label, _color in METHODS:
        n = sum(item.condition == condition for item in out)
        if n != expected_n:
            print(f"Warning: {label} has n={n}, BO has n={expected_n}")
    return out


def make_summary_figure(results: list[RunBest]) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, wspace=0.36, hspace=0.42)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    ax_contact = fig.add_subplot(grid[1, 1])
    plot_metric_box(axes[0], results, "best_f", r"Best observed $F$", lower_better=True)
    plot_metric_box(axes[1], results, "q_at_best_f", r"$Q$ at best $F$", lower_better=False)
    plot_metric_box(axes[2], results, "best_q", r"Best observed $Q$", lower_better=False)
    plot_contact_box(ax_contact, results)
    return fig


def save_individual_panels(results: list[RunBest], panel_dir: Path) -> None:
    panels = [
        ("panel_best_F", "best_f", r"Best observed $F$", True),
        ("panel_Q_at_best_F", "q_at_best_f", r"$Q$ at best $F$", False),
        ("panel_best_Q", "best_q", r"Best observed $Q$", False),
    ]
    for filename, field, ylabel, lower_better in panels:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_metric_box(ax, results, field, ylabel, lower_better=lower_better)
        save_figure(fig, panel_dir / filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=CONTACT_PANEL_FIGSIZE)
    plot_contact_box(ax, results)
    save_figure(fig, panel_dir / "panel_contact_at_best_F")
    plt.close(fig)


def plot_metric_box(
    ax: plt.Axes,
    results: list[RunBest],
    field: str,
    ylabel: str,
    *,
    lower_better: bool,
) -> None:
    grouped = values_by_method(results, field)
    draw_method_boxplot(ax, grouped)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, len(METHODS) + 1))
    ax.set_xticklabels([label for _condition, label, _color in METHODS])
    ax.text(
        0.02,
        1.045,
        r"$\downarrow$ better" if lower_better else r"$\uparrow$ better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="0.35",
        clip_on=False,
    )
    soften_axis(ax)


def plot_contact_box(ax: plt.Axes, results: list[RunBest]) -> None:
    grouped = values_by_method(results, "pressure_at_best_f")
    draw_method_boxplot(ax, grouped)
    positions = np.arange(1, len(METHODS) + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _condition, label, _color in METHODS])
    ax.set_ylabel(r"$F_n$ at best $F$ (N)")
    soften_axis(ax)
    add_bo_median_force_reduction(ax, grouped)


def _draw_contact_metric_boxes(
    ax: plt.Axes,
    data: list[list[float]],
    positions: list[float],
    width: float,
    color: str,
):
    box = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "s", "markerfacecolor": "none", "markeredgecolor": "black", "markersize": 3.0},
        medianprops={"color": "black", "linewidth": 0.75},
        whiskerprops={"color": "0.25", "linewidth": 0.7},
        capprops={"color": "0.25", "linewidth": 0.7},
        flierprops={"marker": "o", "markerfacecolor": "0.25", "markeredgecolor": "0.25", "markersize": 1.8, "alpha": 0.6},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
    return box


def draw_method_boxplot(ax: plt.Axes, grouped: list[list[float]]) -> None:
    positions = np.arange(1, len(grouped) + 1)
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "s", "markerfacecolor": "none", "markeredgecolor": "black", "markersize": 3.3},
        medianprops={"color": "black", "linewidth": 0.85},
        whiskerprops={"color": "0.25", "linewidth": 0.75},
        capprops={"color": "0.25", "linewidth": 0.75},
        flierprops={"marker": "o", "markerfacecolor": "0.25", "markeredgecolor": "0.25", "markersize": 2.1, "alpha": 0.65},
    )
    for patch, (_condition, _label, color) in zip(box["boxes"], METHODS):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.78)
    for x, values, (_condition, _label, color) in zip(positions, grouped, METHODS):
        jittered_scatter(ax, x, np.asarray(values, dtype=float), color, width=0.08, size=7, alpha=0.48)


def add_bo_median_force_reduction(ax: plt.Axes, grouped: list[list[float]]) -> None:
    if len(grouped) < 3:
        return
    bo_median = finite_median(grouped[0])
    if not np.isfinite(bo_median):
        return
    lines = []
    for baseline_label, series in (("Random", grouped[1]), ("LHS", grouped[2])):
        baseline_median = finite_median(series)
        if not np.isfinite(baseline_median):
            continue
        reduction = baseline_median - bo_median
        arrow = r"$\downarrow$" if reduction >= 0 else r"$\uparrow$"
        lines.append(f"vs {baseline_label} {arrow}{abs(reduction):.2f} N")
    if not lines:
        return
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        1.0,
        1.045,
        "\n".join(lines),
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="0.22",
        clip_on=False,
    )


def finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def values_by_method(results: list[RunBest], field: str) -> list[list[float]]:
    return [[float(getattr(item, field)) for item in results if item.condition == condition] for condition, _label, _color in METHODS]


def write_summary(results: list[RunBest], output_dir: Path) -> None:
    rows = [
        {
            "run": item.run_name,
            "condition": item.condition,
            "method": item.method_label,
            "best_F": item.best_f,
            "Q_at_best_F": item.q_at_best_f,
            "best_Q": item.best_q,
            "pressure_at_best_F_N": item.pressure_at_best_f,
            "Ft_at_best_F_N": item.ft_at_best_f,
            "tau_t_at_best_F_Nm": item.tau_t_at_best_f,
            "best_F_trial": item.best_f_trial,
            "best_Q_trial": item.best_q_trial,
            "n_trials": item.n_trials,
        }
        for item in results
    ]
    (output_dir / "figure2_source_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
