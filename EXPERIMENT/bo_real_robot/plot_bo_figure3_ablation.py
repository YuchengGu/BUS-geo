from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EXPERIMENT.bo_real_robot.plot_bo_figure1 import (
    candidate_measurements,
    clear_directory,
    configure_matplotlib,
    jittered_scatter,
    load_measurements,
    save_figure,
    soften_axis,
)


DATA_ROOT = Path("/home/ubuntu22/bc_data/gello")
OUTPUT_DIR = Path("EXPERIMENT/bo_real_robot/results/figure3_ablation")
METHODS = (
    ("bo_no_penalty", r"$-Q$", "#F4A09A"),
    ("bo_force_only", r"$-Q+P_f$", "#7FA6C9"),
    ("bo_torque_only", r"$-Q+P_{\tau}$", "#B279A2"),
    ("bo_full", r"$-Q+P_f+P_{\tau}$", "#20E236"),
)
PANEL_FIGSIZE = (4.5, 3.0)


@dataclass(frozen=True)
class AblationBest:
    run_name: str
    condition: str
    label: str
    best_f: float
    q_at_best_f: float
    best_q: float
    pressure_at_best_f: float
    ft_at_best_f: float
    tau_t_at_best_f: float
    tau_z_at_best_f: float
    best_f_trial: int
    best_q_trial: int
    n_trials: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BO Figure 3: objective ablation.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    configure_matplotlib()
    output_dir = args.output_dir
    panel_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(panel_dir)

    results = load_ablation_results(args.data_root.expanduser().resolve())
    fig = make_summary_figure(results)
    save_figure(fig, output_dir / "figure3_objective_ablation", png=True)
    plt.close(fig)
    save_individual_panels(results, panel_dir)
    write_summary(results, output_dir)
    print(f"Saved Figure 3 outputs under: {output_dir}")
    print(f"Saved individual panel PDFs under: {panel_dir}")


def load_ablation_results(data_root: Path) -> list[AblationBest]:
    label_by_condition = {condition: label for condition, label, _color in METHODS}
    out: list[AblationBest] = []
    for run_json in sorted(data_root.glob("bo_*/surface_bo_run.json"), key=lambda p: p.stat().st_mtime):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        condition = f"{run.get('search_strategy')}_{run.get('objective_variant')}"
        if condition not in label_by_condition:
            continue
        measurements = candidate_measurements(load_measurements(run_json.parent, crop=None, with_images=False))
        if not measurements:
            continue
        best_f = min(measurements, key=lambda m: m.F)
        best_q = max(measurements, key=lambda m: m.Q)
        out.append(
            AblationBest(
                run_name=run_json.parent.name,
                condition=condition,
                label=label_by_condition[condition],
                best_f=float(best_f.F),
                q_at_best_f=float(best_f.Q),
                best_q=float(best_q.Q),
                pressure_at_best_f=float(best_f.pressure),
                ft_at_best_f=float(best_f.force_tangential),
                tau_t_at_best_f=float(best_f.torque_tangential),
                tau_z_at_best_f=float(abs(force_tau_z_from_file(best_f.file))),
                best_f_trial=int(best_f.trial if best_f.trial is not None else -1),
                best_q_trial=int(best_q.trial if best_q.trial is not None else -1),
                n_trials=len(measurements),
            )
        )
    expected_n = len([item for item in out if item.condition == "bo_full"])
    for condition, label, _color in METHODS:
        n = sum(item.condition == condition for item in out)
        if n != expected_n:
            print(f"Warning: {label} has n={n}, full has n={expected_n}")
    return out


def force_tau_z_from_file(pkl_path: Path) -> float:
    # Keep this tiny lazy import out of the hot path until needed.
    import pickle

    with pkl_path.open("rb") as f:
        frame = pickle.load(f)
    force = np.asarray(frame.get("force", np.zeros(6)), dtype=float).reshape(-1)
    if force.size < 6:
        return float("nan")
    return float(force[5])


def make_summary_figure(results: list[AblationBest]) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 6.9), constrained_layout=False)
    grid = fig.add_gridspec(3, 2, wspace=0.36, hspace=0.45)
    panels = [
        ("best_q", r"Best observed $Q$", False),
        ("q_at_best_f", r"$Q$ at best $F$", False),
        ("pressure_at_best_f", r"$F_n$ at best $F$ (N)", True),
        ("ft_at_best_f", r"$F_t$ at best $F$ (N)", True),
        ("tau_t_at_best_f", r"$\tau_t$ at best $F$ (N m)", True),
        ("tau_z_at_best_f", r"$|\tau_z|$ at best $F$ (N m)", True),
    ]
    for ax, (field, ylabel, lower_better) in zip(fig.axes, []):
        pass
    for i, (field, ylabel, lower_better) in enumerate(panels):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        plot_metric_box(ax, results, field, ylabel, lower_better=lower_better)
    return fig


def save_individual_panels(results: list[AblationBest], panel_dir: Path) -> None:
    panels = [
        ("panel_best_Q", "best_q", r"Best observed $Q$", False),
        ("panel_Q_at_best_F", "q_at_best_f", r"$Q$ at best $F$", False),
        ("panel_pressure_at_best_F", "pressure_at_best_f", r"$F_n$ at best $F$ (N)", True),
        ("panel_Ft_at_best_F", "ft_at_best_f", r"$F_t$ at best $F$ (N)", True),
        ("panel_taut_at_best_F", "tau_t_at_best_f", r"$\tau_t$ at best $F$ (N m)", True),
        ("panel_tauz_at_best_F", "tau_z_at_best_f", r"$|\tau_z|$ at best $F$ (N m)", True),
    ]
    for filename, field, ylabel, lower_better in panels:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_metric_box(ax, results, field, ylabel, lower_better=lower_better)
        save_figure(fig, panel_dir / filename)
        plt.close(fig)


def plot_metric_box(
    ax: plt.Axes,
    results: list[AblationBest],
    field: str,
    ylabel: str,
    *,
    lower_better: bool,
) -> None:
    grouped = [[float(getattr(item, field)) for item in results if item.condition == condition] for condition, _label, _color in METHODS]
    positions = np.arange(1, len(METHODS) + 1)
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
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _condition, label, _color in METHODS])
    ax.set_ylabel(ylabel)
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


def write_summary(results: list[AblationBest], output_dir: Path) -> None:
    rows = [
        {
            "run": item.run_name,
            "condition": item.condition,
            "objective": item.label,
            "best_F": item.best_f,
            "Q_at_best_F": item.q_at_best_f,
            "best_Q": item.best_q,
            "pressure_at_best_F_N": item.pressure_at_best_f,
            "Ft_at_best_F_N": item.ft_at_best_f,
            "tau_t_at_best_F_Nm": item.tau_t_at_best_f,
            "abs_tau_z_at_best_F_Nm": item.tau_z_at_best_f,
            "best_F_trial": item.best_f_trial,
            "best_Q_trial": item.best_q_trial,
            "n_trials": item.n_trials,
        }
        for item in results
    ]
    (output_dir / "figure3_source_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
