from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


DATA_ROOT = Path("/home/ubuntu22/bc_data/gello")
TYPICAL_RUN_DIR = DATA_ROOT / "bo_0707_143029"
OUTPUT_DIR = Path("EXPERIMENT/bo_real_robot/results/figure1")
ULTRASOUND_CROP = (99, 769, 542, 1524)  # row0, row1, col0, col1
GP_DIMS = ("rx", "rz")
GP_DIM_TO_XCOL = {"dn": 0, "rx": 1, "ry": 2, "rz": 3}
GP_PANEL_FIGSIZE = (6.0, 3.0)  # width:height = 6:3
Q_BEFORE_AFTER_FIGSIZE = (2.0, 3.0)  # width:height = 2:3
Q_FORCE_FIGSIZE = (4.0, 3.0)  # width:height = 4:3
PROCESS_PANEL_FIGSIZE = (6.0, 1.0)  # width:height = 6:1
GP_DIM_LABELS = {
    "dn": r"$d_n$ (m)",
    "rx": r"$\Delta r_x$ (rad)",
    "ry": r"$\Delta r_y$ (rad)",
    "rz": r"$\Delta r_z$ (rad)",
}


@dataclass(frozen=True)
class Measurement:
    role: str
    trial: int | None
    phase: str
    file: Path
    F: float
    Q: float
    P_f: float
    P_tau: float
    pressure: float
    force_tangential: float
    torque_tangential: float
    image: np.ndarray | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw BO Figure 1 and export every panel as an individual PDF.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, default=TYPICAL_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--gp-dims", type=str, default=",".join(GP_DIMS), help="Two GP slice dimensions, e.g. rx,rz")
    parser.add_argument(
        "--ultrasound-crop",
        type=str,
        default=",".join(str(v) for v in ULTRASOUND_CROP),
        help="Crop ultrasound frames as row0,row1,col0,col1. Use 'none' for full frames.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir
    panel_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(panel_dir)
    clear_directory(output_dir / "ultrasound_frames")

    crop = parse_crop(args.ultrasound_crop)
    gp_dims = parse_gp_dims(args.gp_dims)

    typical_run = load_run_json(run_dir)
    typical_measurements = load_measurements(run_dir, crop=crop, with_images=True)
    full_runs = load_all_run_jsons(data_root, condition="bo_full")
    all_full_measurements = load_all_scalar_measurements(full_runs)

    selected_images = select_ultrasound_examples(typical_measurements)
    trial_colors = make_trial_color_map(selected_images)

    fig = make_full_figure(
        typical_run=typical_run,
        run_dir=run_dir,
        typical_measurements=typical_measurements,
        full_runs=full_runs,
        all_full_measurements=all_full_measurements,
        selected_images=selected_images,
        trial_colors=trial_colors,
        gp_dims=gp_dims,
    )
    save_figure(fig, output_dir / "figure1_bo_real_robot", png=True)
    plt.close(fig)

    save_individual_panels(
        typical_run=typical_run,
        run_dir=run_dir,
        typical_measurements=typical_measurements,
        full_runs=full_runs,
        all_full_measurements=all_full_measurements,
        selected_images=selected_images,
        trial_colors=trial_colors,
        gp_dims=gp_dims,
        panel_dir=panel_dir,
    )
    save_ultrasound_frame_exports(
        measurements=typical_measurements,
        output_dir=output_dir / "ultrasound_frames",
    )
    write_source_data_summary(
        typical_run_name=run_dir.name,
        typical_run=typical_run,
        full_runs=full_runs,
        selected_images=selected_images,
        output_dir=output_dir,
    )
    print(f"Saved Figure 1 outputs under: {output_dir}")
    print(f"Saved individual panel PDFs under: {panel_dir}")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "axes.grid": True,
            "grid.color": "0.86",
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def parse_crop(text: str) -> tuple[int, int, int, int] | None:
    if str(text).strip().lower() in {"", "none", "full"}:
        return None
    values = [int(part.strip()) for part in str(text).split(",")]
    if len(values) != 4:
        raise ValueError("--ultrasound-crop must be row0,row1,col0,col1 or 'none'")
    r0, r1, c0, c1 = values
    if r1 <= r0 or c1 <= c0:
        raise ValueError("--ultrasound-crop end values must be greater than start values")
    return r0, r1, c0, c1


def parse_gp_dims(text: str) -> tuple[str, str]:
    dims = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(dims) != 2:
        raise ValueError("--gp-dims must contain exactly two dimensions, e.g. rx,rz")
    for dim in dims:
        if dim not in GP_DIM_TO_XCOL:
            raise ValueError(f"Unknown GP dimension {dim!r}; expected one of {sorted(GP_DIM_TO_XCOL)}")
    return dims  # type: ignore[return-value]


def load_run_json(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "surface_bo_run.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_run_jsons(data_root: Path, *, condition: str) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for json_path in sorted(data_root.glob("bo_*/surface_bo_run.json"), key=lambda p: p.stat().st_mtime):
        run = json.loads(json_path.read_text(encoding="utf-8"))
        run_condition = f"{run.get('search_strategy')}_{run.get('objective_variant')}"
        if run_condition == condition:
            out.append((json_path.parent, run))
    if not out:
        raise RuntimeError(f"No runs found for condition {condition!r} under {data_root}")
    return out


def crop_image(image: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    image = np.squeeze(image)
    if crop is None:
        return image
    r0, r1, c0, c1 = crop
    r0 = max(0, min(r0, image.shape[0]))
    r1 = max(0, min(r1, image.shape[0]))
    c0 = max(0, min(c0, image.shape[1]))
    c1 = max(0, min(c1, image.shape[1]))
    if r1 <= r0 or c1 <= c0:
        return image
    return image[r0:r1, c0:c1]


def load_measurements(
    run_dir: Path,
    *,
    crop: tuple[int, int, int, int] | None,
    with_images: bool,
) -> list[Measurement]:
    out: list[Measurement] = []
    for pkl_path in sorted(run_dir.glob("*.pkl")):
        with pkl_path.open("rb") as f:
            frame = pickle.load(f)
        meta = dict(frame.get("meta", {}) or {})
        role = meta.get("bo_measurement_role")
        if role is None:
            continue
        image = None
        if with_images:
            image = crop_image(np.asarray(frame["Ultrasound_gray"]), crop).astype(np.uint8, copy=False)
        out.append(
            Measurement(
                role=str(role),
                trial=None if meta.get("bo_trial_index") is None else int(meta["bo_trial_index"]),
                phase=str(meta.get("bo_phase") or ""),
                file=pkl_path,
                F=float(meta["F"]),
                Q=float(meta["Q"]),
                P_f=float(meta["P_f"]),
                P_tau=float(meta["P_tau"]),
                pressure=float(frame.get("force_pressure_n", np.nan)),
                force_tangential=float(frame.get("force_tangential_norm_n", np.nan)),
                torque_tangential=float(frame.get("torque_tangential_norm_nm", np.nan)),
                image=image,
            )
        )
    return sorted(out, key=lambda m: (-1 if m.trial is None else m.trial, m.role))


def load_all_scalar_measurements(runs: list[tuple[Path, dict[str, Any]]]) -> list[Measurement]:
    measurements: list[Measurement] = []
    for run_dir, _run in runs:
        measurements.extend(load_measurements(run_dir, crop=None, with_images=False))
    return measurements


def candidate_measurements(measurements: list[Measurement]) -> list[Measurement]:
    return sorted([m for m in measurements if m.role == "candidate" and m.trial is not None], key=lambda m: m.trial)


def display_candidate_measurements(measurements: list[Measurement]) -> list[Measurement]:
    return [m for m in candidate_measurements(measurements) if m.trial is not None and m.trial > 0]


def select_ultrasound_examples(measurements: list[Measurement]) -> list[tuple[str, Measurement]]:
    before = next(m for m in measurements if m.role == "before")
    verified = next(m for m in measurements if m.role == "verified_best")
    candidates = display_candidate_measurements(measurements)
    if not candidates:
        raise RuntimeError("No candidate measurements with trial > 0 found for typical BO run")
    best_f = min(candidates, key=lambda m: m.F)
    max_penalty = max(candidates, key=lambda m: m.P_f + m.P_tau)
    mid = candidates[len(candidates) // 2]
    selected = [
        ("Before", before),
        (f"Initial\ntrial {candidates[0].trial}", candidates[0]),
        (f"High penalty\ntrial {max_penalty.trial}", max_penalty),
        (f"Mid search\ntrial {mid.trial}", mid),
        (f"Best F\ntrial {best_f.trial}", best_f),
        ("Verified\nbest", verified),
    ]
    # Keep labels unique if the same trial satisfies several roles.
    unique: list[tuple[str, Measurement]] = []
    seen: set[tuple[str, int | None]] = set()
    for label, measurement in selected:
        key = (measurement.role, measurement.trial)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, measurement))
    return unique[:6]


def make_trial_color_map(selected_images: list[tuple[str, Measurement]]) -> dict[int, str]:
    palette = ["#2ca02c", "#17becf", "#9467bd", "#d62728", "#ff7f0e", "#bcbd22"]
    out: dict[int, str] = {}
    for color, (_label, measurement) in zip(palette, selected_images):
        if measurement.trial is not None:
            out[int(measurement.trial)] = color
    return out


def make_full_figure(
    *,
    typical_run: dict[str, Any],
    run_dir: Path,
    typical_measurements: list[Measurement],
    full_runs: list[tuple[Path, dict[str, Any]]],
    all_full_measurements: list[Measurement],
    selected_images: list[tuple[str, Measurement]],
    trial_colors: dict[int, str],
    gp_dims: tuple[str, str],
) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 9.1), constrained_layout=False)
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.25, 1.05, 0.88, 1.0, 1.45],
        hspace=0.58,
    )
    row1 = outer[0].subgridspec(1, 2, wspace=0.28)
    row2 = outer[1].subgridspec(1, 2, wspace=0.34)
    ax_gp0 = fig.add_subplot(row1[0, 0])
    ax_gp1 = fig.add_subplot(row1[0, 1])
    ax_q_before_after = fig.add_subplot(row2[0, 0])
    ax_q_force_bins = fig.add_subplot(row2[0, 1])
    ax_neg_f = fig.add_subplot(outer[2])
    ax_q_penalty = fig.add_subplot(outer[3])
    image_grid = outer[4].subgridspec(1, len(selected_images), wspace=0.035)
    image_axes = [fig.add_subplot(image_grid[0, i]) for i in range(len(selected_images))]

    posterior = np.load(run_dir / typical_run["posterior_file"])
    plot_gp_slice(ax_gp0, posterior, gp_dims[0])
    plot_gp_slice(ax_gp1, posterior, gp_dims[1])
    plot_q_before_after(ax_q_before_after, full_runs)
    plot_q_by_pressure_bins(ax_q_force_bins, all_full_measurements)
    plot_negative_objective_trace(ax_neg_f, typical_measurements, trial_colors)
    plot_q_penalty_trace(ax_q_penalty, typical_measurements, trial_colors)
    plot_ultrasound_strip(image_axes, selected_images)

    return fig


def save_individual_panels(
    *,
    typical_run: dict[str, Any],
    run_dir: Path,
    typical_measurements: list[Measurement],
    full_runs: list[tuple[Path, dict[str, Any]]],
    all_full_measurements: list[Measurement],
    selected_images: list[tuple[str, Measurement]],
    trial_colors: dict[int, str],
    gp_dims: tuple[str, str],
    panel_dir: Path,
) -> None:
    posterior = np.load(run_dir / typical_run["posterior_file"])

    for dim in gp_dims:
        fig, ax = plt.subplots(figsize=GP_PANEL_FIGSIZE)
        plot_gp_slice(ax, posterior, dim)
        save_figure(fig, panel_dir / f"panel_gp_{dim}")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=Q_BEFORE_AFTER_FIGSIZE)
    plot_q_before_after(ax, full_runs)
    save_figure(fig, panel_dir / "panel_q_before_after")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=Q_FORCE_FIGSIZE)
    plot_q_by_pressure_bins(ax, all_full_measurements)
    save_figure(fig, panel_dir / "panel_q_by_pressure")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PROCESS_PANEL_FIGSIZE)
    plot_negative_objective_trace(ax, typical_measurements, trial_colors)
    save_figure(fig, panel_dir / "panel_negative_F_trace")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PROCESS_PANEL_FIGSIZE)
    plot_q_penalty_trace(ax, typical_measurements, trial_colors)
    save_figure(fig, panel_dir / "panel_Q_Pf_Ptau_trace")
    plt.close(fig)

    fig = plt.figure(figsize=(7.0, 1.6), constrained_layout=False)
    grid = fig.add_gridspec(1, len(selected_images), wspace=0.035)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(selected_images))]
    plot_ultrasound_strip(axes, selected_images)
    save_figure(fig, panel_dir / "panel_ultrasound_strip")
    plt.close(fig)


def plot_gp_slice(ax: plt.Axes, posterior: np.lib.npyio.NpzFile, dim: str) -> None:
    grid = posterior[f"{dim}_grid"]
    mean = posterior[f"{dim}_mean"]
    std = posterior[f"{dim}_std"]
    ei = posterior[f"{dim}_ei"]
    observed_x = posterior["observed_x"][:, GP_DIM_TO_XCOL[dim]]
    observed_F = posterior["observed_F"]
    best_idx = int(np.argmin(observed_F))

    ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std, color="#19D3C5", alpha=0.55, lw=0)
    ax.plot(grid, mean, color="#004DFF", lw=1.35, label="Model mean")
    ax.scatter(observed_x, observed_F, s=13, color="black", zorder=4, label="Observation")
    ax.scatter(observed_x[best_idx], observed_F[best_idx], s=22, color="red", zorder=5)
    ax.axvline(observed_x[best_idx], color="red", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlabel(GP_DIM_LABELS[dim])
    ax.set_ylabel(r"$F$", labelpad=-5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_ei = ax.twinx()
    ei_norm = ei / np.nanmax(ei) if np.nanmax(ei) > 0 else ei
    ax_ei.plot(grid, ei_norm, color="#FF7F0E", lw=1.0, label="Utility")
    ax_ei.scatter(grid[int(np.nanargmax(ei_norm))], float(np.nanmax(ei_norm)), s=22, marker="*", color="#FF7F0E", zorder=5)
    ax_ei.set_ylabel("EI", labelpad=-5)
    ax_ei.set_yticks([0, 1])
    ax.tick_params(axis="y", pad=0)
    ax_ei.tick_params(axis="y", pad=0)
    ax_ei.spines["top"].set_visible(False)
    soften_axis(ax)
    soften_axis(ax_ei)


def plot_q_before_after(ax: plt.Axes, full_runs: list[tuple[Path, dict[str, Any]]]) -> None:
    before = np.asarray([run["before"]["Q"] for _path, run in full_runs], dtype=float)
    after = np.asarray([best_trial_objective(run, key="F")["Q"] for _path, run in full_runs], dtype=float)
    box = ax.boxplot(
        [before, after],
        positions=[1, 2],
        widths=0.54,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "s", "markerfacecolor": "none", "markeredgecolor": "black", "markersize": 3.5},
        medianprops={"color": "black", "linewidth": 0.9},
        whiskerprops={"color": "0.25", "linewidth": 0.8},
        capprops={"color": "0.25", "linewidth": 0.8},
        flierprops={"marker": "o", "markerfacecolor": "0.2", "markeredgecolor": "0.2", "markersize": 2.3, "alpha": 0.8},
    )
    for patch, color in zip(box["boxes"], ["#FFD21F", "#20E236"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.85)
    jittered_scatter(ax, 1, before, "#B8860B")
    jittered_scatter(ax, 2, after, "#178C23")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Before\nBO", "Best after\nBO"])
    ax.set_ylabel(r"$Q$")
    ax.set_xlim(0.45, 2.55)
    soften_axis(ax)
    add_before_after_significance(ax, before, after)


def plot_q_by_pressure_bins(ax: plt.Axes, measurements: list[Measurement]) -> None:
    candidates = [m for m in measurements if m.role == "candidate" and np.isfinite(m.pressure) and np.isfinite(m.Q)]
    bins = [(-np.inf, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, np.inf)]
    labels = ["<1", "1-2", "2-3", "3-4", "4-6", "6-8", ">8"]
    grouped: list[list[float]] = []
    for lo, hi in bins:
        grouped.append([m.Q for m in candidates if lo <= m.pressure < hi])
    positions = np.arange(1, len(grouped) + 1)
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "s", "markerfacecolor": "none", "markeredgecolor": "black", "markersize": 3.2},
        medianprops={"color": "black", "linewidth": 0.85},
        whiskerprops={"color": "0.25", "linewidth": 0.75},
        capprops={"color": "0.25", "linewidth": 0.75},
        flierprops={"marker": "o", "markerfacecolor": "0.25", "markeredgecolor": "0.25", "markersize": 2.0, "alpha": 0.65},
    )
    colors = ["#FEE8A8", "#FDD66A", "#BFE38A", "#79C86B", "#35A853", "#4FB7A9", "#B6B6B6"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.75)
    for x, ys, color in zip(positions, grouped, colors):
        if ys:
            jittered_scatter(ax, x, np.asarray(ys), color, width=0.13, size=6, alpha=0.42)
    ax.axvspan(2.5, 6.5, color="#20E236", alpha=0.08, lw=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$F_n$ (N)")
    ax.set_ylabel(r"$Q$")
    soften_axis(ax)


def add_before_after_significance(ax: plt.Axes, before: np.ndarray, after: np.ndarray) -> None:
    label = significance_label(paired_wilcoxon_pvalue(before, after))
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    y = 1.035
    h = 0.035
    ax.plot(
        [1, 1, 2, 2],
        [y, y + h, y + h, y],
        transform=transform,
        color="black",
        linewidth=0.8,
        clip_on=False,
    )
    ax.text(
        1.5,
        y + h + 0.006,
        label,
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="black",
        clip_on=False,
    )


def paired_wilcoxon_pvalue(before: np.ndarray, after: np.ndarray) -> float:
    valid = np.isfinite(before) & np.isfinite(after)
    before = np.asarray(before, dtype=float)[valid]
    after = np.asarray(after, dtype=float)[valid]
    if before.size < 3 or np.allclose(before, after):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(before, after, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def significance_label(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n.s."
    if pvalue < 1e-4:
        return "****"
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 5e-2:
        return "*"
    return "n.s."


def plot_negative_objective_trace(
    ax: plt.Axes,
    measurements: list[Measurement],
    trial_colors: dict[int, str],
    *,
    show_xlabel: bool = True,
) -> None:
    candidates = display_candidate_measurements(measurements)
    trials = np.asarray([m.trial for m in candidates], dtype=float)
    neg_f = np.asarray([-m.F for m in candidates], dtype=float)
    best_so_far = np.maximum.accumulate(neg_f)
    current_color = "#FB6A4A"
    best_color = "#A50F15"
    ax.plot(trials, neg_f, color=current_color, lw=0.9, alpha=0.74)
    ax.scatter(trials, neg_f, s=13, color=current_color, edgecolor="white", linewidth=0.35, alpha=0.78, zorder=3)
    ax_best = ax.twinx()
    ax_best.step(trials, best_so_far, where="post", color=best_color, lw=1.75)
    add_trial_markers(ax, trial_colors)
    add_trial_markers(ax_best, trial_colors)
    ax.set_ylabel(r"$-F$", color=current_color)
    ax.tick_params(axis="y", labelcolor=current_color, color=current_color)
    ax.spines["left"].set_color(current_color)
    ax_best.set_ylabel(r"Best $-F$", color=best_color)
    ax_best.tick_params(axis="y", labelcolor=best_color, color=best_color)
    ax_best.spines["right"].set_color(best_color)
    ax_best.spines["top"].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("BO evaluation")
    else:
        ax.set_xlabel("")
    ax.set_xlim(float(np.nanmin(trials)) - 0.25, float(np.nanmax(trials)) + 0.25)
    soften_axis(ax)
    soften_axis(ax_best)


def plot_q_penalty_trace(
    ax: plt.Axes,
    measurements: list[Measurement],
    trial_colors: dict[int, str],
) -> None:
    candidates = display_candidate_measurements(measurements)
    trials = np.asarray([m.trial for m in candidates], dtype=float)
    q = np.asarray([m.Q for m in candidates], dtype=float)
    pf = np.asarray([m.P_f for m in candidates], dtype=float)
    pt = np.asarray([m.P_tau for m in candidates], dtype=float)
    penalty = pf + pt

    ax.plot(trials, q, color="#FF7F0E", lw=1.25)
    ax.scatter(trials, q, s=11, color="#FF7F0E", edgecolor="white", linewidth=0.25, zorder=3)
    ax.set_ylabel(r"$Q$", color="#FF7F0E")
    ax.set_ylim(min(0.625, float(np.nanmin(q)) - 0.001), max(0.640, float(np.nanmax(q)) + 0.001))
    ax.set_yticks([0.625, 0.630, 0.635, 0.640])
    ax.tick_params(axis="y", labelcolor="#FF7F0E")
    ax_penalty = ax.twinx()
    ax_penalty.plot(trials, penalty, color="#1F77B4", lw=1.05)
    ax_penalty.scatter(trials, penalty, s=9, color="#1F77B4", edgecolor="white", linewidth=0.2, zorder=3)
    ax_penalty.set_ylabel("Penalty", color="#1F77B4")
    ax_penalty.tick_params(axis="y", labelcolor="#1F77B4")
    ax.set_xlabel("BO evaluation")
    add_trial_markers(ax, trial_colors)
    add_trial_markers(ax_penalty, trial_colors)
    soften_axis(ax)
    soften_axis(ax_penalty)
    ax_penalty.spines["top"].set_visible(False)


def plot_ultrasound_strip(axes: list[plt.Axes], selected_images: list[tuple[str, Measurement]]) -> None:
    for ax, (label, measurement) in zip(axes, selected_images):
        if measurement.image is None:
            raise RuntimeError("Selected ultrasound example has no image loaded")
        ax.imshow(measurement.image, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        trial_text = "before" if measurement.trial is None and measurement.role == "before" else (
            "verified" if measurement.role == "verified_best" else f"trial {measurement.trial}"
        )
        ax.set_xlabel(
            f"{label}\n{trial_text}\nQ={measurement.Q:.3f}, F={measurement.F:.2f}",
            fontsize=6.1,
            labelpad=2,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_edgecolor("0.25")


def best_trial_objective(run: dict[str, Any], *, key: str) -> dict[str, float]:
    trials = [trial for trial in run.get("trials", []) if trial.get("objective")]
    if not trials:
        raise RuntimeError("Run has no objective-bearing BO trials")
    if key == "F":
        return min(trials, key=lambda trial: trial["objective"]["F"])["objective"]
    if key == "Q":
        return max(trials, key=lambda trial: trial["objective"]["Q"])["objective"]
    raise ValueError(f"Unsupported best key {key!r}")


def jittered_scatter(
    ax: plt.Axes,
    x: float,
    y: np.ndarray,
    color: str,
    *,
    width: float = 0.08,
    size: float = 9,
    alpha: float = 0.55,
) -> None:
    if len(y) == 0:
        return
    offsets = np.linspace(-width, width, len(y)) if len(y) > 1 else np.asarray([0.0])
    ax.scatter(np.full_like(y, x, dtype=float) + offsets, y, s=size, color=color, alpha=alpha, linewidth=0, zorder=3)


def add_trial_markers(ax: plt.Axes, trial_colors: dict[int, str]) -> None:
    for trial, color in trial_colors.items():
        ax.axvline(trial, color=color, lw=0.75, ls="--", alpha=0.78, zorder=0)


def soften_axis(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=6.4, length=2.5, width=0.6)
    ax.spines["left"].set_color("0.16")
    ax.spines["bottom"].set_color("0.16")


def save_figure(fig: plt.Figure, path_without_suffix: Path, *, png: bool = False) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    if png:
        fig.savefig(path_without_suffix.with_suffix(".png"), dpi=450, bbox_inches="tight")


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_ultrasound_frame_exports(*, measurements: list[Measurement], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [m for m in measurements if m.role == "before"]
    ordered.extend(display_candidate_measurements(measurements))
    ordered.extend([m for m in measurements if m.role == "verified_best"])

    lines = [
        "file\trole\ttrial\tphase\tF\tQ\tP_f\tP_tau\tpressure_N\tforce_tangential_N\ttorque_tangential_Nm\tsource_pkl",
    ]
    for index, measurement in enumerate(ordered):
        if measurement.image is None:
            continue
        if measurement.role == "before":
            name = f"{index:02d}_before"
        elif measurement.role == "verified_best":
            name = f"{index:02d}_verified_best"
        else:
            name = f"{index:02d}_trial_{measurement.trial:02d}"

        fig, ax = plt.subplots(figsize=(2.2, 1.7))
        ax.imshow(measurement.image, cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0)
        fig.savefig(output_dir / f"{name}.png", dpi=450, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        lines.append(
            "\t".join(
                [
                    f"{name}.pdf",
                    measurement.role,
                    "" if measurement.trial is None else str(measurement.trial),
                    measurement.phase,
                    f"{measurement.F:.9g}",
                    f"{measurement.Q:.9g}",
                    f"{measurement.P_f:.9g}",
                    f"{measurement.P_tau:.9g}",
                    f"{measurement.pressure:.9g}",
                    f"{measurement.force_tangential:.9g}",
                    f"{measurement.torque_tangential:.9g}",
                    measurement.file.name,
                ]
            )
        )
    (output_dir / "ultrasound_frame_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_data_summary(
    *,
    typical_run_name: str,
    typical_run: dict[str, Any],
    full_runs: list[tuple[Path, dict[str, Any]]],
    selected_images: list[tuple[str, Measurement]],
    output_dir: Path,
) -> None:
    summary = {
        "typical_run": typical_run_name,
        "typical_strategy": typical_run.get("search_strategy"),
        "typical_objective_variant": typical_run.get("objective_variant"),
        "full_bo_run_count": len(full_runs),
        "q_before_after": [
            {
                "run": run_dir.name,
                "before_Q": run["before"]["Q"],
                "best_observed_Q_at_best_F": best_trial_objective(run, key="F")["Q"],
                "best_observed_F": best_trial_objective(run, key="F")["F"],
            }
            for run_dir, run in full_runs
        ],
        "selected_ultrasound": [
            {
                "label": label,
                "role": measurement.role,
                "trial": measurement.trial,
                "F": measurement.F,
                "Q": measurement.Q,
                "P_f": measurement.P_f,
                "P_tau": measurement.P_tau,
                "pressure_N": measurement.pressure,
                "force_tangential_N": measurement.force_tangential,
                "torque_tangential_Nm": measurement.torque_tangential,
                "file": measurement.file.name,
            }
            for label, measurement in selected_images
        ],
    }
    (output_dir / "figure1_source_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
