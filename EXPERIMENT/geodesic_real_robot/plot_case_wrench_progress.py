#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_geodesic_robot"))

import matplotlib.pyplot as plt

from EXPERIMENT.geodesic_real_robot.force_analysis import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    GROUPS,
    METHOD_COLORS,
    METHOD_LABELS,
    TrialSignals,
    group_config,
    load_group,
    ordered_trials,
    resample_by_progress,
    save_figure,
    style_axis,
)


def plot_wrench_progress(
    trials: dict[str, TrialSignals],
    output_stem: str | Path,
    *,
    group_label: str,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    signal_specs = (
        ("pressure", r"$p=-F_z$ ($\mathrm{N}$)"),
        ("tangential_force", r"$F_t$ ($\mathrm{N}$)"),
        ("tangential_torque", r"$\tau_t$ ($\mathrm{N\,m}$)"),
        ("axial_torque", r"$|\tau_z|$ ($\mathrm{N\,m}$)"),
    )
    for method, trial in ordered_trials(trials):
        values_by_name = {
            "pressure": trial.pressure_n,
            "tangential_force": trial.tangential_force_n,
            "tangential_torque": trial.tangential_torque_nm,
            "axial_torque": trial.axial_torque_nm,
        }
        for axis, (name, _ylabel) in zip(axes.flat, signal_specs):
            grid, values = resample_by_progress(trial.progress, values_by_name[name])
            axis.plot(
                grid,
                values,
                color=METHOD_COLORS[method],
                linewidth=1.25 if method == "geodesic" else 0.95,
                alpha=1.0 if method == "geodesic" else 0.82,
                label=METHOD_LABELS[method],
            )
    axes[0, 0].axhspan(3.0, 4.0, color="#F2C14E", alpha=0.18, linewidth=0)
    axes[0, 0].axhline(8.0, color="#B22222", linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(8.0, color="#B22222", linestyle="--", linewidth=0.8)
    for axis, (_name, ylabel) in zip(axes.flat, signal_specs):
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_xlim(0.0, 1.0)
        style_axis(axis)
    axes[1, 0].set_xlabel("Normalized scan progress", fontsize=9)
    axes[1, 1].set_xlabel("Normalized scan progress", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(group_label, fontsize=11)
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def main() -> None:
    args = _parser().parse_args()
    trials = load_group(
        args.group,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    output_stem = Path(args.output_dir) / f"group_{args.group}" / "wrench_vs_scan_progress"
    pdf_path, png_path = plot_wrench_progress(
        trials,
        output_stem,
        group_label=group_config(args.group)["label"],
    )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot force and torque versus scan progress.")
    parser.add_argument("--group", type=int, choices=tuple(sorted(GROUPS)), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


if __name__ == "__main__":
    main()
