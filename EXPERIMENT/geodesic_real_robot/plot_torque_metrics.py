#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from EXPERIMENT.geodesic_real_robot.force_analysis import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    GROUPS,
    TrialSignals,
    group_config,
    load_group,
    plot_metric_panels,
)


PANELS = (
    {"key": "torque_tangential_mean_nm", "ylabel": r"Mean $\tau_t$ ($\mathrm{N\,m}$)"},
    {"key": "torque_tangential_p95_nm", "ylabel": r"P95 $\tau_t$ ($\mathrm{N\,m}$)"},
    {"key": "torque_tangential_max_nm", "ylabel": r"Maximum $\tau_t$ ($\mathrm{N\,m}$)"},
    {"key": "torque_axial_p95_nm", "ylabel": r"P95 $|\tau_z|$ ($\mathrm{N\,m}$)"},
    {"key": "torque_total_p95_nm", "ylabel": r"P95 $\|\tau\|$ ($\mathrm{N\,m}$)"},
    {"key": "torque_variation_rate_nm_s", "ylabel": r"Torque variation rate ($\mathrm{N\,m\,s^{-1}}$)"},
)


def plot_torque_metrics(
    trials: dict[str, TrialSignals],
    output_stem: str | Path,
    *,
    group_label: str,
) -> tuple[Path, Path]:
    return plot_metric_panels(trials, PANELS, output_stem, group_label=group_label)


def main() -> None:
    args = _parser().parse_args()
    trials = load_group(
        args.group,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    output_stem = Path(args.output_dir) / f"group_{args.group}" / "torque_metrics"
    pdf_path, png_path = plot_torque_metrics(
        trials,
        output_stem,
        group_label=group_config(args.group)["label"],
    )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot torque and contact-eccentricity metrics.")
    parser.add_argument("--group", type=int, choices=tuple(sorted(GROUPS)), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


if __name__ == "__main__":
    main()
