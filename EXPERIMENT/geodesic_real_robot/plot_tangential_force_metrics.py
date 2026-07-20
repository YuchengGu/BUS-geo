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
    {"key": "tangential_force_mean_n", "ylabel": r"Mean $F_t$ ($\mathrm{N}$)"},
    {"key": "tangential_force_p95_n", "ylabel": r"P95 $F_t$ ($\mathrm{N}$)"},
    {"key": "tangential_force_max_n", "ylabel": r"Maximum $F_t$ ($\mathrm{N}$)"},
    {"key": "tangential_variation_rate_n_s", "ylabel": r"$F_t$ variation rate ($\mathrm{N\,s^{-1}}$)"},
    {"key": "tangential_derivative_rms_n_s", "ylabel": r"RMS $\dot F_t$ ($\mathrm{N\,s^{-1}}$)"},
    {"key": "shear_ratio_p95", "ylabel": r"P95 $F_t/p$"},
)


def plot_tangential_force_metrics(
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
    output_stem = Path(args.output_dir) / f"group_{args.group}" / "tangential_force_metrics"
    pdf_path, png_path = plot_tangential_force_metrics(
        trials,
        output_stem,
        group_label=group_config(args.group)["label"],
    )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot tangential force and jamming-risk metrics.")
    parser.add_argument("--group", type=int, choices=tuple(sorted(GROUPS)), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


if __name__ == "__main__":
    main()
