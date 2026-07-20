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
    {"key": "hard_lift_event_count", "ylabel": "Hard-lift events"},
    {"key": "hard_lift_duration_s", "ylabel": "Hard-lift duration (s)"},
    {"key": "hard_lift_ratio", "scale": 100.0, "ylabel": "Hard-lift active time (%)"},
    {"key": "force_offset_max_outward_mm", "ylabel": r"Maximum outward offset ($\mathrm{mm}$)"},
    {
        "key": "force_offset_outward_integral_mm_s",
        "divide_by": "duration_s",
        "ylabel": r"Mean outward offset burden ($\mathrm{mm}$)",
    },
    {"key": "force_offset_variation_mm", "ylabel": r"Offset total variation ($\mathrm{mm}$)"},
)


def plot_rescue_metrics(
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
    output_stem = Path(args.output_dir) / f"group_{args.group}" / "rescue_metrics"
    pdf_path, png_path = plot_rescue_metrics(
        trials,
        output_stem,
        group_label=group_config(args.group)["label"],
    )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot force-controller rescue metrics.")
    parser.add_argument("--group", type=int, choices=tuple(sorted(GROUPS)), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


if __name__ == "__main__":
    main()
