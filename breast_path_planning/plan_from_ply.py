from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.interactive_pointcloud import interactive_segment_point_cloud
from breast_path_planning.path_planner import PathPlannerParams
from breast_path_planning.plan_from_frame import plan_from_segmented_cloud
from breast_path_planning.pointcloud_from_d405 import load_point_cloud_ply
from breast_path_planning.segmentation import SegmentationParams

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "breast_path_planning/results"


def make_default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RESULTS_ROOT / f"ply_{stamp}"


def run_ply_planning(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir is not None else make_default_output_dir()
    raw_cloud = load_point_cloud_ply(args.cloud_ply)
    seed_index, segmented_cloud, region_mask = interactive_segment_point_cloud(
        raw_cloud,
        segmentation_params=SegmentationParams(
            spatial_radius_m=args.spatial_radius_m,
            hue_threshold_deg=args.hue_threshold_deg,
            saturation_threshold=args.saturation_threshold,
            value_threshold=args.value_threshold,
            max_distance_from_seed_m=args.max_distance_from_seed_m,
        ),
    )
    result = plan_from_segmented_cloud(
        raw_cloud=raw_cloud,
        segmented_cloud=segmented_cloud,
        region_mask=region_mask,
        seed_indices=[seed_index],
        output_dir=output_dir,
        segmentation_params=SegmentationParams(
            spatial_radius_m=args.spatial_radius_m,
            hue_threshold_deg=args.hue_threshold_deg,
            saturation_threshold=args.saturation_threshold,
            value_threshold=args.value_threshold,
            max_distance_from_seed_m=args.max_distance_from_seed_m,
        ),
        planner_params=PathPlannerParams(
            step_y_m=args.step_y_m,
            step_x_m=args.step_x_m,
            max_normal_angle_deg=args.max_normal_angle_deg,
        ),
        metadata={
            "source": "ply_point_cloud",
            "cloud_ply": str(Path(args.cloud_ply)),
        },
    )

    print(f"seed_index: {seed_index}")
    print(f"raw points: {len(result.raw_cloud)}")
    print(f"segmented points: {len(result.segmented_cloud)}")
    print(f"path points: {len(result.planned_path)}")
    print(f"planned_path.json written to {output_dir / 'planned_path.json'}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pick a seed in a PLY point cloud, preview segmentation, and plan a path.")
    parser.add_argument("--cloud-ply", required=True, help="Input ASCII PLY point cloud with x/y/z and RGB colors.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to breast_path_planning/results/ply_<time>.")
    parser.add_argument("--spatial-radius-m", type=float, default=0.015)
    parser.add_argument("--hue-threshold-deg", type=float, default=30.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.35)
    parser.add_argument("--value-threshold", type=float, default=0.60)
    parser.add_argument("--max-distance-from-seed-m", type=float, default=0.12)
    parser.add_argument("--step-y-m", type=float, default=0.015)
    parser.add_argument("--step-x-m", type=float, default=0.007)
    parser.add_argument("--max-normal-angle-deg", type=float, default=None)
    return parser


def main() -> None:
    run_ply_planning(build_parser().parse_args())


if __name__ == "__main__":
    main()
