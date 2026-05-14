from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.geometry import rotvec_pose_to_transform
from breast_path_planning.interactive_pointcloud import interactive_segment_point_cloud
from breast_path_planning.path_planner import PathPlannerParams
from breast_path_planning.plan_from_frame import plan_from_segmented_cloud
from breast_path_planning.pointcloud_from_d405 import realsense_frames_to_point_cloud
from breast_path_planning.segmentation import SegmentationParams

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_T_TCP_CAMERA_PATH = REPO_ROOT / "hand_eye_calibration/results_0512_222937_calib_11x8_stride10/T_tcp_camera.npy"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "breast_path_planning/results"


def compute_base_camera_transform(ee_pos_rotvec: np.ndarray, T_tcp_camera: np.ndarray) -> np.ndarray:
    T_base_tcp = rotvec_pose_to_transform(ee_pos_rotvec)
    return T_base_tcp @ np.asarray(T_tcp_camera, dtype=float).reshape(4, 4)


def load_transform(path: str | Path) -> np.ndarray:
    transform = np.load(Path(path))
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform in {path}, got {transform.shape}")
    return np.asarray(transform, dtype=float)


def make_default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RESULTS_ROOT / f"live_{stamp}"


def capture_current_d405_frame(warmup_reads: int) -> tuple[object, object, object]:
    from gello.cameras.D405 import RealSenseD405

    d405 = RealSenseD405()
    for _ in range(max(1, int(warmup_reads))):
        d405.read()
    color_frame, depth_frame = d405.latest_frames()
    if color_frame is None or depth_frame is None:
        raise RuntimeError("D405 did not provide a valid color/depth frame")
    return color_frame, depth_frame, d405


def read_current_ee_pose(*, host: str, robot_port: int) -> np.ndarray:
    from gello.zmq_core.robot_node import ZMQClientRobot

    robot = ZMQClientRobot(port=robot_port, host=host)
    obs = robot.get_observations()
    if "ee_pos_rotvec" not in obs:
        raise KeyError("Robot observations do not contain ee_pos_rotvec")
    return np.asarray(obs["ee_pos_rotvec"], dtype=float)


def run_live_planning(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir is not None else make_default_output_dir()
    T_tcp_camera = load_transform(args.t_tcp_camera)

    color_frame, depth_frame, d405 = capture_current_d405_frame(args.warmup_reads)
    try:
        ee_pos_rotvec = read_current_ee_pose(host=args.hostname, robot_port=args.robot_port)
        T_base_camera = compute_base_camera_transform(ee_pos_rotvec, T_tcp_camera)
        raw_cloud = realsense_frames_to_point_cloud(
            color_frame,
            depth_frame,
            T_base_camera,
            stride=args.point_stride,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if len(raw_cloud) == 0:
            raise RuntimeError("RealSense SDK produced no valid point cloud points")
        segmentation_params = SegmentationParams(
            spatial_radius_m=args.spatial_radius_m,
            hue_threshold_deg=args.hue_threshold_deg,
            saturation_threshold=args.saturation_threshold,
            value_threshold=args.value_threshold,
            max_distance_from_seed_m=args.max_distance_from_seed_m,
        )
        seed_index, segmented_cloud, region_mask = interactive_segment_point_cloud(
            raw_cloud,
            segmentation_params=segmentation_params,
        )
        result = plan_from_segmented_cloud(
            raw_cloud=raw_cloud,
            segmented_cloud=segmented_cloud,
            region_mask=region_mask,
            seed_indices=[seed_index],
            output_dir=output_dir,
            segmentation_params=segmentation_params,
            planner_params=PathPlannerParams(
                step_y_m=args.step_y_m,
                step_x_m=args.step_x_m,
                max_normal_angle_deg=args.max_normal_angle_deg,
            ),
            metadata={
                "source": "D405_realsense_pointcloud",
                "point_stride": args.point_stride,
                "pointcloud_backend": "librealsense",
            },
        )
    finally:
        if hasattr(d405, "pipeline"):
            d405.pipeline.stop()

    print(f"seed_index: {seed_index}")
    print(f"raw points: {len(result.raw_cloud)}")
    print(f"segmented points: {len(result.segmented_cloud)}")
    print(f"path points: {len(result.planned_path)}")
    print(f"planned_path.json written to {output_dir / 'planned_path.json'}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one D405 frame, pick a 3D point cloud seed, and plan a breast scan path.")
    parser.add_argument("--hostname", default="127.0.0.1", help="Robot ZMQ host.")
    parser.add_argument("--robot-port", type=int, default=6001, help="Robot ZMQ port.")
    parser.add_argument(
        "--T-tcp-camera",
        "--t-tcp-camera",
        dest="t_tcp_camera",
        default=DEFAULT_T_TCP_CAMERA_PATH,
        help="Path to T_tcp_camera.npy.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to breast_path_planning/results/live_<time>.")
    parser.add_argument("--warmup-reads", type=int, default=3)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=0.7)
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
    run_live_planning(build_parser().parse_args())


if __name__ == "__main__":
    main()
