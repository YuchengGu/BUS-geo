from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from breast_path_planning.geometry import rodrigues
from gello.zmq_core.robot_node import ZMQClientRobot
from visual_guided_collection_gui.surface_teleop import (
    interpolate_tcp_poses,
    matrix_to_rotvec,
)


CSV_PATH = Path(__file__).with_name("ndi_pose_native.csv")
ROBOT_HOST = "127.0.0.1"
ROBOT_PORT = 6001

# The CSV stores NDI translations in millimeters. UR TCP poses use meters.
NDI_TRANSLATION_SCALE_M = 0.001
HAND_EYE_TRANSLATION_SCALE_M = 0.001

# From hand-eye calibration:
# A = T_Flange_to_Reference (= ProbeMarker), printed in millimeters.
FLANGE_TO_REFERENCE_MM = np.array(
    [
        [-0.12938, -0.32922, 0.93535, 100.03402],
        [-0.21993, -0.91025, -0.35081, -2.23093],
        [0.96690, -0.25110, 0.04536, 22.27041],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# Conservative first-try limits for replaying uncalibrated marker motion.
MAX_TOTAL_POSITION_DELTA_M = 0.40
MAX_TOTAL_ROTATION_DELTA_RAD = 1.30
MAX_INTERPOLATION_POSITION_STEP_M = 0.002
MAX_INTERPOLATION_ROTATION_STEP_RAD = 0.03
COMMAND_PERIOD_S = 0.02
FRAME_DWELL_S = 0.04
ZMQ_TIMEOUT_MS = 3000


def project_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=float).reshape(3, 3)
    u, _s, vt = np.linalg.svd(value)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def project_transform_rotation(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=float).reshape(4, 4).copy()
    value[:3, :3] = project_rotation_matrix(value[:3, :3])
    value[3, :] = [0.0, 0.0, 0.0, 1.0]
    return value


def transform_mm_to_m(transform_mm: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform_mm, dtype=float).reshape(4, 4).copy()
    transform[:3, 3] *= HAND_EYE_TRANSLATION_SCALE_M
    return project_transform_rotation(transform)


FLANGE_TO_REFERENCE = transform_mm_to_m(FLANGE_TO_REFERENCE_MM)


def load_ndi_transforms(csv_path: Path = CSV_PATH) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(float(row.get("probe_visible", "1"))) != 1:
                continue
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = np.array(
                [
                    [float(row["m00"]), float(row["m01"]), float(row["m02"])],
                    [float(row["m10"]), float(row["m11"]), float(row["m12"])],
                    [float(row["m20"]), float(row["m21"]), float(row["m22"])],
                ],
                dtype=float,
            )
            transform[:3, 3] = np.array(
                [float(row["m03"]), float(row["m13"]), float(row["m23"])],
                dtype=float,
            ) * NDI_TRANSLATION_SCALE_M
            transforms.append(transform)
    if not transforms:
        raise ValueError(f"No visible NDI marker poses found in {csv_path}")
    return transforms


def relative_ball_transforms(transforms: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not transforms:
        raise ValueError("transforms must contain at least one pose")
    t0_inv = np.linalg.inv(np.asarray(transforms[0], dtype=float).reshape(4, 4))
    return [t0_inv @ np.asarray(transform, dtype=float).reshape(4, 4) for transform in transforms]


def conjugate_relative_transforms(
    relative_transforms: Sequence[np.ndarray],
    flange_to_marker: np.ndarray = FLANGE_TO_REFERENCE,
) -> list[np.ndarray]:
    x = np.asarray(flange_to_marker, dtype=float).reshape(4, 4)
    x_inv = np.linalg.inv(x)
    return [
        project_transform_rotation(x @ np.asarray(delta, dtype=float).reshape(4, 4) @ x_inv)
        for delta in relative_transforms
    ]


def matrix_from_tcp_pose(tcp_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(tcp_pose, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = rodrigues(pose[3:])
    return transform


def tcp_pose_from_matrix(transform: np.ndarray) -> np.ndarray:
    value = project_transform_rotation(transform)
    return np.concatenate([value[:3, 3], matrix_to_rotvec(value[:3, :3])])


def build_relative_targets(
    current_tcp_pose: np.ndarray,
    relative_transforms: Sequence[np.ndarray],
) -> list[np.ndarray]:
    start = matrix_from_tcp_pose(current_tcp_pose)
    return [tcp_pose_from_matrix(start @ np.asarray(delta, dtype=float).reshape(4, 4)) for delta in relative_transforms]


def validate_targets(targets: Sequence[np.ndarray], start_tcp_pose: np.ndarray) -> None:
    start = np.asarray(start_tcp_pose, dtype=float).reshape(6)
    start_rotation = rodrigues(start[3:])
    max_position_delta = 0.0
    max_rotation_delta = 0.0
    for target in targets:
        pose = np.asarray(target, dtype=float).reshape(6)
        max_position_delta = max(max_position_delta, float(np.linalg.norm(pose[:3] - start[:3])))
        relative_rotation = rodrigues(pose[3:]) @ start_rotation.T
        max_rotation_delta = max(max_rotation_delta, float(np.linalg.norm(matrix_to_rotvec(relative_rotation))))
    if max_position_delta > MAX_TOTAL_POSITION_DELTA_M:
        raise ValueError(
            f"Replay position delta {max_position_delta:.4f} m exceeds "
            f"MAX_TOTAL_POSITION_DELTA_M={MAX_TOTAL_POSITION_DELTA_M:.4f} m"
        )
    if max_rotation_delta > MAX_TOTAL_ROTATION_DELTA_RAD:
        raise ValueError(
            f"Replay rotation delta {max_rotation_delta:.4f} rad exceeds "
            f"MAX_TOTAL_ROTATION_DELTA_RAD={MAX_TOTAL_ROTATION_DELTA_RAD:.4f} rad"
        )


def interpolated_targets(targets: Sequence[np.ndarray]) -> Iterable[np.ndarray]:
    if not targets:
        return
    previous = np.asarray(targets[0], dtype=float).reshape(6)
    yield previous
    for target in targets[1:]:
        current = np.asarray(target, dtype=float).reshape(6)
        for waypoint in interpolate_tcp_poses(
            previous,
            current,
            max_position_step_m=MAX_INTERPOLATION_POSITION_STEP_M,
            max_rotation_step_rad=MAX_INTERPOLATION_ROTATION_STEP_RAD,
        ):
            yield waypoint
        previous = current


def get_current_tcp_pose(robot: ZMQClientRobot) -> np.ndarray:
    obs = robot.get_observations()
    if "ee_pos_rotvec" not in obs:
        raise RuntimeError("Robot observations do not include ee_pos_rotvec")
    return np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)


def replay(robot: ZMQClientRobot, targets: Sequence[np.ndarray]) -> None:
    try:
        for frame_index, target in enumerate(targets):
            for waypoint in interpolated_targets([targets[frame_index - 1], target] if frame_index else [target]):
                robot.command_tcp_pose(np.asarray(waypoint, dtype=float).reshape(6))
                time.sleep(COMMAND_PERIOD_S)
            time.sleep(FRAME_DWELL_S)
    finally:
        robot.stop_servo()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay NDI marker relative motion on the current UR TCP using "
            "the hardcoded T_Flange_to_Reference hand-eye conjugation. Start "
            "the robot server first with "
            "`python experiments/launch_nodes.py --robot ur`."
        )
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--host", default=ROBOT_HOST)
    parser.add_argument("--port", type=int, default=ROBOT_PORT)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-hand-eye", action="store_true", help="Use the old rough mode: no T_Flange_to_Reference conjugation.")
    parser.add_argument("--execute", action="store_true", help="Actually command the robot. Without this, only preview targets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transforms = load_ndi_transforms(args.csv)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    transforms = transforms[:: args.stride]
    if args.max_frames is not None:
        transforms = transforms[: args.max_frames]

    robot = ZMQClientRobot(port=args.port, host=args.host)
    try:
        robot._socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        robot._socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
        start_tcp_pose = get_current_tcp_pose(robot)
        marker_deltas = relative_ball_transforms(transforms)
        replay_deltas = (
            marker_deltas
            if args.no_hand_eye
            else conjugate_relative_transforms(marker_deltas)
        )
        targets = build_relative_targets(start_tcp_pose, replay_deltas)
        validate_targets(targets, start_tcp_pose)
        print(f"Loaded {len(transforms)} visible NDI frames from {args.csv}")
        print(
            "Hand-eye conjugation: "
            + ("disabled (--no-hand-eye)" if args.no_hand_eye else "enabled (T_Flange_to_Reference)")
        )
        if not args.no_hand_eye:
            print(f"T_Flange_to_Reference [m]: {FLANGE_TO_REFERENCE.tolist()}")
        print(f"Robot TCP first frame [x y z rx ry rz]: {start_tcp_pose.tolist()}")
        print(f"First target: {targets[0].tolist()}")
        print(f"Last target:  {targets[-1].tolist()}")
        if not args.execute:
            print("Preview only. Re-run with --execute to send servoL commands.")
            return
        replay(robot, targets)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
