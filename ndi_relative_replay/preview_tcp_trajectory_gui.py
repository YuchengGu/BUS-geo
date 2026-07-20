from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from breast_path_planning.geometry import rodrigues
from gello.zmq_core.robot_node import ZMQClientRobot
from ndi_relative_replay.replay_relative_ndi import (
    CSV_PATH,
    FLANGE_TO_REFERENCE,
    build_relative_targets,
    conjugate_relative_transforms,
    get_current_tcp_pose,
    load_ndi_transforms,
    relative_ball_transforms,
    validate_targets,
)


ROBOT_HOST = "127.0.0.1"
ROBOT_PORT = 6001
ZMQ_TIMEOUT_MS = 3000
DEFAULT_AXIS_LENGTH_M = 0.035
DEFAULT_AXIS_STRIDE = 25
DEFAULT_FRAME_PERIOD_S = 0.05


def make_path_line_data(tcp_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(tcp_poses, dtype=float).reshape(-1, 6)
    points = poses[:, :3].copy()
    if len(points) < 2:
        return points, np.zeros((0, 2), dtype=int)
    lines = np.column_stack(
        [np.arange(len(points) - 1, dtype=int), np.arange(1, len(points), dtype=int)]
    )
    return points, lines


def make_axis_line_data(
    tcp_poses: np.ndarray,
    *,
    axis_length_m: float = DEFAULT_AXIS_LENGTH_M,
    stride: int = DEFAULT_AXIS_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = np.asarray(tcp_poses, dtype=float).reshape(-1, 6)
    if len(poses) == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 2), dtype=int),
            np.zeros((0, 3), dtype=float),
        )
    step = max(1, int(stride))
    sampled = poses[::step]
    if not np.allclose(sampled[-1], poses[-1]):
        sampled = np.vstack([sampled, poses[-1]])

    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []
    axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 1.0])
    for pose in sampled:
        origin = pose[:3]
        rotation = rodrigues(pose[3:])
        base = len(points)
        points.append(origin)
        for axis_index, color in enumerate(axis_colors):
            points.append(origin + rotation[:, axis_index] * float(axis_length_m))
            lines.append([base, base + axis_index + 1])
            colors.append(list(color))
    return np.asarray(points, dtype=float), np.asarray(lines, dtype=int), np.asarray(colors, dtype=float)


def tcp_axis_lineset(
    tcp_pose: np.ndarray,
    *,
    axis_length_m: float = DEFAULT_AXIS_LENGTH_M,
) -> o3d.geometry.LineSet:
    points, lines, colors = make_axis_line_data(
        np.asarray(tcp_pose, dtype=float).reshape(1, 6),
        axis_length_m=axis_length_m,
        stride=1,
    )
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def build_preview_targets(
    start_tcp_pose: np.ndarray,
    *,
    csv_path: Path = CSV_PATH,
    stride: int = 1,
    max_frames: int | None = None,
    use_hand_eye: bool = True,
) -> np.ndarray:
    transforms = load_ndi_transforms(csv_path)
    step = max(1, int(stride))
    transforms = transforms[::step]
    if max_frames is not None:
        transforms = transforms[: int(max_frames)]
    marker_deltas = relative_ball_transforms(transforms)
    replay_deltas = (
        conjugate_relative_transforms(marker_deltas, FLANGE_TO_REFERENCE)
        if use_hand_eye
        else marker_deltas
    )
    targets = np.asarray(build_relative_targets(start_tcp_pose, replay_deltas), dtype=float)
    validate_targets(targets, start_tcp_pose)
    return targets


def get_robot_start_tcp_pose(host: str, port: int) -> np.ndarray:
    robot = ZMQClientRobot(host=host, port=port)
    try:
        robot._socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        robot._socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
        return get_current_tcp_pose(robot)
    finally:
        robot.close()


def make_line_set(
    points: np.ndarray,
    lines: np.ndarray,
    color: list[float],
) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    if len(lines):
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (len(lines), 1)))
    return line_set


def run_preview(
    tcp_poses: np.ndarray,
    *,
    axis_length_m: float,
    axis_stride: int,
    frame_period_s: float,
) -> None:
    poses = np.asarray(tcp_poses, dtype=float).reshape(-1, 6)
    path_points, path_lines = make_path_line_data(poses)
    sampled_axis_points, sampled_axis_lines, sampled_axis_colors = make_axis_line_data(
        poses,
        axis_length_m=axis_length_m,
        stride=axis_stride,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="NDI Relative TCP Replay Preview", width=1400, height=900)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([1.0, 1.0, 1.0])
    render_opt.line_width = 3.0
    render_opt.point_size = 7.0

    path_geom = make_line_set(path_points, path_lines, [0.05, 0.05, 0.05])
    sampled_axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(sampled_axis_points),
        lines=o3d.utility.Vector2iVector(sampled_axis_lines),
    )
    sampled_axes.colors = o3d.utility.Vector3dVector(sampled_axis_colors)
    start_axes = tcp_axis_lineset(poses[0], axis_length_m=axis_length_m * 1.6)
    current_axes = tcp_axis_lineset(poses[0], axis_length_m=axis_length_m * 2.2)

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length_m * 0.18)
    start_sphere.paint_uniform_color([1.0, 0.25, 0.05])
    start_sphere.translate(poses[0, :3])
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length_m * 0.18)
    end_sphere.paint_uniform_color([0.05, 0.25, 1.0])
    end_sphere.translate(poses[-1, :3])

    for geom in (path_geom, sampled_axes, start_axes, current_axes, start_sphere, end_sphere):
        vis.add_geometry(geom)

    bounds = path_geom.get_axis_aligned_bounding_box()
    if not bounds.is_empty():
        center = bounds.get_center()
        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front([0.0, -1.0, 0.35])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.7)

    print("Open3D preview controls: mouse to rotate/zoom. Close the window to stop.")
    print("Red/green/blue axes are TCP x/y/z. Orange sphere is frame 0; blue sphere is final frame.")
    print(f"Displaying {len(poses)} target TCP frames.")
    frame_index = 0
    last_update = 0.0
    while vis.poll_events():
        now = time.monotonic()
        if now - last_update >= float(frame_period_s):
            vis.remove_geometry(current_axes, reset_bounding_box=False)
            current_axes = tcp_axis_lineset(poses[frame_index], axis_length_m=axis_length_m * 2.2)
            vis.add_geometry(current_axes, reset_bounding_box=False)
            print(f"\rframe {frame_index + 1}/{len(poses)} tcp={poses[frame_index].tolist()}", end="", flush=True)
            frame_index = (frame_index + 1) % len(poses)
            last_update = now
        vis.update_renderer()
        time.sleep(0.005)
    print()
    vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an Open3D preview of the NDI relative replay trajectory. "
            "This reads the current UR5 TCP once from the robot server and never sends robot commands."
        )
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--host", default=ROBOT_HOST)
    parser.add_argument("--port", type=int, default=ROBOT_PORT)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--axis-stride", type=int, default=DEFAULT_AXIS_STRIDE)
    parser.add_argument("--axis-length-m", type=float, default=DEFAULT_AXIS_LENGTH_M)
    parser.add_argument("--frame-period-s", type=float, default=DEFAULT_FRAME_PERIOD_S)
    parser.add_argument("--no-hand-eye", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_tcp_pose = get_robot_start_tcp_pose(args.host, args.port)
    targets = build_preview_targets(
        start_tcp_pose,
        csv_path=args.csv,
        stride=args.stride,
        max_frames=args.max_frames,
        use_hand_eye=not args.no_hand_eye,
    )
    print(f"Robot current TCP used as frame 0: {start_tcp_pose.tolist()}")
    print(f"First target: {targets[0].tolist()}")
    print(f"Last target:  {targets[-1].tolist()}")
    run_preview(
        targets,
        axis_length_m=args.axis_length_m,
        axis_stride=args.axis_stride,
        frame_period_s=args.frame_period_s,
    )


if __name__ == "__main__":
    main()
