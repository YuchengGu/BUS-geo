#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.interactive_pointcloud import point_cloud_to_open3d
from breast_path_planning.path_io import PlannedPath, load_planned_path
from breast_path_planning.pointcloud_from_d405 import PointCloud, load_point_cloud_ply


def make_path_line_set(path: PlannedPath):
    o3d = _import_open3d()
    positions = np.asarray(path.positions_base, dtype=float)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    if len(path) >= 2:
        lines = np.array([[i, i + 1] for i in range(len(path) - 1)], dtype=int)
    else:
        lines = np.zeros((0, 2), dtype=int)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 1.0, 0.0]]), (len(lines), 1)))
    return line_set


def make_path_points(path: PlannedPath):
    o3d = _import_open3d()
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(path.positions_base)
    points.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.7, 0.0]]), (len(path), 1)))
    return points


def make_normal_line_set(path: PlannedPath, *, normal_step: int = 1, normal_length_m: float = 0.02):
    o3d = _import_open3d()
    step = max(1, int(normal_step))
    positions = np.asarray(path.positions_base[::step], dtype=float)
    normals = np.asarray(path.normals_base[::step], dtype=float)
    starts = positions
    ends = positions + normals * float(normal_length_m)
    points = np.concatenate([starts, ends], axis=0)
    n = starts.shape[0]
    lines = np.array([[i, i + n] for i in range(n)], dtype=int)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.2, 1.0]]), (n, 1)))
    return line_set


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for path visualization: pip install open3d") from exc
    return o3d


def visualize_path_with_cloud(
    *,
    cloud: PointCloud,
    path: PlannedPath,
    normal_step: int = 1,
    normal_length_m: float = 0.02,
    window_name: str = "Segmented cloud with planned path and normals",
) -> None:
    o3d = _import_open3d()
    geometries = [
        point_cloud_to_open3d(cloud),
        make_path_line_set(path),
        make_path_points(path),
        make_normal_line_set(path, normal_step=normal_step, normal_length_m=normal_length_m),
    ]
    print("Open3D visualization:")
    print("  segmented cloud: original RGB colors")
    print("  path line: green")
    print("  path points: yellow")
    print("  normals: blue")
    print("  Q or close window: exit")
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview segmented point cloud, planned path points, and normals in one Open3D window.")
    parser.add_argument("--cloud-ply", required=True, help="Segmented or raw PLY point cloud.")
    parser.add_argument("--planned-path", required=True, help="planned_path.json.")
    parser.add_argument("--normal-step", type=int, default=1, help="Show one normal every N path points.")
    parser.add_argument("--normal-length-m", type=float, default=0.02)
    parser.add_argument("--window-name", default="Segmented cloud with planned path and normals")
    args = parser.parse_args()

    visualize_path_with_cloud(
        cloud=load_point_cloud_ply(args.cloud_ply),
        path=load_planned_path(args.planned_path),
        normal_step=args.normal_step,
        normal_length_m=args.normal_length_m,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
