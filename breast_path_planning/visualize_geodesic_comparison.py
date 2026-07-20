#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.path_io import PlannedPath, load_planned_path
from breast_path_planning.pointcloud_from_d405 import PointCloud, load_point_cloud_ply


ORIGINAL_COLOR = np.array([1.0, 0.05, 0.05], dtype=float)
OPTIMIZED_COLOR = np.array([0.0, 0.8, 0.1], dtype=float)
NORMAL_COLOR = np.array([0.1, 0.25, 1.0], dtype=float)


def _import_open3d_gui():
    try:
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
    except ImportError as exc:
        raise ImportError("Open3D is required for path visualization: pip install open3d") from exc
    return o3d, gui, rendering


def path_display_color(path: PlannedPath) -> np.ndarray:
    metadata = dict(getattr(path, "metadata", {}) or {})
    method = metadata.get("path_variant_method")
    if metadata.get("geodesic_trigger") == "gui_optimize_geodesic" or metadata.get("geodesic_resample"):
        return OPTIMIZED_COLOR
    if method == "moving_average":
        return np.array([0.55, 0.55, 0.55], dtype=float)
    if method == "b_spline":
        return np.array([0.1, 0.35, 0.9], dtype=float)
    return ORIGINAL_COLOR


def make_path_line_set(o3d, path: PlannedPath, *, color: np.ndarray):
    positions = np.asarray(path.positions_base, dtype=float)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    if len(path) >= 2:
        lines = np.array([[i, i + 1] for i in range(len(path) - 1)], dtype=int)
    else:
        lines = np.zeros((0, 2), dtype=int)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1, 3), (len(lines), 1)))
    return line_set


def make_path_points(o3d, path: PlannedPath):
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(np.asarray(path.positions_base, dtype=float))
    color = path_display_color(path)
    points.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1, 3), (len(path), 1)))
    return points


def make_normal_line_set(
    o3d,
    path: PlannedPath,
    *,
    normal_step: int,
    normal_length_m: float,
    color: np.ndarray,
):
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
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1, 3), (n, 1)))
    return line_set


def comparison_geometries(
    *,
    cloud: PointCloud,
    original_path: PlannedPath,
    optimized_path: PlannedPath,
    normal_step: int,
    normal_length_m: float,
    show_normals: bool,
):
    o3d, _gui, _rendering = _import_open3d_gui()
    cloud_geometry = o3d.geometry.PointCloud()
    cloud_geometry.points = o3d.utility.Vector3dVector(cloud.points_base)
    if cloud.colors_rgb is not None:
        cloud_geometry.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors_rgb, dtype=float) / 255.0)
    geometries = {
        "segmented_cloud": cloud_geometry,
        "planned_path_lines": make_path_line_set(o3d, original_path, color=path_display_color(original_path)),
        "optimized_path_lines": make_path_line_set(o3d, optimized_path, color=path_display_color(optimized_path)),
        "optimized_path_points": make_path_points(o3d, optimized_path),
    }
    if show_normals:
        geometries["optimized_path_normals"] = make_normal_line_set(
                o3d,
                optimized_path,
                normal_step=normal_step,
                normal_length_m=normal_length_m,
                color=NORMAL_COLOR,
        )
    return o3d, geometries


def add_gui_geometry(scene, rendering, name: str, geometry) -> None:
    material = rendering.MaterialRecord()
    if name.endswith("_lines") or name.endswith("_normals"):
        material.shader = "unlitLine"
        material.line_width = 3.0
    elif name.endswith("_points"):
        material.shader = "defaultUnlit"
        material.point_size = 9.0
    else:
        material.shader = "defaultUnlit"
        material.point_size = 3.0
    scene.add_geometry(name, geometry, material)


def visualize_comparison(
    *,
    cloud: PointCloud,
    original_path: PlannedPath,
    optimized_path: PlannedPath,
    normal_step: int,
    normal_length_m: float,
    show_normals: bool,
    save_png: Path | None,
    window_name: str,
) -> None:
    o3d, gui, rendering = _import_open3d_gui()
    _o3d, geometries = comparison_geometries(
        cloud=cloud,
        original_path=original_path,
        optimized_path=optimized_path,
        normal_step=normal_step,
        normal_length_m=normal_length_m,
        show_normals=show_normals,
    )
    print("Open3D visualization:")
    print("  original path before geodesic: red")
    print("  optimized geodesic path: green")
    print("  optimized normals: hidden by default")
    print("  close window: exit")
    if save_png is not None:
        print(f"  S: save screenshot to {save_png}")

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)

    app = gui.Application.instance
    app.initialize()
    window = app.create_window(window_name, 1600, 1000)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    bounds_source = geometries["segmented_cloud"]
    for name, geometry in geometries.items():
        add_gui_geometry(scene_widget.scene, rendering, name, geometry)
    bounds = bounds_source.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _on_layout(_context):
        scene_widget.frame = window.content_rect

    window.set_on_layout(_on_layout)

    if save_png is not None:
        def _save_callback(_event):
            key = getattr(_event, "key", None)
            if key not in {ord("S"), ord("s")} and str(key).lower() not in {"s", "keyname.s"}:
                return gui.Widget.EventCallbackResult.IGNORED
            image = app.render_to_image(scene_widget.scene, 1600, 1000)
            o3d.io.write_image(str(save_png), image)
            print(f"saved screenshot: {save_png}")
            return gui.Widget.EventCallbackResult.HANDLED

        window.set_on_key(_save_callback)

    app.run()


def default_case_dir() -> Path:
    result_root = Path("breast_path_planning/results")
    candidates = sorted(result_root.glob("live_gui_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (
            (candidate / "segmented_breast.ply").exists()
            and (candidate / "planned_path_before_geodesic.json").exists()
            and (candidate / "planned_path.json").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "No live_gui_* directory with segmented_breast.ply, planned_path_before_geodesic.json, "
        "and planned_path.json was found."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize original clinical path and geodesic optimized path on the same segmented cloud."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Directory containing segmented_breast.ply, planned_path_before_geodesic.json, and planned_path.json.",
    )
    parser.add_argument("--cloud-ply", type=Path, default=None, help="Override cloud PLY path.")
    parser.add_argument("--original-path", type=Path, default=None, help="Override original path JSON.")
    parser.add_argument("--optimized-path", type=Path, default=None, help="Override optimized path JSON.")
    parser.add_argument("--show-normals", action="store_true", help="Show optimized path normals in blue.")
    parser.add_argument("--normal-step", type=int, default=5, help="Show one normal every N optimized path points.")
    parser.add_argument("--normal-length-m", type=float, default=0.02)
    parser.add_argument("--save-png", type=Path, default=None, help="If set, press S in the Open3D window to save.")
    parser.add_argument("--window-name", default="Original path vs geodesic optimized path")
    args = parser.parse_args()

    case_dir = args.case_dir or default_case_dir()
    cloud_ply = args.cloud_ply or (case_dir / "segmented_breast.ply")
    original_path = args.original_path or (case_dir / "planned_path_before_geodesic.json")
    optimized_path = args.optimized_path or (case_dir / "planned_path.json")

    missing = [p for p in [cloud_ply, original_path, optimized_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(str(p) for p in missing))

    print(f"case_dir: {case_dir}")
    print(f"cloud: {cloud_ply}")
    print(f"original: {original_path}")
    print(f"optimized: {optimized_path}")

    visualize_comparison(
        cloud=load_point_cloud_ply(cloud_ply),
        original_path=load_planned_path(original_path),
        optimized_path=load_planned_path(optimized_path),
        normal_step=args.normal_step,
        normal_length_m=args.normal_length_m,
        show_normals=args.show_normals,
        save_png=args.save_png,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
