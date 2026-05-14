from __future__ import annotations

import numpy as np

from breast_path_planning.pointcloud_from_d405 import PointCloud
from breast_path_planning.segmentation import SegmentationParams, segment_region_from_seed_indices


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for 3D point cloud seed selection: pip install open3d") from exc
    return o3d


def point_cloud_to_open3d(cloud: PointCloud, *, highlight_mask: np.ndarray | None = None):
    o3d = _import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points_base)

    if highlight_mask is None:
        if cloud.colors_rgb is None:
            colors = np.full((len(cloud), 3), 0.65, dtype=float)
        else:
            colors = cloud.colors_rgb.astype(float) / 255.0
    else:
        mask = np.asarray(highlight_mask, dtype=bool)
        if mask.shape[0] != len(cloud):
            raise ValueError("highlight_mask length must match point cloud length")
        if cloud.colors_rgb is None:
            colors = np.full((len(cloud), 3), 0.35, dtype=float)
        else:
            colors = cloud.colors_rgb.astype(float) / 255.0 * 0.25
        colors[mask] = np.array([1.0, 0.05, 0.05])

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pick_seed_index_from_point_cloud(cloud: PointCloud, *, window_name: str = "Pick breast seed point") -> int:
    o3d = _import_open3d()
    print("Open3D seed selection:")
    print("  Shift + left click: pick one seed point")
    print("  Shift + right click: undo picked point")
    print("  Q or close window: finish picking")
    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name=window_name)
    visualizer.add_geometry(point_cloud_to_open3d(cloud))
    visualizer.run()
    visualizer.destroy_window()
    picked = visualizer.get_picked_points()
    if not picked:
        raise RuntimeError("No point was picked")
    return int(picked[-1])


def show_segmentation_highlight(cloud: PointCloud, region_mask: np.ndarray, *, window_name: str = "Segmented breast preview") -> None:
    o3d = _import_open3d()
    print("Showing highlighted segmentation. Red points are the segmented breast region.")
    print("Close the Open3D window, then answer in the terminal.")
    o3d.visualization.draw_geometries([point_cloud_to_open3d(cloud, highlight_mask=region_mask)], window_name=window_name)


def ask_segmentation_decision() -> str:
    while True:
        answer = input("Accept this segmentation? [y] plan / [r] re-pick seed / [q] cancel: ").strip().lower()
        if answer in {"y", "r", "q"}:
            return answer
        print("Please type y, r, or q.")


def interactive_segment_point_cloud(
    cloud: PointCloud,
    *,
    segmentation_params: SegmentationParams | None = None,
) -> tuple[int, PointCloud, np.ndarray]:
    if cloud.colors_rgb is None:
        raise ValueError("Interactive segmentation requires an RGB point cloud. The PLY must contain red/green/blue colors.")

    while True:
        seed_index = pick_seed_index_from_point_cloud(cloud)
        segmented_cloud, region_mask = segment_region_from_seed_indices(cloud, [seed_index], segmentation_params)
        show_segmentation_highlight(cloud, region_mask)
        decision = ask_segmentation_decision()
        if decision == "y":
            return seed_index, segmented_cloud, region_mask
        if decision == "q":
            raise RuntimeError("Segmentation cancelled")
