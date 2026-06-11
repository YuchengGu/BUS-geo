from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from breast_path_planning.path_io import PlannedPath, save_planned_path
from breast_path_planning.geodesic_path import resample_path_with_surface_geodesics
from breast_path_planning.path_planner import PathPlannerParams, plan_serpentine_path
from breast_path_planning.pointcloud_from_d405 import PointCloud, realsense_frames_to_point_cloud, save_point_cloud_ply
from breast_path_planning.segmentation import SegmentationParams, segment_region_from_seed_indices, segment_region_from_seed_pixels
from breast_path_planning.surface_processing import estimate_normals


@dataclass
class PlanFromFrameResult:
    raw_cloud: PointCloud
    segmented_cloud: PointCloud
    planned_path: PlannedPath
    region_mask: np.ndarray


def plan_from_segmented_cloud(
    *,
    raw_cloud: PointCloud,
    segmented_cloud: PointCloud,
    region_mask: np.ndarray,
    seed_indices: Sequence[int],
    output_dir: str | Path | None = None,
    segmentation_params: SegmentationParams | None = None,
    planner_params: PathPlannerParams | None = None,
    metadata: dict[str, object] | None = None,
) -> PlanFromFrameResult:
    if len(segmented_cloud) < 10:
        raise RuntimeError(f"Segmentation produced too few points: {len(segmented_cloud)}")

    normals = estimate_normals(segmented_cloud.points_base)
    path_metadata = {
        "source": "point_cloud",
        "seed_indices": [int(i) for i in seed_indices],
        "num_raw_points": len(raw_cloud),
        "num_segmented_points": len(segmented_cloud),
    }
    if metadata:
        path_metadata.update(metadata)
    active_planner_params = planner_params or PathPlannerParams()
    serpentine_path = plan_serpentine_path(
        segmented_cloud.points_base,
        normals,
        active_planner_params,
        metadata=path_metadata,
    )
    planned_path = serpentine_path
    original_planned_path = None
    if active_planner_params.use_geodesic_resample:
        original_planned_path = serpentine_path
        planned_path = resample_path_with_surface_geodesics(
            serpentine_path,
            segmented_cloud.points_base,
            surface_normals_base=normals,
            metadata={
                "pre_geodesic_planner": serpentine_path.metadata.get("planner"),
            },
        )

    if output_dir is not None:
        _save_planning_outputs(
            output_dir=output_dir,
            raw_cloud=raw_cloud,
            segmented_cloud=segmented_cloud,
            planned_path=planned_path,
            original_planned_path=original_planned_path,
            segmentation_params=segmentation_params,
            planner_params=active_planner_params,
            report_extra={
                "seed_indices": [int(i) for i in seed_indices],
                **(metadata or {}),
            },
        )

    return PlanFromFrameResult(raw_cloud, segmented_cloud, planned_path, region_mask)


def plan_from_point_cloud(
    *,
    raw_cloud: PointCloud,
    seed_indices: Sequence[int],
    output_dir: str | Path | None = None,
    segmentation_params: SegmentationParams | None = None,
    planner_params: PathPlannerParams | None = None,
    metadata: dict[str, object] | None = None,
) -> PlanFromFrameResult:
    segmented_cloud, region_mask = segment_region_from_seed_indices(raw_cloud, seed_indices, segmentation_params)
    return plan_from_segmented_cloud(
        raw_cloud=raw_cloud,
        segmented_cloud=segmented_cloud,
        region_mask=region_mask,
        seed_indices=seed_indices,
        output_dir=output_dir,
        segmentation_params=segmentation_params,
        planner_params=planner_params,
        metadata=metadata,
    )


def plan_from_frame(
    *,
    color_frame: object,
    depth_frame: object,
    T_base_camera: np.ndarray,
    seed_pixels: Sequence[tuple[int, int]],
    output_dir: str | Path | None = None,
    pointcloud: object | None = None,
    point_stride: int = 2,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
    segmentation_params: SegmentationParams | None = None,
    planner_params: PathPlannerParams | None = None,
) -> PlanFromFrameResult:
    """Plan a breast scan path from current RealSense frames.

    The D405 intrinsics are consumed inside librealsense pointcloud generation.
    Callers should not pass or maintain a separate camera intrinsic matrix here.
    """
    raw_cloud = realsense_frames_to_point_cloud(
        color_frame,
        depth_frame,
        T_base_camera,
        pointcloud=pointcloud,
        stride=point_stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    if len(raw_cloud) == 0:
        raise RuntimeError("RealSense SDK produced no valid point cloud points")

    segmented_cloud, region_mask = segment_region_from_seed_pixels(raw_cloud, seed_pixels, segmentation_params)
    metadata = {
        "source": "D405_realsense_pointcloud",
        "point_stride": point_stride,
        "seed_pixels": [[int(u), int(v)] for u, v in seed_pixels],
        "num_raw_points": len(raw_cloud),
        "num_segmented_points": len(segmented_cloud),
        "pointcloud_backend": "librealsense",
    }
    return plan_from_segmented_cloud(
        raw_cloud=raw_cloud,
        segmented_cloud=segmented_cloud,
        region_mask=region_mask,
        seed_indices=[],
        output_dir=output_dir,
        segmentation_params=segmentation_params,
        planner_params=planner_params,
        metadata=metadata,
    )


def _planner_params_to_dict(params: PathPlannerParams) -> dict[str, object]:
    data = asdict(params)
    data["normal_reference_direction"] = np.asarray(params.normal_reference_direction, dtype=float).tolist()
    return data


def _save_planning_outputs(
    *,
    output_dir: str | Path,
    raw_cloud: PointCloud,
    segmented_cloud: PointCloud,
    planned_path: PlannedPath,
    original_planned_path: PlannedPath | None = None,
    segmentation_params: SegmentationParams | None = None,
    planner_params: PathPlannerParams | None = None,
    report_extra: dict[str, object] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_point_cloud_ply(raw_cloud, output / "raw_cloud_base.ply")
    save_point_cloud_ply(segmented_cloud, output / "segmented_breast.ply")
    if original_planned_path is not None:
        save_planned_path(original_planned_path, output / "planned_path_serpentine.json")
    save_planned_path(planned_path, output / "planned_path.json")
    report = {
        "num_raw_points": len(raw_cloud),
        "num_segmented_points": len(segmented_cloud),
        "num_path_points": len(planned_path),
        "segmentation_params": asdict(segmentation_params or SegmentationParams()),
        "planner_params": _planner_params_to_dict(planner_params or PathPlannerParams()),
        "requires_user_camera_intrinsics": False,
    }
    if report_extra:
        report.update(report_extra)
    with open(output / "planning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
