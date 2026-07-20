from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breast_path_planning.geometry import rotvec_pose_to_transform
from breast_path_planning.path_io import PlannedPath, save_planned_path
from breast_path_planning.plan_from_frame import PlanFromFrameResult, plan_from_point_cloud
from breast_path_planning.pointcloud_from_d405 import (
    PointCloud,
    realsense_frames_to_point_cloud,
    rgbd_arrays_to_point_cloud,
    save_point_cloud_ply,
)
from breast_path_planning.segmentation import (
    SegmentationParams,
    segment_region_from_seed_indices,
)
from breast_path_planning.path_planner import PathPlannerParams
from breast_path_planning.surface_processing import estimate_normals
from visual_guided_collection_gui.images import depth_to_display_rgb


@dataclass
class FrozenFrame:
    rgb: np.ndarray
    depth: np.ndarray
    raw_cloud: PointCloud
    T_base_camera: np.ndarray
    robot_obs: dict[str, Any]
    camera_name: str
    pointcloud_backend: str


@dataclass
class SurfaceSegmentationResult:
    raw_cloud: PointCloud
    segmented_cloud: PointCloud
    normals_base: np.ndarray
    region_mask: np.ndarray


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (H, W, 3), got {image.shape}")
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image

        Image.fromarray(image.astype(np.uint8, copy=False)).save(path)


def _write_depth_png(path: Path, depth: np.ndarray) -> None:
    value = np.asarray(depth)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    try:
        import cv2

        cv2.imwrite(str(path), value)
    except Exception:
        from PIL import Image

        Image.fromarray(value).save(path)


def save_capture_rgb_depth(output_dir: Path, frame: FrozenFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(frame.rgb)
    depth = np.asarray(frame.depth)
    depth_vis = depth_to_display_rgb(depth)
    _write_rgb_png(output_dir / "capture_rgb.png", rgb)
    _write_rgb_png(output_dir / "capture_depth_vis.png", depth_vis)
    np.save(output_dir / "capture_depth_raw.npy", depth)
    _write_depth_png(output_dir / "capture_depth_raw.png", depth)


class PlanningSession:
    def __init__(
        self,
        *,
        t_tcp_camera_path: str | Path,
        output_root: str | Path,
        point_stride: int = 2,
        min_depth_m: float = 0.05,
        max_depth_m: float = 2.0,
        segmentation_params: SegmentationParams | None = None,
        planner_params: PathPlannerParams | None = None,
        capture_settle_s: float = 0.5,
    ) -> None:
        self.t_tcp_camera_path = Path(t_tcp_camera_path).expanduser()
        self.T_tcp_camera = np.load(self.t_tcp_camera_path)
        self.output_root = Path(output_root).expanduser()
        self.point_stride = int(point_stride)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.segmentation_params = segmentation_params or SegmentationParams()
        self.planner_params = planner_params or PathPlannerParams()
        self.capture_settle_s = float(capture_settle_s)
        self.frozen_frame: FrozenFrame | None = None
        self.plan_result: PlanFromFrameResult | None = None
        self.surface_result: SurfaceSegmentationResult | None = None
        self.output_dir: Path | None = None

    def capture(self, devices) -> FrozenFrame:
        if devices.camera is None or devices.robot_client is None:
            raise RuntimeError("Devices must be connected before capture")
        if self.capture_settle_s > 0.0:
            time.sleep(self.capture_settle_s)
        rgb, depth = devices.camera.read()
        robot_obs = devices.robot_client.get_observations()
        T_base_tcp = rotvec_pose_to_transform(robot_obs["ee_pos_rotvec"])
        T_base_camera = T_base_tcp @ self.T_tcp_camera
        camera_name = devices.wrist_camera_name
        if camera_name == "D405":
            color_frame, depth_frame = devices.latest_realsense_frames()
            raw_cloud = realsense_frames_to_point_cloud(
                color_frame,
                depth_frame,
                T_base_camera,
                stride=self.point_stride,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            pointcloud_backend = "librealsense"
        else:
            depth_scale = devices.camera_depth_scale_m_per_unit()
            raw_cloud = rgbd_arrays_to_point_cloud(
                rgb,
                depth,
                devices.camera_intrinsics(),
                T_base_camera,
                depth_scale_m_per_unit=depth_scale,
                stride=self.point_stride,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            pointcloud_backend = "rgbd_intrinsics"
        if len(raw_cloud) == 0:
            depth_values = np.asarray(depth)
            if depth_values.ndim == 3 and depth_values.shape[2] == 1:
                depth_values = depth_values[:, :, 0]
            nonzero = depth_values[depth_values > 0]
            if nonzero.size:
                min_raw = int(np.min(nonzero))
                max_raw = int(np.max(nonzero))
                scale = devices.camera_depth_scale_m_per_unit() if camera_name != "D405" else None
                extra = f"depth_raw_nonzero=[{min_raw}, {max_raw}]"
                if scale is not None:
                    extra += f", depth_scale_m_per_unit={scale:.8f}"
            else:
                extra = "no nonzero depth pixels"
            raise RuntimeError(
                f"{camera_name} produced 0 valid point cloud points after depth filter "
                f"[{self.min_depth_m}, {self.max_depth_m}] m; {extra}"
            )
        self.frozen_frame = FrozenFrame(
            rgb=np.asarray(rgb),
            depth=np.asarray(depth),
            raw_cloud=raw_cloud,
            T_base_camera=T_base_camera,
            robot_obs=dict(robot_obs),
            camera_name=camera_name,
            pointcloud_backend=pointcloud_backend,
        )
        return self.frozen_frame

    def plan_from_seed(self, seed_index: int, output_dir: str | Path | None = None) -> PlanFromFrameResult:
        if self.frozen_frame is None:
            raise RuntimeError("Capture a frame before planning")
        if output_dir is None:
            stamp = datetime.datetime.now().strftime("live_gui_%m%d_%H%M%S")
            output_dir = self.output_root / stamp
        self.output_dir = Path(output_dir)
        save_capture_rgb_depth(self.output_dir, self.frozen_frame)
        self.plan_result = plan_from_point_cloud(
            raw_cloud=self.frozen_frame.raw_cloud,
            seed_indices=[int(seed_index)],
            output_dir=self.output_dir,
            segmentation_params=self.segmentation_params,
            planner_params=self.planner_params,
            metadata={
                "source": "visual_guided_collection_gui",
                "point_stride": self.point_stride,
                "t_tcp_camera_path": str(self.t_tcp_camera_path),
                "camera_name": self.frozen_frame.camera_name,
                "pointcloud_backend": self.frozen_frame.pointcloud_backend,
            },
        )
        return self.plan_result

    def segment_from_seed(
        self,
        seed_index: int,
        output_dir: str | Path | None = None,
    ) -> SurfaceSegmentationResult:
        if self.frozen_frame is None:
            raise RuntimeError("Capture a frame before segmentation")
        if output_dir is None:
            stamp = datetime.datetime.now().strftime("live_bo_%m%d_%H%M%S")
            output_dir = self.output_root / stamp
        self.output_dir = Path(output_dir)
        save_capture_rgb_depth(self.output_dir, self.frozen_frame)
        segmented, region_mask = segment_region_from_seed_indices(
            self.frozen_frame.raw_cloud,
            [int(seed_index)],
            self.segmentation_params,
        )
        if len(segmented) < 10:
            raise RuntimeError(f"Segmentation produced too few points: {len(segmented)}")
        normals = estimate_normals(
            segmented.points_base,
            k_neighbors=self.planner_params.normal_k_neighbors,
            reference_direction=self.planner_params.normal_reference_direction,
        )
        self.surface_result = SurfaceSegmentationResult(
            raw_cloud=self.frozen_frame.raw_cloud,
            segmented_cloud=segmented,
            normals_base=normals,
            region_mask=region_mask,
        )
        self.plan_result = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_point_cloud_ply(self.frozen_frame.raw_cloud, self.output_dir / "raw_cloud_base.ply")
        save_point_cloud_ply(segmented, self.output_dir / "segmented_breast.ply")
        np.save(self.output_dir / "segmented_normals_base.npy", normals)
        np.save(self.output_dir / "region_mask.npy", region_mask)
        report = {
            "schema_version": "bo_surface_segmentation_v1",
            "seed_index": int(seed_index),
            "num_raw_points": len(self.frozen_frame.raw_cloud),
            "num_segmented_points": len(segmented),
            "camera_name": self.frozen_frame.camera_name,
            "pointcloud_backend": self.frozen_frame.pointcloud_backend,
        }
        (self.output_dir / "segmentation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.surface_result

    @property
    def planned_path(self) -> PlannedPath | None:
        return None if self.plan_result is None else self.plan_result.planned_path

    def replace_planned_path(
        self,
        planned_path: PlannedPath,
        *,
        backup_name: str = "planned_path_before_geodesic.json",
    ) -> Path:
        if self.plan_result is None:
            raise RuntimeError("No plan result exists to replace")
        if self.output_dir is None:
            raise RuntimeError("No output directory exists for the current plan")
        output_dir = Path(self.output_dir)
        backup_path = output_dir / backup_name
        if not backup_path.exists():
            save_planned_path(self.plan_result.planned_path, backup_path)
        self.plan_result.planned_path = planned_path
        output_path = output_dir / "planned_path.json"
        save_planned_path(planned_path, output_path)
        return output_path
