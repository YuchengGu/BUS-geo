import json

import numpy as np

from breast_path_planning.path_features import PathFeatureParams, compute_path_features
from breast_path_planning.path_io import PlannedPath, load_planned_path, save_planned_path
from breast_path_planning.path_planner import PathPlannerParams, plan_serpentine_path
from breast_path_planning.plan_from_frame import plan_from_point_cloud
from breast_path_planning.pointcloud_from_d405 import PointCloud, load_point_cloud_ply, realsense_frames_to_point_cloud, rgbd_arrays_to_point_cloud, save_point_cloud_ply
from breast_path_planning.segmentation import SegmentationParams, segment_region_from_seed_pixels
from breast_path_planning.live_plan_from_d405 import compute_base_camera_transform


class FakeFrame:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class FakeRsPoints:
    def __init__(self, vertices, texcoords):
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._texcoords = np.asarray(texcoords, dtype=np.float32)

    def get_vertices(self):
        return self._vertices

    def get_texture_coordinates(self):
        return self._texcoords


class FakePointCloudGenerator:
    def __init__(self, vertices, texcoords):
        self._points = FakeRsPoints(vertices, texcoords)
        self.mapped_to = None
        self.calculated_from = None

    def map_to(self, color_frame):
        self.mapped_to = color_frame

    def calculate(self, depth_frame):
        self.calculated_from = depth_frame
        return self._points


def test_realsense_frames_to_point_cloud_uses_sdk_vertices_and_texture_mapping():
    color = np.array(
        [
            [[10, 0, 0], [20, 0, 0]],
            [[30, 0, 0], [40, 0, 0]],
        ],
        dtype=np.uint8,
    )
    color_frame = FakeFrame(color)
    depth_frame = FakeFrame(np.zeros((2, 2), dtype=np.uint16))
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 0.01],  # filtered by min_depth_m
        ],
        dtype=np.float32,
    )
    texcoords = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    generator = FakePointCloudGenerator(vertices, texcoords)
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    cloud = realsense_frames_to_point_cloud(
        color_frame,
        depth_frame,
        transform,
        pointcloud=generator,
        stride=1,
        min_depth_m=0.05,
    )

    assert generator.mapped_to is color_frame
    assert generator.calculated_from is depth_frame
    assert len(cloud) == 2
    np.testing.assert_allclose(cloud.points_base[0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(cloud.colors_rgb[0], [10, 0, 0])
    np.testing.assert_allclose(cloud.colors_rgb[1], [40, 0, 0])
    np.testing.assert_allclose(cloud.pixels_uv, [[0, 0], [1, 1]])


def test_rgbd_arrays_to_point_cloud_uses_intrinsics_and_depth_scale():
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[[1000], [2000]], [[0], [3000]]], dtype=np.uint16)
    intrinsics = {"fx": 100.0, "fy": 200.0, "cx": 0.0, "cy": 0.0}

    cloud = rgbd_arrays_to_point_cloud(
        rgb,
        depth,
        intrinsics,
        np.eye(4),
        depth_scale_m_per_unit=0.001,
        stride=1,
        min_depth_m=0.05,
        max_depth_m=2.5,
    )

    assert len(cloud) == 2
    np.testing.assert_allclose(cloud.points_base, [[0.0, 0.0, 1.0], [0.02, 0.0, 2.0]])
    np.testing.assert_array_equal(cloud.colors_rgb, [[255, 0, 0], [0, 255, 0]])
    np.testing.assert_array_equal(cloud.pixels_uv, [[0, 0], [1, 0]])


def test_point_cloud_ply_roundtrip_preserves_xyz_and_rgb(tmp_path):
    cloud = PointCloud(
        points_base=np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),
        colors_rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
    )
    output = tmp_path / "cloud.ply"

    save_point_cloud_ply(cloud, output)
    loaded = load_point_cloud_ply(output)

    np.testing.assert_allclose(loaded.points_base, cloud.points_base)
    np.testing.assert_array_equal(loaded.colors_rgb, cloud.colors_rgb)
    assert loaded.pixels_uv is None


def test_binary_little_endian_ply_loads_xyz_and_rgb(tmp_path):
    output = tmp_path / "binary_cloud.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertices = np.array(
        [
            (0.0, 0.1, 0.2, 10, 20, 30),
            (0.3, 0.4, 0.5, 40, 50, 60),
        ],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    output.write_bytes(header + vertices.tobytes())

    loaded = load_point_cloud_ply(output)

    np.testing.assert_allclose(loaded.points_base, [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], atol=1e-7)
    np.testing.assert_array_equal(loaded.colors_rgb, [[10, 20, 30], [40, 50, 60]])


def test_compute_base_camera_transform_uses_current_tcp_pose_and_hand_eye_transform():
    ee_pos_rotvec = np.array([0.5, -0.2, 0.1, 0.0, 0.0, np.pi / 2.0])
    t_tcp_camera = np.eye(4)
    t_tcp_camera[:3, 3] = [0.1, 0.2, 0.3]

    t_base_camera = compute_base_camera_transform(ee_pos_rotvec, t_tcp_camera)

    expected = np.array(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(t_base_camera, expected, atol=1e-8)


def test_plan_from_point_cloud_uses_seed_indices_and_writes_outputs(tmp_path):
    xs = np.linspace(0.0, 0.03, 4)
    ys = np.linspace(0.0, 0.02, 3)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    colors = np.tile(np.array([[200, 120, 100]], dtype=np.uint8), (len(points), 1))
    cloud = PointCloud(points, colors)

    result = plan_from_point_cloud(
        raw_cloud=cloud,
        seed_indices=[0],
        output_dir=tmp_path,
        segmentation_params=SegmentationParams(spatial_radius_m=0.02, max_distance_from_seed_m=0.1),
        planner_params=PathPlannerParams(step_y_m=0.01, step_x_m=0.01, min_points_per_slice=3),
    )

    assert len(result.segmented_cloud) == len(points)
    assert len(result.planned_path) > 0
    assert (tmp_path / "raw_cloud_base.ply").exists()
    assert (tmp_path / "segmented_breast.ply").exists()
    assert (tmp_path / "planned_path.json").exists()
    assert (tmp_path / "planning_report.json").exists()


def test_plan_from_point_cloud_keeps_serpentine_path_by_default(tmp_path):
    xs = np.linspace(0.0, 0.03, 4)
    ys = np.linspace(0.0, 0.02, 3)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    colors = np.tile(np.array([[200, 120, 100]], dtype=np.uint8), (len(points), 1))
    cloud = PointCloud(points, colors)

    result = plan_from_point_cloud(
        raw_cloud=cloud,
        seed_indices=[0],
        output_dir=tmp_path,
        segmentation_params=SegmentationParams(spatial_radius_m=0.02, max_distance_from_seed_m=0.1),
        planner_params=PathPlannerParams(step_y_m=0.01, step_x_m=0.01, min_points_per_slice=3),
    )

    assert result.planned_path.metadata["geodesic_resample"] is False
    assert result.planned_path.metadata["planner"] == "adaptive_slice_serpentine_v1"
    assert not (tmp_path / "planned_path_serpentine.json").exists()
    assert (tmp_path / "planned_path.json").exists()


def test_region_growing_segments_seed_colored_component():
    points = np.array(
        [
            [0.00, 0.00, 0.0],
            [0.01, 0.00, 0.0],
            [0.02, 0.00, 0.0],
            [0.20, 0.00, 0.0],
        ]
    )
    colors = np.array(
        [
            [200, 120, 100],
            [202, 122, 101],
            [201, 121, 100],
            [20, 200, 20],
        ],
        dtype=np.uint8,
    )
    pixels = np.array([[0, 0], [1, 0], [2, 0], [20, 0]])
    cloud = PointCloud(points, colors, pixels)

    segmented, mask = segment_region_from_seed_pixels(
        cloud,
        [(0, 0)],
        SegmentationParams(spatial_radius_m=0.02, max_distance_from_seed_m=0.05),
    )

    assert len(segmented) == 3
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_serpentine_planner_outputs_positions_and_normals_only():
    xs = np.linspace(0.0, 0.02, 3)
    ys = np.linspace(0.0, 0.02, 3)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))

    path = plan_serpentine_path(
        points,
        normals,
        PathPlannerParams(
            step_y_m=0.01,
            step_x_m=0.01,
            slice_tolerance_ratio=0.35,
            min_points_per_slice=3,
        ),
    )

    assert len(path) > 0
    assert path.positions_base[0, 0] < path.positions_base[1, 0]
    assert np.all(np.isin(path.positions_base[:, 0], xs))
    assert np.all(np.isin(path.positions_base[:, 1], ys))
    np.testing.assert_allclose(path.normals_base, np.tile([0.0, 0.0, 1.0], (len(path), 1)))
    assert path.metadata["contains_tangent"] is False
    assert path.metadata["contains_reverse_y"] is False
    assert path.metadata["normal_constrain_enabled"] is False


def test_serpentine_planner_records_row_endpoint_corner_indices():
    xs = np.linspace(0.0, 0.02, 3)
    ys = np.linspace(0.0, 0.02, 3)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))

    path = plan_serpentine_path(
        points,
        normals,
        PathPlannerParams(
            step_y_m=0.01,
            step_x_m=0.01,
            slice_tolerance_ratio=0.35,
            min_points_per_slice=3,
        ),
    )

    assert path.metadata["corner_indices"] == [1, 2]


def test_serpentine_planner_uses_smaller_scan_area_and_fewer_points_by_default():
    xs = np.linspace(0.0, 0.04, 21)
    ys = np.linspace(0.0, 0.04, 21)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))

    path = plan_serpentine_path(
        points,
        normals,
        PathPlannerParams(
            step_y_m=0.005,
            step_x_m=0.005,
            slice_tolerance_ratio=0.35,
            min_points_per_slice=3,
        ),
    )

    assert float(np.min(path.positions_base[:, 0])) >= 0.004 - 1e-9
    assert float(np.max(path.positions_base[:, 0])) <= 0.036 + 1e-9
    assert float(np.min(path.positions_base[:, 1])) >= 0.004 - 1e-9
    assert float(np.max(path.positions_base[:, 1])) <= 0.036 + 1e-9
    assert len(np.unique(path.positions_base[:, 1])) <= 7


def test_serpentine_planner_preserves_steep_normals_by_default():
    xs = np.linspace(0.0, 0.02, 3)
    ys = np.linspace(0.0, 0.02, 3)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    steep = np.array([1.0, 0.0, 0.0])
    normals = np.tile(steep, (len(points), 1))

    path = plan_serpentine_path(
        points,
        normals,
        PathPlannerParams(
            step_y_m=0.01,
            step_x_m=0.01,
            slice_tolerance_ratio=0.35,
            min_points_per_slice=3,
        ),
    )

    np.testing.assert_allclose(path.normals_base, np.tile(steep, (len(path), 1)))


def test_planned_path_json_roundtrip_excludes_tangent_and_reverse_y(tmp_path):
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        metadata={"source": "test"},
    )
    output = tmp_path / "planned_path.json"

    save_planned_path(path, output)
    data = json.loads(output.read_text())

    assert set(data["points"][0]) == {"index", "position_base", "normal_base"}
    loaded = load_planned_path(output)
    np.testing.assert_allclose(loaded.positions_base, path.positions_base)
    np.testing.assert_allclose(loaded.normals_base, path.normals_base)
    assert loaded.metadata["source"] == "test"


def test_compute_path_features_uses_position_and_normal():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        normals_base=np.tile(np.array([0.0, 0.0, 1.0]), (3, 1)),
    )

    features = compute_path_features(
        path,
        np.array([0.011, 0.0, 0.0]),
        params=PathFeatureParams(lookahead=4),
    )

    assert features["path_nearest_index"] == 1
    np.testing.assert_allclose(features["path_residuals_base"][0], [-0.001, 0.0, 0.0])
    np.testing.assert_array_equal(features["path_lookahead_mask"], [True, True, False, False])
    np.testing.assert_allclose(features["path_normals_base"][0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(features["path_reference_tcp_rotvecs_base"][0], [0.0, np.pi, 0.0], atol=1e-8)
    assert "path_tangents_base" not in features
    assert "path_darboux_frames_base" not in features
