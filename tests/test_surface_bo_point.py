import numpy as np

from breast_path_planning.path_io import PlannedPath
from breast_path_planning.pointcloud_from_d405 import PointCloud
from visual_guided_collection_gui.surface_bo_point import (
    select_random_path_bo_reference,
    select_random_surface_reference,
)


def test_random_path_bo_reference_uses_planned_path_point_and_tangent():
    path = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.1, 0.0],
            ],
            dtype=float,
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
        metadata={"path_variant_method": "geodesic"},
    )

    selected = select_random_path_bo_reference(path, rng=np.random.default_rng(1))

    assert 0 <= selected.path_index < len(path)
    np.testing.assert_allclose(
        selected.position_base,
        path.positions_base[selected.path_index],
    )
    np.testing.assert_allclose(selected.normal_base, [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(selected.tangent_base), 1.0)


def test_random_surface_reference_builds_single_point_path_without_serpentine_planning():
    cloud = PointCloud(
        points_base=np.array(
            [
                [0.10, 0.20, 0.30],
                [0.11, 0.20, 0.31],
                [0.12, 0.20, 0.32],
            ],
            dtype=float,
        ),
        colors_rgb=np.zeros((3, 3), dtype=np.uint8),
        pixels_uv=np.zeros((3, 2), dtype=np.int32),
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1))

    selected = select_random_surface_reference(
        cloud,
        normals,
        rng=np.random.default_rng(4),
    )

    assert len(selected.reference_path) == 1
    np.testing.assert_allclose(
        selected.reference_path.positions_base[0],
        cloud.points_base[selected.cloud_index],
    )
    np.testing.assert_allclose(selected.reference_path.normals_base[0], [0.0, 0.0, 1.0])
    assert selected.reference_path.metadata["path_variant_method"] == "bo_random_surface_point"
