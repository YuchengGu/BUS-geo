import numpy as np

from breast_path_planning.geodesic_path import GeodesicPathParams, resample_path_with_surface_geodesics
from breast_path_planning.path_io import PlannedPath


def test_resample_path_with_surface_geodesics_follows_point_cloud_graph():
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    source_sink_path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=float),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
        metadata={"planner": "source_sink_test"},
    )

    geodesic = resample_path_with_surface_geodesics(
        source_sink_path,
        surface_points,
        surface_normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (len(surface_points), 1)),
        params=GeodesicPathParams(k_neighbors=2),
    )

    np.testing.assert_allclose(geodesic.positions_base, surface_points)
    np.testing.assert_allclose(geodesic.normals_base, np.tile([0.0, 0.0, 1.0], (4, 1)))
    assert geodesic.metadata["geodesic_resample"] is True
    assert geodesic.metadata["geodesic_source_path_points"] == 2

