import subprocess
import sys
from pathlib import Path

import numpy as np

from EXPERIMENT.geodesic_real_robot.plot_coverage_area import (
    PATH_SAMPLE_SPACING_M,
    PLOT_METHOD_ORDER,
    build_mesh_adjacency,
    covered_triangle_area,
    densify_polyline,
    multi_source_mesh_distances,
)


def test_coverage_plot_uses_fine_sampling_and_excludes_original():
    assert PATH_SAMPLE_SPACING_M == 0.0008
    assert PLOT_METHOD_ORDER == ("moving_average", "b_spline", "geodesic")


def test_script_can_run_directly_outside_repo(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "EXPERIMENT"
        / "geodesic_real_robot"
        / "plot_coverage_area.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "Compute mesh-union probe coverage" in result.stdout


def test_densify_polyline_limits_sample_spacing():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])

    dense = densify_polyline(points, spacing_m=0.001)

    np.testing.assert_allclose(dense[[0, -1]], points)
    assert np.max(np.linalg.norm(np.diff(dense, axis=0), axis=1)) <= 0.001 + 1e-12


def test_mesh_distance_and_covered_area_use_surface_edges():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    adjacency = build_mesh_adjacency(vertices, triangles)

    distances = multi_source_mesh_distances(adjacency, source_indices=np.array([0]))
    area, covered = covered_triangle_area(
        vertices,
        triangles,
        distances,
        radius_m=0.8,
    )

    np.testing.assert_allclose(distances, [0.0, 1.0, 1.0, 2.0])
    assert covered == 1
    assert area == 0.5
