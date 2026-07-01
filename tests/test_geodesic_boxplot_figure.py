from pathlib import Path

import numpy as np

from EXPERIMENT.plot_geodesic_boxplots import boxplot_output_stems, compute_method_metrics


def test_compute_method_metrics_reports_curvature_and_displacement():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    reference = positions.copy()
    shifted = positions.copy()
    shifted[:, 2] += 0.001

    metrics = compute_method_metrics(shifted, normals, reference)

    assert metrics["kg_squared_sum"] >= 0.0
    assert metrics["max_abs_kg"] >= 0.0
    np.testing.assert_allclose(metrics["mean_displacement_mm"], 1.0)


def test_boxplot_output_stems_returns_four_independent_figures(tmp_path):
    stems = boxplot_output_stems(tmp_path)

    assert stems["kg_squared_sum"] == tmp_path / "kg_squared_sum_boxplot"
    assert stems["max_abs_kg"] == tmp_path / "max_abs_kg_boxplot"
    assert stems["mean_displacement_mm"] == tmp_path / "mean_displacement_boxplot"
    assert stems["runtime_s"] == tmp_path / "runtime_boxplot"
