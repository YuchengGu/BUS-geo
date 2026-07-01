from pathlib import Path

import numpy as np

from EXPERIMENT.plot_geodesic_progress_figure import (
    AXIS_TICK_LABEL_SIZE,
    aggregate_metric_curves,
    individual_panel_stem,
    normalize_to_initial,
    separate_panel_stems,
)


def test_normalize_to_initial_uses_first_finite_value():
    values = np.array([2.0, 1.0, 0.5], dtype=float)

    normalized = normalize_to_initial(values)

    np.testing.assert_allclose(normalized, [1.0, 0.5, 0.25])


def test_aggregate_metric_curves_interpolates_to_common_progress_grid():
    curves = [
        {"progress": [0.0, 0.5, 1.0], "metric": [1.0, 0.5, 0.0]},
        {"progress": [0.0, 1.0], "metric": [1.0, 0.0]},
    ]

    grid, stack, stats = aggregate_metric_curves(curves, "metric", num_grid=3)

    np.testing.assert_allclose(grid, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(stack, [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    np.testing.assert_allclose(stats["mean"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(stats["q25"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(stats["q75"], [1.0, 0.5, 0.0])


def test_individual_panel_stem_uses_metric_name_under_panels_dir():
    stem = individual_panel_stem(Path("out/geodesic_progress_figure"), "E_over_E0")

    assert stem == Path("out/panels/E_over_E0")


def test_separate_panel_stems_returns_four_panel_outputs():
    stems = separate_panel_stems(Path("out/geodesic_progress_figure"))

    assert len(stems) == 4
    assert all(stem.parent == Path("out/panels") for stem in stems.values())


def test_axis_tick_label_size_is_smaller_than_base_font():
    assert AXIS_TICK_LABEL_SIZE == 6.5
