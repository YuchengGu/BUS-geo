import numpy as np

from EXPERIMENT.plot_geodesic_case_profiles import (
    case_profile_output_stems,
    normalized_arclength,
    pointwise_displacement_mm,
    resolve_input_dirs,
)
from breast_path_planning.path_smoothing import moving_average_smooth_path


def test_normalized_arclength_maps_polyline_to_unit_interval():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    values = normalized_arclength(points)

    np.testing.assert_allclose(values, [0.0, 3.0 / 7.0, 1.0])


def test_pointwise_displacement_mm_uses_corresponding_path_points():
    before = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]], dtype=float)
    after = np.array([[0.0, 0.002, 0.0], [0.001, 0.003, 0.004]], dtype=float)

    values = pointwise_displacement_mm(before, after)

    np.testing.assert_allclose(values, [2.0, 5.0])


def test_moving_average_smooth_path_preserves_endpoints():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    smoothed = moving_average_smooth_path(points, window=3, passes=1)

    np.testing.assert_allclose(smoothed[0], points[0])
    np.testing.assert_allclose(smoothed[-1], points[-1])
    assert smoothed.shape == points.shape


def test_case_profile_output_stems_are_two_independent_figures(tmp_path):
    stems = case_profile_output_stems(tmp_path, "case_profile")

    assert stems["kg"] == tmp_path / "case_profile" / "kg"
    assert stems["displacement"] == tmp_path / "case_profile" / "displacement"


def test_case_profile_resolve_input_dirs_filters_valid_cases(tmp_path):
    valid = tmp_path / "live_gui_valid"
    valid.mkdir()
    (valid / "planned_path_before_geodesic.json").write_text("{}", encoding="utf-8")
    (valid / "planned_path.json").write_text("{}", encoding="utf-8")
    invalid = tmp_path / "live_gui_invalid"
    invalid.mkdir()

    resolved = resolve_input_dirs(
        configured_dir=None,
        configured_dirs=[],
        input_glob=str(tmp_path / "live_gui_*"),
    )

    assert resolved == [valid]
