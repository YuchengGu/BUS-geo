import numpy as np

from breast_path_planning.path_smoothing import b_spline_smooth_path, moving_average_smooth_path


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


def test_moving_average_smooth_path_makes_even_window_odd():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 0.0],
            [3.0, 6.0, 0.0],
            [4.0, 8.0, 0.0],
        ],
        dtype=float,
    )

    even = moving_average_smooth_path(points, window=4, passes=1)
    odd = moving_average_smooth_path(points, window=5, passes=1)

    np.testing.assert_allclose(even, odd)


def test_b_spline_smooth_path_preserves_shape_and_endpoints():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0],
            [0.2, -0.1, 0.0],
            [0.3, 0.2, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=float,
    )

    smoothed = b_spline_smooth_path(points, smoothing_factor=1e-4)

    assert smoothed.shape == points.shape
    np.testing.assert_allclose(smoothed[0], points[0])
    np.testing.assert_allclose(smoothed[-1], points[-1])
