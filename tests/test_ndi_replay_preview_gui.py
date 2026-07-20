from __future__ import annotations

import numpy as np

from ndi_relative_replay.preview_tcp_trajectory_gui import (
    make_axis_line_data,
    make_path_line_data,
)


def test_make_path_line_data_connects_tcp_positions_in_order():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    points, lines = make_path_line_data(poses)

    np.testing.assert_allclose(points, poses[:, :3])
    np.testing.assert_array_equal(lines, [[0, 1], [1, 2]])


def test_make_axis_line_data_samples_xyz_axes_from_tcp_poses():
    poses = np.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, np.pi / 2.0],
        ],
        dtype=float,
    )

    points, lines, colors = make_axis_line_data(poses, axis_length_m=0.1, stride=1)

    assert points.shape == (8, 3)
    assert lines.shape == (6, 2)
    assert colors.shape == (6, 3)
    np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(points[1], [1.1, 2.0, 3.0])
    np.testing.assert_allclose(points[2], [1.0, 2.1, 3.0])
    np.testing.assert_allclose(points[3], [1.0, 2.0, 3.1])
    np.testing.assert_allclose(colors[:3], [[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 1.0]])
