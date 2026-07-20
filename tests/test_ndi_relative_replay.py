from __future__ import annotations

import numpy as np

from ndi_relative_replay.replay_relative_ndi import (
    FLANGE_TO_REFERENCE,
    build_relative_targets,
    conjugate_relative_transforms,
    load_ndi_transforms,
    matrix_from_tcp_pose,
    relative_ball_transforms,
    tcp_pose_from_matrix,
    validate_targets,
)


def _transform(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def test_relative_ball_transforms_use_initial_inverse_on_left():
    t0 = _transform(np.eye(3), [1.0, 2.0, 3.0])
    ti = _transform(np.eye(3), [1.2, 1.9, 3.4])

    deltas = relative_ball_transforms([t0, ti])

    np.testing.assert_allclose(deltas[0], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(deltas[1], np.linalg.inv(t0) @ ti, atol=1e-12)


def test_conjugate_relative_transforms_maps_marker_delta_into_flange_frame():
    flange_to_marker = _transform(np.eye(3), [0.1, 0.0, 0.0])
    marker_delta = _transform(np.eye(3), [0.0, 0.02, 0.0])

    deltas = conjugate_relative_transforms([marker_delta], flange_to_marker)

    expected = flange_to_marker @ marker_delta @ np.linalg.inv(flange_to_marker)
    np.testing.assert_allclose(deltas[0], expected, atol=1e-12)


def test_build_relative_targets_right_multiplies_current_tcp_with_supplied_delta():
    current_tcp = np.array([0.5, -0.1, 0.25, 0.0, 0.0, np.pi / 2.0])
    delta = _transform(np.eye(3), [0.01, 0.02, 0.0])

    targets = build_relative_targets(current_tcp, [np.eye(4), delta])

    expected = matrix_from_tcp_pose(current_tcp) @ delta
    np.testing.assert_allclose(matrix_from_tcp_pose(targets[0]), matrix_from_tcp_pose(current_tcp), atol=1e-12)
    np.testing.assert_allclose(matrix_from_tcp_pose(targets[1]), expected, atol=1e-12)
    np.testing.assert_allclose(targets[1], tcp_pose_from_matrix(expected), atol=1e-12)


def test_conjugated_replay_targets_validate_for_near_pi_start_pose():
    current_tcp = np.array(
        [
            -0.02033324857719026,
            0.07479597259631347,
            0.08509672909755428,
            1.746864594091489,
            1.664155337238736,
            -1.7000166251618265,
        ],
        dtype=float,
    )

    marker_deltas = relative_ball_transforms(load_ndi_transforms())
    replay_deltas = conjugate_relative_transforms(marker_deltas, FLANGE_TO_REFERENCE)
    targets = build_relative_targets(current_tcp, replay_deltas)

    validate_targets(targets, current_tcp)
