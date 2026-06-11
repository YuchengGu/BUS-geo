import numpy as np

from breast_path_planning.geometry import rodrigues
from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.surface_teleop import (
    DEFAULT_APPROACH_HEIGHT_M,
    DEFAULT_CONTACT_HEIGHT_M,
    DEFAULT_PROBE_LENGTH_M,
    SurfaceCartesianTeleopController,
    SurfaceTeleopState,
    build_tcp_frame,
    build_tcp_target,
    choose_tcp_frame_axis_mode_for_rotation,
    first_darboux_scan_line_tcp_poses,
    gello_rotation_increment_to_tcp_local,
    gello_translation_to_surface_delta,
    interpolate_tcp_poses,
    matrix_to_rotvec,
    path_start_tcp_targets,
    staged_surface_start_tcp_sequence,
)


def test_surface_teleop_defaults_match_requested_first_version():
    assert DEFAULT_APPROACH_HEIGHT_M == 0.2
    assert DEFAULT_CONTACT_HEIGHT_M == 0.05
    assert DEFAULT_PROBE_LENGTH_M == 0.2


def test_build_tcp_frame_defaults_tcp_x_to_world_y_projection_and_negative_normal():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    frame = build_tcp_frame(tangent, normal)

    np.testing.assert_allclose(frame[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(frame[:, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(frame[:, 2], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-10)
    assert np.linalg.det(frame) > 0.999999


def test_build_tcp_frame_world_y_mode_projects_world_y_to_surface_tangent_plane():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    frame = build_tcp_frame(tangent, normal, frame_axis_mode="world-y")

    np.testing.assert_allclose(frame[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(frame[:, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(frame[:, 2], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-10)
    assert np.linalg.det(frame) > 0.999999


def test_build_tcp_frame_world_y_mode_keeps_tcp_x_in_world_y_normal_plane():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.3, 0.0, 0.9539392014169457])

    frame = build_tcp_frame(tangent, normal, frame_axis_mode="world-y")

    world_y = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(frame[:, 0], world_y, atol=1e-10)
    np.testing.assert_allclose(np.dot(frame[:, 0], normal), 0.0, atol=1e-10)
    np.testing.assert_allclose(frame[:, 2], -normal / np.linalg.norm(normal), atol=1e-10)


def test_choose_tcp_frame_axis_mode_for_rotation_can_pick_world_y_sign_without_180_flip():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    current = build_tcp_frame(tangent, normal, frame_axis_mode="-world-y")

    mode = choose_tcp_frame_axis_mode_for_rotation(
        -tangent,
        normal,
        current,
        allowed_modes=("world-y", "-world-y"),
    )

    assert mode == "-world-y"


def test_choose_tcp_frame_axis_mode_for_rotation_respects_allowed_modes():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    current = build_tcp_frame(tangent, normal, frame_axis_mode="-b")

    scan_mode = choose_tcp_frame_axis_mode_for_rotation(
        tangent,
        normal,
        current,
        allowed_modes=("b", "-b"),
    )
    corner_mode = choose_tcp_frame_axis_mode_for_rotation(
        tangent,
        normal,
        current,
        allowed_modes=("t", "-t"),
    )

    assert scan_mode == "-b"
    assert corner_mode in {"t", "-t"}


def test_gello_translation_maps_xyz_to_t_minus_b_n():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    delta = gello_translation_to_surface_delta(
        np.array([0.01, 0.02, 0.03]),
        tangent,
        normal,
    )

    # b = t x n = [0, -1, 0], so -b is world +y.
    np.testing.assert_allclose(delta, [0.01, 0.02, 0.03])


def test_gello_rotations_map_to_swapped_tcp_xy_and_negative_z_rotations():
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base = build_tcp_frame(tangent, normal)
    x_angle = np.deg2rad(10.0)
    y_angle = np.deg2rad(7.0)
    z_angle = np.deg2rad(5.0)

    delta_tcp = gello_rotation_increment_to_tcp_local(np.array([x_angle, y_angle, z_angle]))
    right_multiplied = base @ rodrigues(delta_tcp)

    tcp_x_axis = base[:, 0]
    tcp_y_axis = base[:, 1]
    tcp_z_axis = base[:, 2]
    left_multiplied = rodrigues(y_angle * tcp_x_axis + x_angle * tcp_y_axis - z_angle * tcp_z_axis) @ base

    np.testing.assert_allclose(delta_tcp, [y_angle, x_angle, -z_angle])
    np.testing.assert_allclose(right_multiplied, left_multiplied, atol=1e-10)


def test_build_tcp_target_places_tcp_outward_from_probe_tip():
    state = SurfaceTeleopState(progress_m=0.0, lateral_offset_m=0.01, normal_offset_m=0.005)
    path_position = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    target = build_tcp_target(path_position, tangent, normal, state)

    # b = [0, -1, 0], so lateral +0.01 moves the probe tip in -y.
    np.testing.assert_allclose(target.probe_tip_position_base, [0.0, -0.01, 0.005])
    np.testing.assert_allclose(target.tcp_position_base, [0.0, -0.01, 0.205])
    np.testing.assert_allclose(target.tcp_rotation_base, build_tcp_frame(tangent, normal))


def test_path_start_tcp_targets_use_approach_contact_and_probe_length():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )

    pre, start = path_start_tcp_targets(path)

    np.testing.assert_allclose(pre.probe_tip_position_base, [0.0, 0.0, DEFAULT_APPROACH_HEIGHT_M])
    np.testing.assert_allclose(pre.tcp_position_base, [0.0, 0.0, DEFAULT_APPROACH_HEIGHT_M + DEFAULT_PROBE_LENGTH_M])
    np.testing.assert_allclose(start.probe_tip_position_base, [0.0, 0.0, DEFAULT_CONTACT_HEIGHT_M])
    np.testing.assert_allclose(start.tcp_position_base, [0.0, 0.0, DEFAULT_CONTACT_HEIGHT_M + DEFAULT_PROBE_LENGTH_M])


def test_interpolate_tcp_poses_limits_position_and_rotation_steps():
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.2])

    poses = interpolate_tcp_poses(
        start,
        target,
        max_position_step_m=0.01,
        max_rotation_step_rad=0.1,
    )

    assert len(poses) == 3
    np.testing.assert_allclose(poses[-1], target)
    deltas = np.diff(np.vstack([start, poses]), axis=0)
    assert np.max(np.linalg.norm(deltas[:, :3], axis=1)) <= 0.0100001
    assert np.max(np.linalg.norm(deltas[:, 3:], axis=1)) <= 0.1000001


def test_interpolate_tcp_poses_uses_so3_short_rotation_path_across_rotvec_branch():
    start = np.array([0.0, 0.0, 0.0, -0.3278002941538622, -3.062080844194702, 0.4860290223082538])
    target = np.array([0.0, 0.0, 0.0, -0.11151441, 2.61743103, 0.63090761])

    poses = interpolate_tcp_poses(
        start,
        target,
        max_position_step_m=0.01,
        max_rotation_step_rad=0.05,
    )

    assert len(poses) < 25
    rotations = [rodrigues(start[3:])] + [rodrigues(pose[3:]) for pose in poses]
    step_angles = []
    for previous, current in zip(rotations[:-1], rotations[1:]):
        step_angles.append(np.linalg.norm(matrix_to_rotvec(current @ previous.T)))
    assert max(step_angles) <= 0.050001
    np.testing.assert_allclose(poses[-1], target, atol=1e-10)


def test_staged_surface_start_sequence_translates_halfway_before_rotating():
    current = np.array([0.1, -0.2, 0.3, 0.01, 0.02, 0.03])
    pre = build_tcp_target(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        SurfaceTeleopState(normal_offset_m=0.2),
        probe_length_m=0.2,
    )
    start = build_tcp_target(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        SurfaceTeleopState(normal_offset_m=0.05),
        probe_length_m=0.2,
    )

    mid_translate_pose, mid_rotate_pose, pre_pose, start_pose = staged_surface_start_tcp_sequence(current, pre, start)
    target_rotvec = start.tcp_pose_rotvec()[3:]
    expected_mid_position = 0.5 * (current[:3] + pre.tcp_position_base)

    np.testing.assert_allclose(mid_translate_pose[:3], expected_mid_position)
    np.testing.assert_allclose(mid_translate_pose[3:], current[3:])
    np.testing.assert_allclose(mid_rotate_pose[:3], expected_mid_position)
    np.testing.assert_allclose(mid_rotate_pose[3:], target_rotvec)
    np.testing.assert_allclose(pre_pose[:3], pre.tcp_position_base)
    np.testing.assert_allclose(pre_pose[3:], target_rotvec)
    np.testing.assert_allclose(start_pose, start.tcp_pose_rotvec())


def test_surface_cartesian_controller_uses_calibrated_axes_for_translation_and_rotation():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    start_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    start_tcp_pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    controller.set_neutral(
        gello_tcp_pose=np.zeros(6),
        ur_tcp_pose=start_tcp_pose,
    )
    controller.calibrate_x(np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)

    target = controller.update(
        gello_tcp_pose=np.array([-0.02, 0.01, 0.0, -0.2, 0.1, 0.0]),
        ur_tcp_pose=start_tcp_pose,
    )

    np.testing.assert_allclose(target[:3], [0.01, 0.02, 0.25], atol=1e-10)
    expected_rotation = start_rotation @ rodrigues(np.array([0.2, 0.1, 0.0]))
    np.testing.assert_allclose(rodrigues(target[3:]), expected_rotation, atol=1e-10)


def test_surface_cartesian_controller_requires_x_and_z_calibration_before_motion():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=np.zeros(6))

    try:
        controller.update(gello_tcp_pose=np.ones(6) * 0.01, ur_tcp_pose=np.zeros(6))
    except RuntimeError as exc:
        assert "Calibrate +X and +Z" in str(exc)
    else:
        raise AssertionError("expected missing calibration to block surface motion")


def test_surface_cartesian_controller_follows_nearest_path_orientation_without_rotation_input():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    )
    start_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[0])
    start_tcp_pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)

    target = controller.update(
        gello_tcp_pose=np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ur_tcp_pose=start_tcp_pose,
    )

    expected_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[1])
    np.testing.assert_allclose(target[:3], [0.2, 0.25, 0.0], atol=1e-10)
    np.testing.assert_allclose(rodrigues(target[3:]), expected_rotation, atol=1e-10)


def test_surface_cartesian_controller_uses_fixed_frame_axis_mode_after_confirm():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    pre, _start = path_start_tcp_targets(
        path,
        preferred_tcp_x_axis_base=np.array([0.0, -1.0, 0.0]),
    )
    assert pre.frame_axis_mode == "-world-y"
    start_rotation = pre.tcp_rotation_base
    start_tcp_pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2, frame_axis_mode=pre.frame_axis_mode)
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)

    target = controller.update(
        gello_tcp_pose=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ur_tcp_pose=start_tcp_pose,
    )

    np.testing.assert_allclose(rodrigues(target[3:])[:, 0], [0.0, -1.0, 0.0], atol=1e-10)


def test_path_start_tcp_target_keeps_tcp_x_on_world_y_projection():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )

    pre, _start = path_start_tcp_targets(
        path,
        preferred_tcp_x_axis_base=np.array([0.0, 1.0, 0.0]),
    )

    assert pre.frame_axis_mode == "world-y"
    x_axis = pre.tcp_rotation_base[:, 0]
    np.testing.assert_allclose(x_axis, [0.0, 1.0, 0.0], atol=1e-10)


def test_first_darboux_scan_line_tcp_poses_include_all_path_points_with_turnbacks():
    path = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.0],
                [0.1, 0.2, 0.0],
                [0.2, 0.2, 0.0],
                [0.2, 0.3, 0.0],
                [0.1, 0.3, 0.0],
                [0.0, 0.3, 0.0],
            ]
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (12, 1)),
        metadata={"corner_indices": [2, 3, 5, 6, 8, 9]},
    )

    poses = first_darboux_scan_line_tcp_poses(
        path,
        contact_height_m=0.05,
        probe_length_m=0.2,
        frame_axis_mode="world-y",
    )

    assert len(poses) == 12
    np.testing.assert_allclose(poses[2][:3], [0.2, 0.0, 0.25])
    np.testing.assert_allclose(poses[3][:3], [0.2, 0.1, 0.25])
    np.testing.assert_allclose(poses[5][:3], [0.0, 0.1, 0.25])
    np.testing.assert_allclose(poses[6][:3], [0.0, 0.2, 0.25])
    np.testing.assert_allclose(poses[8][:3], [0.2, 0.2, 0.25])
    np.testing.assert_allclose(poses[9][:3], [0.2, 0.3, 0.25])
    np.testing.assert_allclose(poses[-1][:3], [0.0, 0.3, 0.25])
    last_rotation = rodrigues(poses[-1][3:])
    np.testing.assert_allclose(last_rotation[:, 0], [0.0, 1.0, 0.0], atol=1e-10)


def test_surface_cartesian_controller_keeps_world_y_x_sign_through_corner():
    path = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
        metadata={"corner_indices": [2]},
    )
    scan_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[0], frame_axis_mode="-world-y")
    scan_tcp_pose = np.concatenate([[0.1, 0.0, 0.2], matrix_to_rotvec(scan_rotation)])
    controller = SurfaceCartesianTeleopController(
        path=path,
        probe_length_m=0.2,
        frame_axis_mode="-world-y",
        use_corner_frame_modes=True,
    )
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=scan_tcp_pose)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=scan_tcp_pose)

    scan_target = controller.update(gello_tcp_pose=np.zeros(6), ur_tcp_pose=scan_tcp_pose)
    corner_tcp_pose = scan_tcp_pose.copy()
    corner_tcp_pose[:3] = [0.2, 0.0, 0.2]
    corner_target = controller.update(gello_tcp_pose=np.zeros(6), ur_tcp_pose=corner_tcp_pose)

    scan_x = rodrigues(scan_target[3:])[:, 0]
    corner_x = rodrigues(corner_target[3:])[:, 0]
    np.testing.assert_allclose(scan_x, [0.0, -1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(corner_x, [0.0, -1.0, 0.0], atol=1e-10)


def test_surface_cartesian_controller_uses_absolute_gello_translation_without_accumulating():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    start_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[0])
    start_tcp_pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)

    first = controller.update(
        gello_tcp_pose=np.array([0.03, 0.01, 0.02, 0.0, 0.0, 0.0]),
        ur_tcp_pose=start_tcp_pose,
    )
    held = controller.update(
        gello_tcp_pose=np.array([0.03, 0.01, 0.02, 0.0, 0.0, 0.0]),
        ur_tcp_pose=first,
    )
    returned = controller.update(
        gello_tcp_pose=np.zeros(6),
        ur_tcp_pose=held,
    )

    np.testing.assert_allclose(first[:3], [0.03, 0.01, 0.27], atol=1e-10)
    np.testing.assert_allclose(held[:3], first[:3], atol=1e-10)
    np.testing.assert_allclose(returned[:3], start_tcp_pose[:3], atol=1e-10)


def test_surface_cartesian_controller_maps_absolute_x_to_path_arclength_reference_with_smoothed_corner_tangent():
    path = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.1, 0.0],
            ]
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
    )
    start_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[0])
    start_tcp_pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)

    target = controller.update(
        gello_tcp_pose=np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ur_tcp_pose=start_tcp_pose,
    )

    expected_tangent = np.array([0.3826834323650898, 0.9238795325112867, 0.0])
    expected_rotation = build_tcp_frame(expected_tangent, path.normals_base[1])
    np.testing.assert_allclose(target[:3], [0.1, 0.05, 0.25], atol=1e-10)
    np.testing.assert_allclose(rodrigues(target[3:]), expected_rotation, atol=1e-10)


def test_surface_cartesian_controller_uses_absolute_gello_rotation_without_accumulating():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    start_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[0])
    start_tcp_pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=start_tcp_pose)

    rotated = controller.update(
        gello_tcp_pose=np.array([0.0, 0.0, 0.0, 0.1, 0.2, -0.3]),
        ur_tcp_pose=start_tcp_pose,
    )
    held = controller.update(
        gello_tcp_pose=np.array([0.0, 0.0, 0.0, 0.1, 0.2, -0.3]),
        ur_tcp_pose=rotated,
    )
    returned = controller.update(
        gello_tcp_pose=np.zeros(6),
        ur_tcp_pose=held,
    )

    expected_rotation = start_rotation @ rodrigues(np.array([0.2, 0.1, 0.3]))
    np.testing.assert_allclose(rodrigues(rotated[3:]), expected_rotation, atol=1e-10)
    np.testing.assert_allclose(rodrigues(held[3:]), expected_rotation, atol=1e-10)
    np.testing.assert_allclose(rodrigues(returned[3:]), start_rotation, atol=1e-10)
