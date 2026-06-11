import numpy as np
from pathlib import Path

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.surface_teleop import SurfaceCartesianTeleopController


def test_recenter_updates_gello_reference_without_moving_ur_target():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    target = np.array([0.0, 0.0, 0.25, np.pi, 0.0, 0.0])

    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    recentered = controller.recenter(
        gello_tcp_pose=np.array([0.2, -0.1, 0.05, 0.0, 0.0, 0.0]),
        ur_tcp_pose=target,
    )
    moved = controller.update(
        gello_tcp_pose=np.array([0.21, -0.1, 0.05, 0.0, 0.0, 0.0]),
        ur_tcp_pose=target,
    )

    np.testing.assert_allclose(recentered, target)
    np.testing.assert_allclose(moved[:3], target[:3] + [0.01, 0.0, 0.0], atol=1e-10)


def test_recenter_synchronizes_target_to_current_ur_tcp_pose():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    old_target = np.array([0.0, 0.0, 0.25, np.pi, 0.0, 0.0])
    current_tcp = np.array([0.04, 0.01, 0.25, np.pi, 0.0, 0.0])

    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=old_target)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    recentered = controller.recenter(
        gello_tcp_pose=np.array([0.2, -0.1, 0.05, 0.0, 0.0, 0.0]),
        ur_tcp_pose=current_tcp,
    )
    held = controller.update(
        gello_tcp_pose=np.array([0.2, -0.1, 0.05, 0.0, 0.0, 0.0]),
        ur_tcp_pose=current_tcp,
    )

    np.testing.assert_allclose(recentered, current_tcp)
    np.testing.assert_allclose(held, current_tcp, atol=1e-10)


def test_clutch_recenters_continuously_without_moving_ur_target():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    target = np.array([0.0, 0.0, 0.25, np.pi, 0.0, 0.0])
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)

    controller.set_clutch(True)
    held = controller.update(
        gello_tcp_pose=np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ur_tcp_pose=target,
    )
    controller.set_clutch(False)
    moved = controller.update(
        gello_tcp_pose=np.array([0.31, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ur_tcp_pose=target,
    )

    np.testing.assert_allclose(held, target)
    np.testing.assert_allclose(moved[:3], target[:3] + [0.01, 0.0, 0.0], atol=1e-10)


def test_clutch_release_uses_release_pose_as_new_neutral_and_allows_motion():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    target = np.array([0.0, 0.0, 0.25, np.pi, 0.0, 0.0])
    release_gello = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])

    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)
    controller.set_clutch(True)
    controller.update(gello_tcp_pose=release_gello, ur_tcp_pose=target)

    controller.recenter(gello_tcp_pose=release_gello, ur_tcp_pose=target)
    controller.set_clutch(False)
    moved = controller.update(
        gello_tcp_pose=release_gello + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ur_tcp_pose=target,
    )

    np.testing.assert_allclose(moved[:3], target[:3] + [0.01, 0.0, 0.0], atol=1e-10)


def test_gui_clutch_toggle_does_not_restart_surface_teleop_loop():
    app_source = Path("visual_guided_collection_gui/app.py").read_text(encoding="utf-8")
    method_source = app_source.split("    def _on_surface_recenter", 1)[1].split("    def _on_start_recording", 1)[0]

    assert "self.teleop_loop.stop()" not in method_source
    assert "self.teleop_loop.start_surface_positioning(" not in method_source


def test_surface_gains_scale_absolute_translation_from_neutral():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    controller = SurfaceCartesianTeleopController(
        path=path,
        probe_length_m=0.2,
        translation_gains_xyz=np.array([0.25, 0.5, 0.75]),
    )
    target = np.array([0.0, 0.0, 0.25, np.pi, 0.0, 0.0])

    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=target)
    moved = controller.update(
        gello_tcp_pose=np.array([0.04, 0.0, 0.02, 0.0, 0.0, 0.0]),
        ur_tcp_pose=target,
    )

    np.testing.assert_allclose(moved[:3], [0.01, 0.0, 0.265], atol=1e-10)
