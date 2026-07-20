import datetime
import pickle
import time

import numpy as np

from breast_path_planning.path_io import PlannedPath
from breast_path_planning.geometry import rodrigues
from visual_guided_collection_gui.collection_session import LoopConfig, TeleopLoop
from visual_guided_collection_gui.device_manager import DeviceConfig, DeviceManager
from visual_guided_collection_gui.episode_recorder import EpisodeRecorder, add_probe_tip_observation
from visual_guided_collection_gui.images import depth_to_display_rgb
from visual_guided_collection_gui.main import build_parser, resolve_args
from visual_guided_collection_gui.picking import pick_nearest_projected_point, project_points_to_screen
from visual_guided_collection_gui.app import (
    SURFACE_AUTOSCAN_POSITION_STEP_M,
    SURFACE_AUTOSCAN_ROTATION_STEP_RAD,
    SURFACE_CONFIRM_POSITION_STEP_M,
    SURFACE_CONFIRM_ROTATION_STEP_RAD,
    format_surface_bo_status_lines,
    force_display_image,
    all_path_point_geometry_names,
    path_display_color,
    path_point_geometry_name,
    path_preview_point_colors,
    path_point_colors,
)
from visual_guided_collection_gui.state import GuiStage, enabled_actions_for_stage


def test_project_points_to_screen_uses_ndc_coordinates():
    points = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.0]])
    screen, valid = project_points_to_screen(
        points,
        view_matrix=np.eye(4),
        projection_matrix=np.eye(4),
        width=100,
        height=100,
    )

    np.testing.assert_allclose(screen[0], [50.0, 50.0])
    np.testing.assert_allclose(screen[1], [75.0, 75.0])
    assert valid.tolist() == [True, True]


def test_pick_nearest_projected_point_returns_none_when_click_is_too_far():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    assert (
        pick_nearest_projected_point(
            points,
            click_xy=(2.0, 2.0),
            view_matrix=np.eye(4),
            projection_matrix=np.eye(4),
            width=100,
            height=100,
            max_pixel_distance=5.0,
        )
        is None
    )

    assert (
        pick_nearest_projected_point(
            points,
            click_xy=(51.0, 49.0),
            view_matrix=np.eye(4),
            projection_matrix=np.eye(4),
            width=100,
            height=100,
            max_pixel_distance=5.0,
        )
        == 0
    )


def test_add_probe_tip_observation_uses_tcp_z_axis_offset():
    obs = {
        "tcp_position_base": np.array([1.0, 2.0, 3.0]),
        "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
        "tcp_y_axis_base": np.array([0.0, 1.0, 0.0]),
        "tcp_z_axis_base": np.array([0.0, 0.0, -1.0]),
    }

    out = add_probe_tip_observation(obs, probe_tip_offset_m=0.2)

    np.testing.assert_allclose(out["probe_tip_position_base"], [1.0, 2.0, 2.8])
    np.testing.assert_allclose(out["probe_z_axis_base"], [0.0, 0.0, -1.0])
    assert "probe_tip_position_base" not in obs


def test_depth_to_display_rgb_returns_uint8_rgb_image():
    depth = np.array([[[0], [1000]], [[2000], [3000]]], dtype=np.uint16)

    image = depth_to_display_rgb(depth)

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert np.any(image[0, 0] != image[-1, -1])


def test_force_display_image_returns_large_rgb_panel():
    image = force_display_image(np.array([1.2, -2.3, 3.4, 0.1, -0.2, 0.3]))

    assert image.shape == (220, 420, 3)
    assert image.dtype == np.uint8
    assert image[..., 0].max() > 200
    assert image[..., 1].max() > 180


def test_force_display_image_shows_compensated_raw_gravity_and_valid_status(monkeypatch):
    drawn_text = []

    def fake_put_text(image, text, *args, **kwargs):
        drawn_text.append(str(text))
        return image

    monkeypatch.setattr("visual_guided_collection_gui.app.cv2.putText", fake_put_text)

    image = force_display_image(
        {
            "force": np.array([1.2, -2.3, -3.4, 0.1, -0.2, 0.3]),
            "force_raw": np.array([4.2, -5.3, 6.4, 0.4, -0.5, 0.6]),
            "force_gravity": np.array([0.2, 0.3, 6.8, 0.01, 0.02, 0.03]),
            "force_gravity_calibrated": True,
            "force_zeroed": True,
            "force_sensor_valid": True,
        }
    )

    assert image.shape == (260, 520, 3)
    text = "\n".join(drawn_text)
    assert "comp F" in text
    assert "raw  F" in text
    assert "grav F" in text
    assert "valid=True" in text
    assert "zero=True" in text
    assert "grav=True" in text


def test_surface_automatic_motion_step_constants_are_halved():
    assert SURFACE_CONFIRM_POSITION_STEP_M == 0.001
    assert SURFACE_CONFIRM_ROTATION_STEP_RAD == 0.006
    assert SURFACE_AUTOSCAN_POSITION_STEP_M == 0.0005
    assert SURFACE_AUTOSCAN_ROTATION_STEP_RAD == 0.003


def test_surface_bo_status_lines_show_candidate_and_measurement_values():
    lines = format_surface_bo_status_lines(
        {
            "auto_phase": "bo",
            "bo_is_measurement": True,
            "bo_trial_index": 2,
            "bo_phase": "EI",
            "bo_x": [0.004, 0.01, -0.02, 0.03],
            "bo_target_tcp_pose": [0.1, -0.2, 0.3, 0.01, 0.02, 0.03],
            "Q": 0.72,
            "F": -0.62,
            "D": 0.5,
            "E": 0.6,
            "C": 0.7,
            "S": 0.8,
            "P_f": 0.01,
            "P_tau": 0.02,
            "force_valid": True,
        }
    )

    text = "\n".join(lines)
    assert "BO trial 3" in text
    assert "EI" in text
    assert "dn=4.0mm" in text
    assert "target p: 0.100, -0.200, 0.300" in text
    assert "Q=0.7200" in text
    assert "F=-0.6200" in text
    assert "D/E/C/S=0.500/0.600/0.700/0.800" in text
    assert "Pf=0.0100" in text
    assert "force valid=True" in text


def test_comparison_recording_checks_endpoint_and_timeout_from_loop_samples():
    from pathlib import Path

    app_source = Path("visual_guided_collection_gui/app.py").read_text(
        encoding="utf-8"
    )
    method_source = app_source.split("    def _on_loop_sample", 1)[1].split(
        "    def _on_scene_mouse",
        1,
    )[0]

    assert "_check_comparison_scan_completion" in method_source
    assert '_request_comparison_finish("reached")' in app_source
    assert '_request_comparison_finish("timeout")' in app_source


def test_comparison_darboux_start_rebases_current_pose_before_releasing_clutch():
    from pathlib import Path

    app_source = Path("visual_guided_collection_gui/app.py").read_text(
        encoding="utf-8"
    )
    method_source = app_source.split(
        "    def _on_comparison_start_scan",
        1,
    )[1].split(
        "    def _on_comparison_finish_trial",
        1,
    )[0]
    assert "_current_surface_calibration_inputs()" in method_source
    assert ".recenter(" in method_source
    assert method_source.index(".recenter(") < method_source.index(
        "set_clutch(False)"
    )


def test_path_point_colors_highlight_start_current_and_future_points():
    colors = path_point_colors(12, nearest_index=4, future_count=8)

    np.testing.assert_allclose(colors[0], [1.0, 0.05, 0.05])
    np.testing.assert_allclose(colors[4], [1.0, 0.55, 0.0])
    np.testing.assert_allclose(colors[5:12], np.tile([[0.0, 0.85, 1.0]], (7, 1)))
    np.testing.assert_allclose(colors[3], [0.0, 0.8, 0.1])


def test_path_display_color_is_red_before_geodesic_and_green_after():
    plain = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0]]),
    )
    optimized = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0]]),
        metadata={"geodesic_trigger": "gui_optimize_geodesic"},
    )
    moving_average = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0]]),
        metadata={"path_variant_method": "moving_average"},
    )
    b_spline = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0]]),
        metadata={"path_variant_method": "b_spline"},
    )

    np.testing.assert_allclose(path_display_color(plain), [1.0, 0.05, 0.05])
    np.testing.assert_allclose(path_display_color(moving_average), [0.55, 0.55, 0.55])
    np.testing.assert_allclose(path_display_color(b_spline), [0.1, 0.35, 0.9])
    np.testing.assert_allclose(path_display_color(optimized), [0.0, 0.8, 0.1])
    np.testing.assert_allclose(path_preview_point_colors(plain), np.array([[1.0, 0.05, 0.05]]))
    np.testing.assert_allclose(path_preview_point_colors(optimized), np.array([[0.0, 0.8, 0.1]]))
    assert path_point_geometry_name(plain) == "planned_path_points"
    assert path_point_geometry_name(moving_average) == "moving_average_path_points"
    assert path_point_geometry_name(b_spline) == "b_spline_path_points"
    assert path_point_geometry_name(optimized) == "optimized_path_points"


def test_all_path_point_geometry_names_cover_every_path_variant():
    assert set(all_path_point_geometry_names()) == {
        "planned_path_points",
        "optimized_path_points",
        "moving_average_path_points",
        "b_spline_path_points",
    }


def test_path_preview_point_colors_use_same_uniform_color_as_line():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
        metadata={"geodesic_trigger": "gui_optimize_geodesic"},
    )

    np.testing.assert_allclose(path_preview_point_colors(path), np.tile([[0.0, 0.8, 0.1]], (3, 1)))


def test_gui_defaults_match_legacy_control_loop_speed():
    args = build_parser().parse_args([])

    assert args.hz == 50.0
    assert args.force_ip == "192.168.1.100"
    assert args.max_joint_step_rad == 0.0
    assert args.gui_update_hz > 0.0
    assert args.wrist_camera == "Orbbec"
    assert args.control_tcp is False
    assert args.surface_approach_height_m == 0.15
    assert args.surface_contact_height_m == 0.02
    assert args.probe_tip_offset_m == 0.2


def test_device_config_defaults_to_force_sensor_signal_processor_ip():
    assert DeviceConfig().force_ip == "192.168.1.100"


def test_gui_accepts_orbbec_wrist_camera_choice():
    args = resolve_args(build_parser().parse_args(["--wrist-camera", "Orbbec"]))

    assert args.wrist_camera == "Orbbec"
    assert args.t_tcp_camera == "hand_eye_calibration/Results_Orbbec/T_tcp_camera.npy"


def test_gui_keeps_explicit_t_tcp_camera_path():
    args = resolve_args(
        build_parser().parse_args(
            [
                "--wrist-camera",
                "Orbbec",
                "--t-tcp-camera",
                "hand_eye_calibration/custom.npy",
            ]
        )
    )

    assert args.t_tcp_camera == "hand_eye_calibration/custom.npy"


def test_device_manager_joint_step_limit_is_optional():
    obs = {"joint_positions": np.zeros(3)}
    action = np.array([0.2, -0.4, 0.1])

    unclamped = DeviceManager(DeviceConfig(max_joint_step_rad=0.0)).clamp_action(obs, action)
    np.testing.assert_allclose(unclamped, action)

    clamped = DeviceManager(DeviceConfig(max_joint_step_rad=0.05)).clamp_action(obs, action)
    assert np.max(np.abs(clamped - obs["joint_positions"])) <= 0.0500001


def test_episode_recorder_saves_path_features_and_fine_scan_flag(tmp_path):
    path = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ]
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1)),
    )
    recorder = EpisodeRecorder(
        data_dir=tmp_path,
        agent_name="gello",
        planned_path=path,
        probe_tip_offset_m=0.0,
    )
    recorder.start(episode_id="episode_a")
    recorder.set_fine_scan_flag(True)

    obs = {
        "joint_positions": np.zeros(6),
        "joint_velocities": np.zeros(6),
        "ee_pos_quat": np.zeros(7),
        "tcp_position_base": np.array([0.11, 0.0, 0.0]),
        "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
        "tcp_y_axis_base": np.array([0.0, 1.0, 0.0]),
        "tcp_z_axis_base": np.array([0.0, 0.0, 1.0]),
    }
    timestamp = datetime.datetime(2026, 5, 14, 22, 0, 0)

    enriched = recorder.save_sample(obs, np.ones(6), meta={"sample_index": 0}, timestamp=timestamp)

    saved_file = tmp_path / "gello" / "episode_a" / f"{timestamp.isoformat()}.pkl"
    with open(saved_file, "rb") as f:
        frame = pickle.load(f)

    assert frame["fine_scan_flag"] == 1
    assert frame["path_nearest_index"] == 1
    np.testing.assert_allclose(frame["path_residuals_base"][0], [-0.01, 0.0, 0.0])
    np.testing.assert_allclose(frame["path_reference_tcp_positions_base"][0], [0.1, 0.0, 0.0])
    reference_pose = frame["path_reference_tcp_poses_base"][0]
    np.testing.assert_allclose(reference_pose[:3], [0.1, 0.0, 0.0], atol=1e-8)
    reference_rotation = rodrigues(reference_pose[3:])
    np.testing.assert_allclose(reference_rotation[:, 0], [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(reference_rotation[:, 1], [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(reference_rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-8)
    assert "path_tcp_frames_base" not in frame
    assert enriched["path_nearest_index"] == frame["path_nearest_index"]
    assert enriched["path_progress"] == frame["path_progress"]
    assert enriched["path_distance_to_nearest_m"] == frame["path_distance_to_nearest_m"]
    assert frame["meta"]["fine_scan_flag"] == 1
    assert recorder.sample_index == 1


def test_episode_recorder_can_skip_rgb_and_depth_without_mutating_live_observation(tmp_path):
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    recorder = EpisodeRecorder(
        data_dir=tmp_path,
        agent_name="gello",
        planned_path=path,
        record_rgb_depth=False,
    )
    recorder.start(episode_id="compact")
    obs = {
        "Orbbec_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "Orbbec_depth": np.zeros((2, 2, 1), dtype=np.uint16),
        "D405_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "D405_depth": np.zeros((2, 2, 1), dtype=np.uint16),
        "Ultrasound_gray": np.ones((2, 2, 1), dtype=np.uint8),
        "force": np.arange(6, dtype=float),
        "tcp_position_base": np.zeros(3),
        "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
        "tcp_y_axis_base": np.array([0.0, 1.0, 0.0]),
        "tcp_z_axis_base": np.array([0.0, 0.0, 1.0]),
    }
    timestamp = datetime.datetime(2026, 7, 1, 12, 0, 0)

    enriched = recorder.save_sample(obs, np.zeros(6), timestamp=timestamp)

    with open(tmp_path / "gello" / "compact" / f"{timestamp.isoformat()}.pkl", "rb") as handle:
        frame = pickle.load(handle)
    assert "Orbbec_rgb" not in frame
    assert "Orbbec_depth" not in frame
    assert "D405_rgb" not in frame
    assert "D405_depth" not in frame
    assert "Ultrasound_gray" in frame
    assert "force" in frame
    assert "Orbbec_rgb" in obs
    assert "Orbbec_depth" in obs
    assert "Orbbec_rgb" in enriched
    assert "Orbbec_depth" in enriched


class FakePositioningDevices:
    def __init__(self):
        self.count = 0

    def get_obs(self):
        return {"joint_positions": np.zeros(6)}

    def step_agent(self, obs):
        self.count += 1
        next_obs = {"joint_positions": np.full(6, self.count, dtype=float)}
        action = np.full(6, self.count + 0.5, dtype=float)
        return next_obs, action, {}, {}


def test_teleop_positioning_reports_action_without_stopping_on_name_error():
    loop = TeleopLoop(
        devices=FakePositioningDevices(),
        config=LoopConfig(idle_sleep_s=0.001),
    )
    samples = []
    statuses = []

    loop.start_positioning(on_sample=samples.append, on_status=statuses.append)
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline and not samples:
        time.sleep(0.001)
    loop.stop()

    assert samples
    assert statuses == []
    np.testing.assert_allclose(samples[0]["action"], np.full(6, 1.5))


def test_enabled_actions_follow_linear_gui_state():
    assert "capture_frame" not in enabled_actions_for_stage(GuiStage.DISCONNECTED)
    assert "connect" in enabled_actions_for_stage(GuiStage.DISCONNECTED)
    assert "segment" not in enabled_actions_for_stage(GuiStage.FRAME_CAPTURED)
    assert "start_photo_positioning" in enabled_actions_for_stage(GuiStage.CONNECTED)
    assert "start_gello_handover" not in enabled_actions_for_stage(GuiStage.CONNECTED)
    assert "capture_frame" in enabled_actions_for_stage(GuiStage.POSITIONING)
    assert "start_recording" not in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "start_photo_positioning" not in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "start_gello_handover" in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "start_recording" in enabled_actions_for_stage(GuiStage.TELEOP_READY)
    assert "capture_frame" not in enabled_actions_for_stage(GuiStage.TELEOP_READY)
    assert "start_recording" not in enabled_actions_for_stage(GuiStage.PATH_PLANNED)
