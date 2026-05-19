import datetime
import pickle
import time

import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.collection_session import LoopConfig, TeleopLoop
from visual_guided_collection_gui.device_manager import DeviceConfig, DeviceManager
from visual_guided_collection_gui.episode_recorder import EpisodeRecorder, add_probe_tip_observation
from visual_guided_collection_gui.images import depth_to_display_rgb
from visual_guided_collection_gui.main import build_parser, resolve_args
from visual_guided_collection_gui.picking import pick_nearest_projected_point, project_points_to_screen
from visual_guided_collection_gui.app import force_display_image, path_point_colors
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


def test_path_point_colors_highlight_start_current_and_future_points():
    colors = path_point_colors(12, nearest_index=4, future_count=8)

    np.testing.assert_allclose(colors[0], [1.0, 0.05, 0.05])
    np.testing.assert_allclose(colors[4], [1.0, 0.55, 0.0])
    np.testing.assert_allclose(colors[5:12], np.tile([[0.0, 0.85, 1.0]], (7, 1)))
    np.testing.assert_allclose(colors[3], [0.0, 0.8, 0.1])


def test_gui_defaults_match_legacy_control_loop_speed():
    args = build_parser().parse_args([])

    assert args.hz == 50.0
    assert args.max_joint_step_rad == 0.0
    assert args.gui_update_hz > 0.0
    assert args.wrist_camera == "Orbbec"


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
    assert enriched["path_nearest_index"] == frame["path_nearest_index"]
    assert enriched["path_progress"] == frame["path_progress"]
    assert enriched["path_distance_to_nearest_m"] == frame["path_distance_to_nearest_m"]
    assert frame["meta"]["fine_scan_flag"] == 1
    assert recorder.sample_index == 1


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
