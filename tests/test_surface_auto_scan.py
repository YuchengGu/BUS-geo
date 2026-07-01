import threading

import numpy as np

from visual_guided_collection_gui.surface_auto_scan import (
    SurfaceForceServoConfig,
    _update_force_servo_command,
    run_surface_auto_scan,
)


class FakeAutoScanDevices:
    class Env:
        control_rate_hz = 50.0
        last_step_timing = {
            "step_start_mono_ns": 10,
            "action_send_start_mono_ns": 20,
            "action_send_end_mono_ns": 30,
            "step_end_mono_ns": 40,
            "action_mode": "tcp_pose",
        }

    def __init__(self):
        self.env = self.Env()
        self.obs = {
            "joint_positions": np.zeros(6),
            "joint_velocities": np.zeros(6),
            "ee_pos_quat": np.zeros(7),
            "ee_pos_rotvec": np.zeros(6),
            "tcp_position_base": np.array([0.0, 0.0, 0.2]),
            "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
            "tcp_y_axis_base": np.array([0.0, -1.0, 0.0]),
            "tcp_z_axis_base": np.array([0.0, 0.0, -1.0]),
            "Orbbec_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "Orbbec_depth": np.zeros((2, 2, 1), dtype=np.uint16),
            "Ultrasound_gray": np.zeros((2, 2), dtype=np.uint8),
            "force": np.zeros(6),
        }
        self.actions = []

    def get_obs(self):
        return dict(self.obs)

    def step_tcp_pose(self, tcp_pose):
        action = np.asarray(tcp_pose, dtype=float).copy()
        self.actions.append(action)
        obs_meta = {
            "obs_read_start_mono_ns": 1,
            "obs_read_end_mono_ns": 2,
            "modalities": {
                "Orbbec": {"valid": True},
                "Ultrasound": {"valid": True},
                "robot": {"valid": True},
                "force": {"valid": True},
            },
        }
        self.obs["ee_pos_rotvec"] = action.copy()
        self.obs["tcp_position_base"] = action[:3].copy()
        return dict(self.obs), action, obs_meta, {"tcp_pose_send_start_mono_ns": 3, "tcp_pose_send_end_mono_ns": 4, "action_mode": "tcp_pose"}


class SequenceForceAutoScanDevices(FakeAutoScanDevices):
    def __init__(self, forces):
        super().__init__()
        self.forces = [np.asarray(force, dtype=float) for force in forces]
        self.force_index = 0
        self.obs["force"] = self.forces[0].copy()

    def step_tcp_pose(self, tcp_pose):
        out = super().step_tcp_pose(tcp_pose)
        self.force_index = min(self.force_index + 1, len(self.forces) - 1)
        self.obs["force"] = self.forces[self.force_index].copy()
        return out


class FakeRecorder:
    def __init__(self):
        self.samples = []
        self.sample_index = 0
        self.episode_dir = None

    def save_sample(self, obs, action, *, meta=None, timestamp=None):
        self.samples.append({"obs": dict(obs), "action": np.asarray(action), "meta": dict(meta or {})})
        self.sample_index += 1
        return dict(obs)


def test_surface_auto_scan_records_each_interpolated_tcp_step():
    devices = FakeAutoScanDevices()
    recorder = FakeRecorder()
    poses = [
        np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=poses,
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
    )

    assert result.completed is True
    assert result.saved_samples == len(recorder.samples)
    assert len(recorder.samples) >= 4
    assert recorder.samples[0]["meta"]["operation_mode"] == "auto"
    assert recorder.samples[0]["meta"]["auto_phase"] == "scan"
    assert recorder.samples[0]["meta"]["auto_scan_pose_index"] == 0
    assert recorder.samples[0]["meta"]["auto_scan_waypoint_index"] == 1
    assert "Orbbec_rgb" in recorder.samples[0]["obs"]
    assert "Orbbec_depth" in recorder.samples[0]["obs"]
    assert "Ultrasound_gray" in recorder.samples[0]["obs"]
    assert "force" in recorder.samples[0]["obs"]


def test_surface_auto_scan_records_same_time_alignment_meta_shape_as_gello_recording():
    devices = FakeAutoScanDevices()
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
    )

    assert result.completed is True
    sample = recorder.samples[0]
    meta = sample["meta"]
    assert meta["schema_version"] == "time_alignment_v1"
    assert meta["sample_semantics"] == "obs_t_to_action_t"
    assert meta["control_loop_hz_config"] == 50.0
    assert meta["timing"]["obs_read_start_mono_ns"] == 1
    assert meta["timing"]["tcp_pose_send_start_mono_ns"] == 3
    assert meta["timing"]["step_start_mono_ns"] == 10
    assert meta["modalities"]["Ultrasound"]["valid"] is True
    assert meta["modalities"]["force"]["valid"] is True
    assert meta["auto_phase"] == "scan"
    assert np.allclose(sample["obs"]["ee_pos_rotvec"], np.zeros(6))
    assert np.allclose(sample["action"][:3], [0.001, 0.0, 0.0])


def test_surface_auto_scan_stop_event_ends_before_next_waypoint():
    devices = FakeAutoScanDevices()
    recorder = FakeRecorder()
    stop_event = threading.Event()

    def on_sample(_sample):
        stop_event.set()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])],
        recorder=recorder,
        stop_event=stop_event,
        on_sample=on_sample,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
    )

    assert result.completed is False
    assert result.stopped is True
    assert len(recorder.samples) == 1


def test_surface_auto_scan_force_servo_presses_along_normal_when_fz_is_low():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.zeros(6)],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0, hard_lift_pressure_n=999.0),
    )

    assert result.completed is True
    np.testing.assert_allclose(devices.actions[0][:3], [0.0, 0.0, -0.00025])
    assert recorder.samples[0]["meta"]["auto_force_servo_enabled"] is True
    assert recorder.samples[0]["meta"]["auto_force_servo_direction"] == "press"
    assert recorder.samples[0]["meta"]["auto_force_servo_acceleration_m_s2"] < 0.0
    assert abs(recorder.samples[0]["meta"]["auto_force_servo_velocity_m_s"]) <= 0.0125001


def test_surface_auto_scan_force_servo_lifts_along_normal_when_fz_is_high():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.zeros(6)],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0, hard_lift_pressure_n=999.0),
    )

    assert result.completed is True
    np.testing.assert_allclose(devices.actions[0][:3], [0.0, 0.0, 0.00025])
    assert recorder.samples[0]["meta"]["auto_force_servo_direction"] == "lift"


def test_surface_auto_scan_force_servo_limits_total_offset_to_five_mm():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.zeros(6) for _ in range(50)],
        normals_base=[np.array([0.0, 0.0, 1.0]) for _ in range(50)],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0, hard_lift_pressure_n=999.0),
    )

    assert result.completed is True
    assert np.max([action[2] for action in devices.actions]) <= 0.0050001
    assert recorder.samples[-1]["meta"]["auto_force_servo_max_offset_m"] == 0.005


def test_surface_auto_scan_force_servo_uses_negative_pressure_sign():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.zeros(6)],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0),
    )

    assert result.completed is True
    np.testing.assert_allclose(devices.actions[0][:3], [0.0, 0.0, -0.00025])
    assert recorder.samples[0]["meta"]["auto_force_servo_direction"] == "press"


def test_surface_auto_scan_force_servo_direction_guard_prevents_press_from_lifting():
    command = np.zeros(6, dtype=float)

    offset, velocity, _filtered, _hard_lift_active, meta = _update_force_servo_command(
        {"force": np.array([0.0, 0.0, -2.2, 0.0, 0.0, 0.0])},
        command,
        np.array([0.0, 0.0, 1.0]),
        reference_position=np.zeros(3),
        force_offset_m=-0.0013,
        force_velocity_m_s=-0.0125,
        filtered_pressure_n=2.2,
        hard_lift_active=False,
        dt_s=0.02,
        config=SurfaceForceServoConfig(enabled=True),
    )

    assert meta["auto_force_servo_direction"] == "press"
    assert meta["auto_force_servo_delta_offset_m"] <= 0.0
    assert offset <= -0.0013
    assert velocity <= 0.0
    assert command[2] <= 0.0


def test_surface_auto_scan_force_servo_direction_guard_prevents_lift_from_pressing():
    command = np.zeros(6, dtype=float)

    offset, velocity, _filtered, _hard_lift_active, meta = _update_force_servo_command(
        {"force": np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])},
        command,
        np.array([0.0, 0.0, 1.0]),
        reference_position=np.zeros(3),
        force_offset_m=0.0013,
        force_velocity_m_s=0.0125,
        filtered_pressure_n=5.0,
        hard_lift_active=False,
        dt_s=0.02,
        config=SurfaceForceServoConfig(enabled=True),
    )

    assert meta["auto_force_servo_direction"] == "lift"
    assert meta["auto_force_servo_delta_offset_m"] >= 0.0
    assert offset >= 0.0013
    assert velocity >= 0.0
    assert command[2] >= 0.0


def test_surface_auto_scan_force_servo_blocks_inward_waypoint_when_overpressure():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()
    stop_event = threading.Event()

    def on_sample(_sample):
        stop_event.set()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0])],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        stop_event=stop_event,
        on_sample=on_sample,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0),
    )

    assert result.stopped is True
    assert devices.actions[0][2] >= 0.0
    assert recorder.samples[0]["meta"]["auto_force_servo_direction"] == "lift"
    assert recorder.samples[0]["meta"]["auto_force_servo_inward_motion_blocked"] is True


def test_surface_auto_scan_force_servo_offset_is_relative_to_current_waypoint():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()
    stop_event = threading.Event()

    def on_sample(_sample):
        stop_event.set()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        stop_event=stop_event,
        on_sample=on_sample,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0),
    )

    assert result.stopped is True
    np.testing.assert_allclose(devices.actions[0][2], 0.00125)
    np.testing.assert_allclose(recorder.samples[0]["meta"]["auto_force_servo_command_offset_m"], 0.00025)


def test_surface_auto_scan_hard_lift_pauses_path_until_pressure_recovers():
    devices = SequenceForceAutoScanDevices(
        [
            [0.0, 0.0, -10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -4.0, 0.0, 0.0, 0.0],
        ]
    )
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(
            enabled=True,
            lowpass_alpha=1.0,
            hard_lift_pressure_n=8.0,
            hard_lift_resume_pressure_n=4.5,
            hard_lift_step_m=0.00025,
        ),
    )

    assert result.completed is True
    first = recorder.samples[0]
    second = recorder.samples[1]
    later_actions = devices.actions[2:]
    assert first["meta"]["auto_force_servo_hard_lift_active"] is True
    assert second["meta"]["auto_force_servo_hard_lift_active"] is True
    np.testing.assert_allclose(devices.actions[0][:2], [0.0, 0.0])
    np.testing.assert_allclose(devices.actions[1][:2], [0.0, 0.0])
    assert devices.actions[1][2] > devices.actions[0][2]
    assert any(action[0] > 0.0 for action in later_actions)


def test_surface_auto_scan_hard_lift_uses_separate_eight_cm_escape_limit():
    devices = SequenceForceAutoScanDevices([[0.0, 0.0, -10.0, 0.0, 0.0, 0.0]])
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(
            enabled=True,
            lowpass_alpha=1.0,
            max_offset_m=0.006,
            max_step_m=0.005,
            hard_lift_pressure_n=8.0,
            hard_lift_resume_pressure_n=4.5,
            hard_lift_step_m=0.005,
            hard_lift_max_m=0.08,
        ),
    )

    assert result.completed is False
    assert result.error is not None
    assert "hard lift limit" in result.error
    assert np.max([action[2] for action in devices.actions]) > 0.006
    assert np.max([action[2] for action in devices.actions]) <= 0.0800001
    assert recorder.samples[-1]["meta"]["auto_force_servo_hard_lift_limit_reached"] is True
    assert recorder.samples[-1]["meta"]["auto_force_servo_hard_lift_max_m"] == 0.08


def test_surface_auto_scan_hard_lift_triggers_on_large_lateral_force():
    devices = SequenceForceAutoScanDevices(
        [
            [9.0, 0.0, -3.5, 0.0, 0.0, 0.0],
            [9.0, 0.0, -3.5, 0.0, 0.0, 0.0],
            [4.0, 0.0, -3.5, 0.0, 0.0, 0.0],
            [4.0, 0.0, -3.5, 0.0, 0.0, 0.0],
        ]
    )
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(
            enabled=True,
            lowpass_alpha=1.0,
            hard_lift_pressure_n=8.0,
            hard_lift_lateral_force_n=8.0,
            hard_lift_resume_pressure_n=4.5,
            hard_lift_lateral_resume_n=4.5,
            hard_lift_step_m=0.0001,
        ),
    )

    assert result.completed is True
    first = recorder.samples[0]["meta"]
    second = recorder.samples[1]["meta"]
    assert first["auto_force_servo_hard_lift_active"] is True
    assert first["auto_force_servo_hard_lift_reason"] == "lateral"
    assert first["auto_force_servo_lateral_force_n"] == 9.0
    assert second["auto_force_servo_hard_lift_active"] is True
    np.testing.assert_allclose(devices.actions[0][:2], [0.0, 0.0])
    assert 0.0 < devices.actions[0][2] <= 0.0002001


def test_surface_auto_scan_press_limit_stays_at_configured_max_offset():
    devices = FakeAutoScanDevices()
    devices.obs["force"] = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.zeros(6) for _ in range(50)],
        normals_base=[np.array([0.0, 0.0, 1.0]) for _ in range(50)],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True, lowpass_alpha=1.0, max_offset_m=0.003),
    )

    assert result.completed is True
    assert np.min([action[2] for action in devices.actions]) >= -0.0030001


def test_surface_auto_scan_force_servo_holds_when_force_is_missing():
    devices = FakeAutoScanDevices()
    devices.obs.pop("force")
    recorder = FakeRecorder()

    result = run_surface_auto_scan(
        devices=devices,
        tcp_poses=[np.zeros(6)],
        normals_base=[np.array([0.0, 0.0, 1.0])],
        recorder=recorder,
        max_position_step_m=0.001,
        max_rotation_step_rad=0.01,
        force_servo=SurfaceForceServoConfig(enabled=True),
    )

    assert result.completed is True
    np.testing.assert_allclose(devices.actions[0][:3], [0.0, 0.0, 0.0])
    assert recorder.samples[0]["meta"]["auto_force_servo_valid"] is False
