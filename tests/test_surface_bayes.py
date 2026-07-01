import threading

import numpy as np

from breast_path_planning.geometry import rodrigues
from visual_guided_collection_gui.surface_bayes import (
    SurfaceBOConfig,
    SurfaceBOStopSignal,
    compute_candidate_tcp_pose,
    compute_force_torque_penalty,
    compute_online_objective,
    parse_local_bounds,
    run_surface_bayes_optimization,
    select_current_tcp_bo_reference,
)


def test_candidate_tcp_pose_offsets_along_surface_normal_and_right_multiplies_local_rotation():
    reference_pose = np.array([0.2, -0.3, 0.4, 0.0, 0.0, np.pi / 2.0])
    normal = np.array([0.0, 0.0, 1.0])
    local_x = np.array([0.01, 0.1, 0.0, 0.0])

    candidate = compute_candidate_tcp_pose(reference_pose, normal, local_x)

    assert np.allclose(candidate[:3], [0.2, -0.3, 0.41])
    expected_rotation = rodrigues(reference_pose[3:]) @ rodrigues(local_x[1:])
    assert np.allclose(rodrigues(candidate[3:]), expected_rotation, atol=1e-6)


def test_force_torque_penalty_is_zero_when_force_sensor_is_missing():
    penalty = compute_force_torque_penalty(None, SurfaceBOConfig(force_enabled=False))

    assert penalty.force_penalty == 0.0
    assert penalty.torque_penalty == 0.0
    assert penalty.force_valid is False


def test_online_objective_uses_live_ultrasound_frame_and_zero_force_when_missing():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    obs = {"Ultrasound_gray": frame}

    result = compute_online_objective(obs, SurfaceBOConfig(force_enabled=False))

    assert result.Q > 0.0
    assert result.force_penalty == 0.0
    assert result.torque_penalty == 0.0
    assert np.isclose(result.F, -result.Q)


def test_manual_stop_signal_can_cancel_bo_before_next_trial():
    signal = SurfaceBOStopSignal()

    assert signal.should_stop() is False
    signal.request_stop()

    assert signal.should_stop() is True


def test_manual_stop_signal_can_wrap_threading_event():
    event = threading.Event()
    signal = SurfaceBOStopSignal(event)

    event.set()

    assert signal.should_stop() is True


def test_parse_local_bounds_accepts_named_bo_dimensions_in_expected_order():
    bounds = parse_local_bounds("dn=-0.05,0.05;rx=-0.1,0.1;ry=-0.2,0.2;rz=-0.3,0.3")

    assert bounds == ((-0.05, 0.05), (-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3))


def test_bo_reference_uses_current_tcp_pose_and_nearest_path_normal():
    obs = {"ee_pos_rotvec": np.array([0.1, -0.2, 0.3, 0.01, 0.02, -0.03])}
    enriched = {
        "path_reference_tcp_poses_base": np.array([[9.0, 8.0, 7.0, 0.0, np.pi, 0.0]]),
        "path_normals_base": np.array([[0.0, 0.0, 2.0]]),
        "path_nearest_index": 4,
    }

    reference_pose, normal, nearest = select_current_tcp_bo_reference(obs, enriched)

    np.testing.assert_allclose(reference_pose, obs["ee_pos_rotvec"])
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])
    assert nearest == 4


class _FakeDevices:
    def __init__(self, obs, stop_signal=None):
        self.obs = dict(obs)
        self.stop_signal = stop_signal
        self.targets = []
        self.step_targets = []

    def move_tcp_pose_linear(self, target, **_kwargs):
        self.targets.append(np.asarray(target, dtype=float).copy())
        self.obs["ee_pos_rotvec"] = np.asarray(target, dtype=float).copy()
        if self.stop_signal is not None:
            self.stop_signal.request_stop()
        return dict(self.obs)

    def step_tcp_pose(self, target):
        action = np.asarray(target, dtype=float).copy()
        self.step_targets.append(action)
        self.obs["ee_pos_rotvec"] = action.copy()
        if self.stop_signal is not None:
            self.stop_signal.request_stop()
        return dict(self.obs), action, {}, {"action_mode": "tcp_pose"}

    def get_obs(self):
        return dict(self.obs)


class _FakeRecorder:
    def __init__(self):
        self.samples = []

    def save_sample(self, obs, action, *, meta=None, timestamp=None):
        self.samples.append({"obs": dict(obs), "action": np.asarray(action), "meta": dict(meta or {})})
        return dict(obs)


def test_run_surface_bayes_records_online_trial_with_live_observation():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    obs = {"Ultrasound_gray": frame, "ee_pos_rotvec": np.zeros(6)}
    devices = _FakeDevices(obs)
    recorder = _FakeRecorder()

    result = run_surface_bayes_optimization(
        devices=devices,
        reference_tcp_pose=np.zeros(6),
        normal_base=np.array([0.0, 0.0, 1.0]),
        recorder=recorder,
        config=SurfaceBOConfig(n_initial=1, n_ei=0, force_enabled=False, settle_s=0.0),
    )

    assert result.trial_count == 1
    assert result.cancelled is False
    assert len(devices.step_targets) >= 1
    assert len(recorder.samples) >= 2
    assert recorder.samples[0]["meta"]["auto_phase"] == "bo_move"
    assert recorder.samples[0]["meta"]["bo_is_measurement"] is False
    assert recorder.samples[0]["meta"]["bo_waypoint_index"] == 1
    assert recorder.samples[-1]["meta"]["auto_phase"] == "bo"
    assert recorder.samples[-1]["meta"]["bo_is_measurement"] is True
    assert recorder.samples[-1]["meta"]["bo_trial_index"] == 0
    assert recorder.samples[-1]["meta"]["force_valid"] is False


def test_run_surface_bayes_manual_stop_returns_after_current_trial():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    signal = SurfaceBOStopSignal()
    devices = _FakeDevices({"Ultrasound_gray": frame, "ee_pos_rotvec": np.zeros(6)}, stop_signal=signal)

    result = run_surface_bayes_optimization(
        devices=devices,
        reference_tcp_pose=np.zeros(6),
        normal_base=np.array([0.0, 0.0, 1.0]),
        recorder=None,
        config=SurfaceBOConfig(n_initial=3, n_ei=12, force_enabled=False, settle_s=0.0),
        stop_signal=signal,
    )

    assert result.cancelled is True
    assert result.trial_count == 1
    assert len(devices.step_targets) == 1
