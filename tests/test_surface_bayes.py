import threading

import numpy as np

from breast_path_planning.geometry import rodrigues
from visual_guided_collection_gui.surface_bayes import (
    SurfaceBOConfig,
    SurfaceBOStopSignal,
    compute_candidate_tcp_pose,
    compute_force_torque_penalty,
    compute_online_objective,
    parse_ultrasound_crop,
    post_run_reset_tcp_targets,
    parse_local_bounds,
    run_surface_bayes_optimization,
    select_current_tcp_bo_reference,
    ultrasound_frame_from_obs,
)


def test_candidate_tcp_pose_offsets_along_surface_normal_and_right_multiplies_local_rotation():
    reference_pose = np.array([0.2, -0.3, 0.4, 0.0, 0.0, np.pi / 2.0])
    normal = np.array([0.0, 0.0, 1.0])
    local_x = np.array([0.01, 0.1, 0.0, 0.0])

    candidate = compute_candidate_tcp_pose(reference_pose, normal, local_x)

    assert np.allclose(candidate[:3], [0.2, -0.3, 0.41])
    expected_rotation = rodrigues(reference_pose[3:]) @ rodrigues(local_x[1:])
    assert np.allclose(rodrigues(candidate[3:]), expected_rotation, atol=1e-6)


def test_post_run_reset_targets_retreat_along_normal_and_keep_exact_x0_orientation():
    reference_pose = np.array([0.2, -0.3, 0.4, 0.1, 2.9, -0.2])
    normal = np.array([0.0, 0.0, 2.0])

    retreat, returned = post_run_reset_tcp_targets(
        reference_pose,
        normal,
        retreat_distance_m=0.15,
    )

    np.testing.assert_allclose(retreat[:3], [0.2, -0.3, 0.55])
    np.testing.assert_allclose(retreat[3:], reference_pose[3:])
    np.testing.assert_allclose(returned, reference_pose)


def test_force_torque_penalty_is_zero_when_force_sensor_is_missing():
    penalty = compute_force_torque_penalty(None, SurfaceBOConfig(force_enabled=False))

    assert penalty.force_penalty == 0.0
    assert penalty.torque_penalty == 0.0
    assert penalty.force_valid is False


def test_surface_bo_default_force_and_torque_limits_match_online_experiment():
    config = SurfaceBOConfig()

    assert config.settle_s == 0.2
    assert config.pressure_min == 2.0
    assert config.pressure_max == 8.0
    assert config.shear_max == 6.0
    assert config.torque_tangential_max == 0.8
    assert config.torque_axial_max == 0.5


def test_force_torque_penalty_uses_soft_pressure_shear_and_torque_constraints():
    config = SurfaceBOConfig(
        lambda_pressure=0.08,
        lambda_shear=0.03,
        lambda_torque=0.02,
        lambda_axial_torque=0.005,
        pressure_min=2.0,
        pressure_max=8.0,
        shear_max=6.0,
        torque_tangential_max=0.8,
        torque_axial_max=0.5,
    )
    force = np.array([3.0, 4.0, -10.0, 0.3, 0.4, 0.25], dtype=float)

    penalty = compute_force_torque_penalty(force, config)

    expected_force = 0.08 * ((2.0 / 8.0) ** 2) + 0.03 * ((5.0 / 6.0) ** 2)
    expected_torque = 0.02 * ((0.5 / 0.8) ** 2) + 0.005 * ((0.25 / 0.5) ** 2)
    assert penalty.force_valid is True
    assert np.isclose(penalty.force_penalty, expected_force)
    assert np.isclose(penalty.torque_penalty, expected_torque)


def test_run_surface_bayes_waits_before_candidate_and_verified_best_measurements(monkeypatch):
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    devices = _FakeDevices({"Ultrasound_gray": frame, "ee_pos_rotvec": np.zeros(6)})
    sleep_calls = []

    monkeypatch.setattr(
        "visual_guided_collection_gui.surface_bayes.time.sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    run_surface_bayes_optimization(
        devices=devices,
        reference_tcp_pose=np.zeros(6),
        normal_base=np.array([0.0, 0.0, 1.0]),
        recorder=None,
        config=SurfaceBOConfig(n_initial=1, n_ei=0, force_enabled=False, settle_s=0.2),
    )

    assert sleep_calls == [0.2, 0.2]


def test_force_torque_penalty_penalizes_low_pressure_without_ratio_blowup():
    config = SurfaceBOConfig(
        lambda_pressure=0.08,
        lambda_shear=0.03,
        lambda_torque=0.02,
        lambda_axial_torque=0.005,
        pressure_min=2.0,
        pressure_max=8.0,
        shear_max=6.0,
        torque_tangential_max=0.8,
        torque_axial_max=0.5,
    )
    force = np.array([0.6, 0.8, -0.5, 0.01, 0.02, 0.0], dtype=float)

    penalty = compute_force_torque_penalty(force, config)

    expected_force = 0.08 * ((1.5 / 2.0) ** 2) + 0.03 * ((1.0 / 6.0) ** 2)
    expected_torque = 0.02 * ((np.hypot(0.01, 0.02) / 0.8) ** 2)
    assert np.isclose(penalty.force_penalty, expected_force)
    assert np.isclose(penalty.torque_penalty, expected_torque)


def test_online_objective_uses_live_ultrasound_frame_and_zero_force_when_missing():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    obs = {"Ultrasound_gray": frame}

    result = compute_online_objective(obs, SurfaceBOConfig(force_enabled=False))

    assert result.Q > 0.0
    assert result.force_penalty == 0.0
    assert result.torque_penalty == 0.0
    assert np.isclose(result.F, -result.Q)


def test_online_objective_can_crop_ultrasound_before_scoring():
    frame = np.zeros((100, 120), dtype=np.uint8)
    frame[20:80, 30:90] = 160
    obs = {"Ultrasound_gray": frame}
    crop = parse_ultrasound_crop("20,80,30,90")

    cropped = ultrasound_frame_from_obs(obs, crop=crop)
    full = compute_online_objective(obs, SurfaceBOConfig(force_enabled=False, ultrasound_crop=None))
    cropped_result = compute_online_objective(
        obs,
        SurfaceBOConfig(force_enabled=False, ultrasound_crop=crop),
    )

    assert cropped.shape == (60, 60)
    assert cropped_result.Q != full.Q


def test_online_objective_variant_changes_only_objective_value_not_recorded_terms():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    obs = {
        "Ultrasound_gray": frame,
        "force": np.array([3.0, 4.0, -10.0, 0.3, 0.4, 0.25], dtype=float),
    }
    full = compute_online_objective(obs, SurfaceBOConfig(objective_variant="full"))
    no_penalty = compute_online_objective(obs, SurfaceBOConfig(objective_variant="no_penalty"))
    force_only = compute_online_objective(obs, SurfaceBOConfig(objective_variant="force_only"))
    torque_only = compute_online_objective(obs, SurfaceBOConfig(objective_variant="torque_only"))

    assert np.isclose(no_penalty.F, -no_penalty.Q)
    assert np.isclose(force_only.F, -force_only.Q + force_only.force_penalty)
    assert np.isclose(torque_only.F, -torque_only.Q + torque_only.torque_penalty)
    assert np.isclose(full.F, -full.Q + full.force_penalty + full.torque_penalty)
    assert full.force_penalty == no_penalty.force_penalty
    assert full.torque_penalty == no_penalty.torque_penalty


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
        callback = _kwargs.get("waypoint_callback")
        if callback is not None:
            callback(
                {
                    "kind": "waypoint",
                    "index": 1,
                    "count": 1,
                    "target_tcp_pose": np.asarray(target, dtype=float).tolist(),
                }
            )
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
    measurements = [
        sample for sample in recorder.samples
        if sample["meta"].get("bo_is_measurement")
    ]
    assert [sample["meta"]["bo_measurement_role"] for sample in measurements] == [
        "before",
        "candidate",
        "verified_best",
    ]
    assert measurements[0]["meta"]["bo_counts_toward_budget"] is False
    assert measurements[1]["meta"]["bo_counts_toward_budget"] is True
    assert measurements[2]["meta"]["bo_counts_toward_budget"] is False
    assert measurements[1]["meta"]["bo_trial_index"] == 0
    assert measurements[2]["meta"]["bo_x"] == result.best_x.tolist()
    assert measurements[2]["meta"]["force_valid"] is False
    assert measurements[1]["meta"]["schema_version"] == "time_alignment_v1"
    assert "timing" in measurements[1]["meta"]
    assert "modalities" in measurements[1]["meta"]
    assert result.before_objective is not None
    assert result.verified_best_objective is not None


def test_run_surface_bayes_random_strategy_records_requested_method_and_uses_budget():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    obs = {"Ultrasound_gray": frame, "ee_pos_rotvec": np.zeros(6)}
    recorder = _FakeRecorder()

    result = run_surface_bayes_optimization(
        devices=_FakeDevices(obs),
        reference_tcp_pose=np.zeros(6),
        normal_base=np.array([0.0, 0.0, 1.0]),
        recorder=recorder,
        config=SurfaceBOConfig(
            n_initial=1,
            n_ei=2,
            force_enabled=False,
            settle_s=0.0,
            search_strategy="random",
            objective_variant="no_penalty",
            random_state=7,
        ),
    )

    assert result.trial_count == 3
    assert [trial.phase for trial in result.trials] == ["random", "random", "random"]
    measurements = [
        sample["meta"] for sample in recorder.samples
        if sample["meta"].get("bo_measurement_role") == "candidate"
    ]
    assert {meta["bo_search_strategy"] for meta in measurements} == {"random"}
    assert {meta["bo_objective_variant"] for meta in measurements} == {"no_penalty"}


def test_run_surface_bayes_lhs_strategy_space_fills_each_dimension_once_per_stratum():
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[10:44, 12:68] = 150
    obs = {"Ultrasound_gray": frame, "ee_pos_rotvec": np.zeros(6)}
    config = SurfaceBOConfig(
        n_initial=3,
        n_ei=15,
        force_enabled=False,
        settle_s=0.0,
        search_strategy="lhs",
        random_state=11,
    )

    result = run_surface_bayes_optimization(
        devices=_FakeDevices(obs),
        reference_tcp_pose=np.zeros(6),
        normal_base=np.array([0.0, 0.0, 1.0]),
        recorder=None,
        config=config,
    )

    xs = np.asarray([trial.x for trial in result.trials], dtype=float)
    bounds = np.asarray(config.bounds, dtype=float)
    unit = (xs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    strata = np.floor(unit * config.max_trials).astype(int)

    assert result.trial_count == config.max_trials
    assert [trial.phase for trial in result.trials] == ["lhs"] * config.max_trials
    for dim in range(xs.shape[1]):
        assert sorted(strata[:, dim].tolist()) == list(range(config.max_trials))
    assert not np.array_equal(strata[:, 0], strata[:, 1])


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
    assert len(devices.targets) == 1
