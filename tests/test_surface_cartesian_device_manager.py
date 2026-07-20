import numpy as np
import pytest
import threading
import time

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.force_gravity import (
    ForceGravityCalibration,
    demo1_force_gravity_compensation,
    fit_force_gravity_calibration,
    load_force_gravity_calibration,
    save_force_gravity_calibration,
)
from visual_guided_collection_gui.device_manager import DeviceConfig, DeviceManager, _interpolate_joint_positions
from visual_guided_collection_gui.surface_teleop import (
    SurfaceCartesianTeleopController,
    build_tcp_frame,
    matrix_to_rotvec,
)


class FakeTcpPoseEnv:
    control_rate_hz = 50.0

    def __init__(self):
        self.pose = np.zeros(6, dtype=float)
        self.commands = []
        self.last_obs_meta = {}
        self.last_step_timing = {}

    def get_obs(self):
        return {
            "ee_pos_rotvec": self.pose.copy(),
            "joint_positions": np.zeros(6, dtype=float),
        }

    def step_tcp_pose(self, pose):
        self.pose = np.asarray(pose, dtype=float).copy()
        self.commands.append(self.pose.copy())
        return self.get_obs()


class FakeJointEnv:
    control_rate_hz = 50.0

    def __init__(self):
        self.joints = np.zeros(6, dtype=float)
        self.commands = []
        self.servo_stop_calls = 0
        self.last_obs_meta = {}
        self.last_step_timing = {}

    def get_obs(self):
        return {
            "ee_pos_rotvec": np.zeros(6, dtype=float),
            "joint_positions": self.joints.copy(),
        }

    def step(self, joints):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.commands.append(self.joints.copy())
        return self.get_obs()

    def robot(self):
        return self

    def stop_servo(self):
        self.servo_stop_calls += 1


class LaggingJointEnv(FakeJointEnv):
    def step(self, joints):
        command = np.asarray(joints, dtype=float).copy()
        self.joints += 0.25 * (command - self.joints)
        self.commands.append(command)
        return self.get_obs()


class FakeForceEnv(FakeTcpPoseEnv):
    def __init__(self, force):
        super().__init__()
        self.force = np.asarray(force, dtype=float)

    def get_obs(self):
        obs = super().get_obs()
        obs["force"] = self.force.copy()
        return obs


class FakeForceSensor:
    def __init__(self, samples):
        self.samples = [np.asarray(sample, dtype=float) for sample in samples]
        self.index = 0

    def read_values(self):
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return sample.copy()


class CountingForceEnv(FakeTcpPoseEnv):
    def __init__(self, pose):
        super().__init__()
        self.pose = np.asarray(pose, dtype=float)
        self.get_obs_count = 0

    def get_obs(self):
        self.get_obs_count += 1
        return super().get_obs()


class FakeRobotClient:
    def __init__(self, pose):
        self.pose = np.asarray(pose, dtype=float)

    def get_observations(self):
        return {"ee_pos_rotvec": self.pose.copy()}


class FakeSurfaceAgent:
    def __init__(self):
        self.actions = [
            np.zeros(6, dtype=float),
            np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        ]

    def act(self, _obs):
        return self.actions.pop(0)


class ConstantJointAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def act(self, _obs):
        return self.action.copy()


def test_joint_handover_error_compares_current_robot_and_gello_commands():
    manager = DeviceManager(DeviceConfig())
    manager.agent = ConstantJointAgent([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    obs = {"joint_positions": np.array([0.08, -0.18, 0.25, 0.0, 0.0, 0.0])}

    error, target = manager.joint_handover_error(obs)

    assert error == pytest.approx(0.05)
    np.testing.assert_allclose(target, [0.1, -0.2, 0.3, 0.0, 0.0, 0.0])


def test_joint_step_reports_joint_position_action_mode():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeJointEnv()
    manager.agent = ConstantJointAgent(np.full(6, 0.1))
    obs = manager.env.get_obs()

    _next_obs, _action, _obs_meta, timing = manager.step_agent(obs)

    assert timing["action_mode"] == "joint_position"


def test_move_tcp_pose_linear_interpolates_and_waits_until_target_reached():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeTcpPoseEnv()
    target = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.2])

    obs = manager.move_tcp_pose_linear(
        target,
        max_position_step_m=0.01,
        max_rotation_step_rad=0.1,
        position_tolerance_m=1e-9,
        rotation_tolerance_rad=1e-9,
        timeout_s=1.0,
    )

    assert len(manager.env.commands) == 3
    np.testing.assert_allclose(manager.env.commands[-1], target)
    np.testing.assert_allclose(obs["ee_pos_rotvec"], target)


def test_device_manager_zeroes_force_and_preserves_raw_values():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeForceEnv([4.0, 5.0, 6.0, 0.4, 0.5, 0.6])
    manager.force_sensor = FakeForceSensor(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [3.0, 4.0, 5.0, 0.3, 0.4, 0.5],
        ]
    )

    bias = manager.zero_force(samples=2)
    obs = manager.get_obs()

    np.testing.assert_allclose(bias, [2.0, 3.0, 4.0, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(obs["force_raw"], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(obs["force_bias"], bias)
    np.testing.assert_allclose(obs["force"], [2.0, 2.0, 2.0, 0.2, 0.2, 0.2])
    assert obs["force_zeroed"] is True


def test_device_manager_applies_force_zero_to_tcp_step_observation():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeForceEnv([5.0, 4.0, 3.0, 0.5, 0.4, 0.3])
    manager.force_bias = np.array([1.0, 1.5, 2.0, 0.1, 0.2, 0.3], dtype=float)

    obs, _action, _obs_meta, _timing = manager.step_tcp_pose(np.zeros(6))

    np.testing.assert_allclose(obs["force_raw"], [5.0, 4.0, 3.0, 0.5, 0.4, 0.3])
    np.testing.assert_allclose(obs["force"], [4.0, 2.5, 1.0, 0.4, 0.2, 0.0])


def test_device_manager_does_not_subtract_bias_from_invalid_force_frame():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeForceEnv(np.zeros(6))
    manager.env.last_obs_meta = {"modalities": {"force": {"valid": False}}}
    manager.force_bias = np.ones(6, dtype=float)

    obs = manager.get_obs()

    np.testing.assert_allclose(obs["force_raw"], np.zeros(6))
    np.testing.assert_allclose(obs["force"], np.zeros(6))
    assert obs["force_zeroed"] is False


def test_force_gravity_calibration_subtracts_pose_dependent_gravity():
    calibration = ForceGravityCalibration(
        force_bias_sensor=np.array([1.0, 2.0, 3.0]),
        gravity_base=np.array([0.0, 0.0, -9.0]),
        torque_bias_sensor=np.array([0.1, 0.2, 0.3]),
        sample_count=4,
    )

    compensated, bias, gravity = calibration.compensate(
        raw_wrench=np.array([2.0, 4.0, -5.0, 0.5, 0.7, 0.9]),
        tcp_pose=np.zeros(6, dtype=float),
    )

    np.testing.assert_allclose(bias, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(gravity, [0.0, 0.0, -9.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(compensated, [1.0, 2.0, 1.0, 0.4, 0.5, 0.6])


def test_force_gravity_calibration_round_trips_npz(tmp_path):
    calibration = ForceGravityCalibration(
        force_bias_sensor=np.array([1.0, 2.0, 3.0]),
        gravity_base=np.array([0.0, 0.0, -9.0]),
        torque_bias_sensor=np.array([0.1, 0.2, 0.3]),
        sample_count=8,
    )
    path = tmp_path / "gravity_calib.npz"

    save_force_gravity_calibration(path, calibration)
    loaded = load_force_gravity_calibration(path)

    np.testing.assert_allclose(loaded.force_bias_sensor, calibration.force_bias_sensor)
    np.testing.assert_allclose(loaded.gravity_base, calibration.gravity_base)
    np.testing.assert_allclose(loaded.torque_bias_sensor, calibration.torque_bias_sensor)
    assert loaded.sample_count == 8


def test_fit_force_gravity_calibration_recovers_bias_and_gravity():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0],
        ],
        dtype=float,
    )
    truth = ForceGravityCalibration(
        force_bias_sensor=np.array([0.2, -0.1, 0.3]),
        gravity_base=np.array([0.0, 0.0, -1.5]),
        torque_bias_sensor=np.array([0.01, 0.02, 0.03]),
        sample_count=poses.shape[0],
    )
    wrenches = []
    for pose in poses:
        gravity = np.zeros(6, dtype=float)
        gravity[:3] = truth.gravity_force_sensor(pose[3:6])
        wrenches.append(truth.bias_wrench + gravity)

    fitted = fit_force_gravity_calibration(poses, np.vstack(wrenches))

    np.testing.assert_allclose(fitted.force_bias_sensor, truth.force_bias_sensor, atol=1e-10)
    np.testing.assert_allclose(fitted.gravity_base, truth.gravity_base, atol=1e-10)
    np.testing.assert_allclose(fitted.torque_bias_sensor, truth.torque_bias_sensor, atol=1e-10)


def test_device_manager_applies_loaded_force_gravity_calibration():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeForceEnv([2.0, 4.0, -5.0, 0.5, 0.7, 0.9])
    manager.force_gravity_calibration = ForceGravityCalibration(
        force_bias_sensor=np.array([1.0, 2.0, 3.0]),
        gravity_base=np.array([0.0, 0.0, -9.0]),
        torque_bias_sensor=np.array([0.1, 0.2, 0.3]),
        sample_count=4,
    )

    obs = manager.get_obs()

    np.testing.assert_allclose(obs["force_raw"], [2.0, 4.0, -5.0, 0.5, 0.7, 0.9])
    np.testing.assert_allclose(obs["force_gravity"], [0.0, 0.0, -9.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["force"], [1.0, 2.0, 1.0, 0.4, 0.5, 0.6])
    assert obs["force_gravity_calibrated"] is True


def test_demo_force_gravity_model_makes_downward_probe_gravity_positive_fz():
    model = demo1_force_gravity_compensation()
    tcp_pose = np.array([0.0, 0.0, 0.0, np.pi, 0.0, 0.0], dtype=float)

    gravity = model.gravity_wrench_sensor(tcp_pose)

    assert gravity[2] > 0.0
    np.testing.assert_allclose(gravity[2], 0.69177 * 9.80665, rtol=1e-6)


def test_device_manager_zeroes_force_after_subtracting_demo_gravity():
    manager = DeviceManager(DeviceConfig())
    tcp_pose = np.array([0.0, 0.0, 0.0, np.pi, 0.0, 0.0], dtype=float)
    model = demo1_force_gravity_compensation()
    gravity = model.gravity_wrench_sensor(tcp_pose)
    true_bias = np.array([0.1, -0.2, 3.0, 0.01, -0.02, 0.03], dtype=float)
    raw_free_space = true_bias + gravity
    manager.robot_client = FakeRobotClient(tcp_pose)
    manager.force_sensor = FakeForceSensor([raw_free_space, raw_free_space])
    manager.force_gravity_calibration = model
    manager.env = FakeForceEnv(raw_free_space)
    manager.env.pose = tcp_pose.copy()

    bias = manager.zero_force_with_gravity(samples=2, sample_interval_s=0.0)
    obs = manager.get_obs()

    np.testing.assert_allclose(bias, true_bias)
    np.testing.assert_allclose(obs["force"], np.zeros(6), atol=1e-10)


def test_device_manager_reads_force_only_with_gravity_compensation_without_full_env_obs():
    tcp_pose = np.array([0.1, -0.2, 0.3, np.pi, 0.0, 0.0], dtype=float)
    model = demo1_force_gravity_compensation()
    gravity = model.gravity_wrench_sensor(tcp_pose)
    bias = np.array([0.2, -0.1, 0.3, 0.01, -0.02, 0.03], dtype=float)
    contact = np.array([1.0, 2.0, -3.0, 0.4, 0.5, -0.6], dtype=float)
    raw = bias + gravity + contact

    manager = DeviceManager(DeviceConfig())
    manager.env = CountingForceEnv(tcp_pose)
    manager.robot_client = FakeRobotClient(tcp_pose)
    manager.force_sensor = FakeForceSensor([raw])
    manager.force_gravity_calibration = model.with_bias(bias)
    manager.force_bias = bias

    obs = manager.get_force_obs()

    np.testing.assert_allclose(obs["ee_pos_rotvec"], tcp_pose)
    np.testing.assert_allclose(obs["force_raw"], raw)
    np.testing.assert_allclose(obs["force_gravity"], gravity)
    np.testing.assert_allclose(obs["force"], contact)
    assert obs["force_sensor_valid"] is True
    assert obs["force_zeroed"] is True
    assert manager.env.get_obs_count == 0


def test_step_surface_cartesian_teleop_sends_tcp_pose_not_joint_command():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeTcpPoseEnv()
    manager.agent = FakeSurfaceAgent()
    manager.gello_action_to_tcp_pose = lambda action: np.asarray(action, dtype=float)
    start_rotation = build_tcp_frame(np.array([1.0, 0.0, 0.0]), path.normals_base[0])
    manager.env.pose = np.concatenate([[0.0, 0.0, 0.25], matrix_to_rotvec(start_rotation)])
    controller = SurfaceCartesianTeleopController(path=path, probe_length_m=0.2)
    obs = manager.env.get_obs()
    controller.set_neutral(gello_tcp_pose=np.zeros(6), ur_tcp_pose=obs["ee_pos_rotvec"])
    controller.calibrate_x(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    controller.calibrate_z(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    controller.recenter(gello_tcp_pose=np.zeros(6), ur_tcp_pose=obs["ee_pos_rotvec"])

    next_obs, action, _obs_meta, timing = manager.step_surface_cartesian_teleop(controller, obs)
    next_obs, action, _obs_meta, timing = manager.step_surface_cartesian_teleop(controller, next_obs)

    assert timing["action_mode"] == "tcp_pose"
    assert len(manager.env.commands) == 2
    np.testing.assert_allclose(action[:3], [0.01, 0.0, 0.25], atol=1e-10)
    np.testing.assert_allclose(next_obs["ee_pos_rotvec"], action)


class NonReentrantTcpPoseEnv(FakeTcpPoseEnv):
    def __init__(self):
        super().__init__()
        self.step_started = threading.Event()
        self.finish_step = threading.Event()
        self.in_step = False

    def get_obs(self):
        if self.in_step:
            raise RuntimeError("concurrent env access")
        return super().get_obs()

    def step_tcp_pose(self, pose):
        self.in_step = True
        self.step_started.set()
        self.finish_step.wait(timeout=1.0)
        self.pose = np.asarray(pose, dtype=float).copy()
        self.commands.append(self.pose.copy())
        self.in_step = False
        return self.get_obs()


def test_device_manager_serializes_get_obs_against_tcp_step():
    manager = DeviceManager(DeviceConfig())
    manager.env = NonReentrantTcpPoseEnv()
    target = np.ones(6, dtype=float)
    errors = []
    observations = []

    def step_target():
        try:
            manager.step_tcp_pose(target)
        except Exception as exc:  # pragma: no cover - assertion reports captured exception
            errors.append(exc)

    def read_obs():
        try:
            observations.append(manager.get_obs())
        except Exception as exc:  # pragma: no cover - assertion reports captured exception
            errors.append(exc)

    step_thread = threading.Thread(target=step_target)
    step_thread.start()
    assert manager.env.step_started.wait(timeout=1.0)

    read_thread = threading.Thread(target=read_obs)
    read_thread.start()
    time.sleep(0.05)
    assert observations == []

    manager.env.finish_step.set()
    step_thread.join(timeout=1.0)
    read_thread.join(timeout=1.0)

    assert errors == []
    assert len(observations) == 1


def test_interpolate_joint_positions_limits_each_joint_step():
    start = np.zeros(6)
    target = np.array([0.0, -0.1, 0.2, -0.3, 0.4, 0.5])

    waypoints = _interpolate_joint_positions(start, target, max_joint_step_rad=0.1)

    assert len(waypoints) == 5
    np.testing.assert_allclose(waypoints[-1], target)
    previous = start
    for waypoint in waypoints:
        assert np.max(np.abs(waypoint - previous)) <= 0.1000001
        previous = waypoint


def test_move_joint_positions_linear_uses_interpolated_joint_steps():
    manager = DeviceManager(DeviceConfig())
    manager.env = FakeJointEnv()
    target = np.array([0.0, -0.1, 0.2, -0.3, 0.4, 0.5])

    obs = manager.move_joint_positions_linear(target, max_joint_step_rad=0.1, timeout_s=1.0)

    assert len(manager.env.commands) == 5
    np.testing.assert_allclose(manager.env.commands[-1], target)
    np.testing.assert_allclose(obs["joint_positions"], target)
    assert manager.env.servo_stop_calls == 1


def test_move_joint_positions_linear_keeps_commanding_target_until_actual_joints_arrive():
    manager = DeviceManager(DeviceConfig())
    manager.env = LaggingJointEnv()
    target = np.array([0.0, -0.1, 0.2, -0.3, 0.4, 0.5])

    obs = manager.move_joint_positions_linear(
        target,
        max_joint_step_rad=0.1,
        joint_tolerance_rad=0.01,
        timeout_s=1.0,
    )

    assert len(manager.env.commands) > 5
    np.testing.assert_allclose(manager.env.commands[-1], target)
    assert np.max(np.abs(obs["joint_positions"] - target)) <= 0.01
    assert manager.env.servo_stop_calls == 1
