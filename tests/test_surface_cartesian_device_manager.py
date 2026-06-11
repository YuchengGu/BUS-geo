import numpy as np
import threading
import time

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.device_manager import DeviceConfig, DeviceManager
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


class FakeSurfaceAgent:
    def __init__(self):
        self.actions = [
            np.zeros(6, dtype=float),
            np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        ]

    def act(self, _obs):
        return self.actions.pop(0)


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
