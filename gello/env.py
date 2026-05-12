import time
from typing import Any, Dict, Optional

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.monotonic()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.monotonic():
            time.sleep(0.0001)
        self.last = time.monotonic()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        force_sensor = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._control_rate_hz = control_rate_hz
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._force_sensor = force_sensor
        self.last_obs_meta: Dict[str, Any] = {}
        self.last_step_timing: Dict[str, int] = {}

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    @property
    def control_rate_hz(self) -> float:
        return self._control_rate_hz

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        step_start = time.monotonic_ns()
        action_send_start = time.monotonic_ns()
        self._robot.command_joint_state(joints)
        action_send_end = time.monotonic_ns()
        self._rate.sleep()
        obs = self.get_obs()
        step_end = time.monotonic_ns()
        self.last_step_timing = {
            "step_start_mono_ns": step_start,
            "action_send_start_mono_ns": action_send_start,
            "action_send_end_mono_ns": action_send_end,
            "step_end_mono_ns": step_end,
        }
        return obs

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        obs_read_start = time.monotonic_ns()
        observations = {}
        modalities: Dict[str, Dict[str, Any]] = {}
        for name, camera in self._camera_dict.items():
            read_start = time.monotonic_ns()
            image, depth = camera.read()
            read_end = time.monotonic_ns()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth
            camera_meta = dict(getattr(camera, "last_metadata", {}) or {})
            camera_meta.setdefault("read_start_mono_ns", read_start)
            camera_meta.setdefault("read_end_mono_ns", read_end)
            camera_meta.setdefault("valid", True)
            modalities[name] = camera_meta

        robot_read_start = time.monotonic_ns()
        robot_obs = self._robot.get_observations()
        robot_read_end = time.monotonic_ns()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        modalities["robot"] = {
            "read_start_mono_ns": robot_read_start,
            "read_end_mono_ns": robot_read_end,
            "valid": True,
        }
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        if "ee_pos_rotvec" in robot_obs:
            observations["ee_pos_rotvec"] = robot_obs["ee_pos_rotvec"]
        if "gripper_position" in robot_obs:
            observations["gripper_position"] = robot_obs["gripper_position"]

        if self._force_sensor is not None:
            force_read_start = time.monotonic_ns()
            force_data = self._force_sensor.read_values()
            force_read_end = time.monotonic_ns()
            force_meta = dict(getattr(self._force_sensor, "last_metadata", {}) or {})
            force_meta.setdefault("read_start_mono_ns", force_read_start)
            force_meta.setdefault("read_end_mono_ns", force_read_end)
            if force_data is not None:
                observations["force"] = np.array(force_data)
                force_meta.setdefault("valid", True)
                force_meta.setdefault("error", None)
            else:
                observations["force"] = np.zeros(6) # 如果这一帧没读到，用0填充防崩溃
                force_meta.setdefault("valid", False)
                force_meta.setdefault("error", "read_values returned None")
            modalities["force"] = force_meta

        obs_read_end = time.monotonic_ns()
        self.last_obs_meta = {
            "obs_read_start_mono_ns": obs_read_start,
            "obs_read_end_mono_ns": obs_read_end,
            "modalities": modalities,
        }

        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
