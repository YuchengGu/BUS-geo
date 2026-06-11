from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from visual_guided_collection_gui.gello_kinematics import gello_ur5_joint_action_to_tcp_pose
from visual_guided_collection_gui.surface_teleop import interpolate_tcp_poses


@dataclass
class DeviceConfig:
    hostname: str = "127.0.0.1"
    robot_port: int = 6001
    hz: float = 50.0
    agent_name: str = "gello"
    gello_port: str | None = None
    force_ip: str = "192.168.1.160"
    use_force: bool = True
    use_ultrasound: bool = True
    ultrasound_index: int = 5
    max_joint_step_rad: float = 0.0
    wrist_camera: str = "D405"


class DeviceManager:
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.robot_client = None
        self.camera = None
        self.ultrasound_camera = None
        self.force_sensor = None
        self.env = None
        self.agent = None
        self._io_lock = threading.RLock()

    def connect(self) -> None:
        from gello.agents.agent import DummyAgent
        from gello.agents.gello_agent import GelloAgent
        from gello.cameras.Ultrasound import UltrasoundCamera
        from gello.env import RobotEnv
        from gello.force_sensor_mtcp import ForceSensorMTCP
        from gello.zmq_core.robot_node import ZMQClientRobot

        self.robot_client = ZMQClientRobot(port=self.config.robot_port, host=self.config.hostname)
        camera_name = self.config.wrist_camera
        if camera_name == "D405":
            from gello.cameras.D405 import RealSenseD405

            self.camera = RealSenseD405()
        elif camera_name == "Orbbec":
            from gello.cameras.Orbbec import OrbbecCamera

            self.camera = OrbbecCamera()
        else:
            raise ValueError(f"Unsupported wrist camera: {camera_name}")
        camera_dict = {camera_name: self.camera}
        if self.config.use_ultrasound:
            self.ultrasound_camera = UltrasoundCamera(camera_index=self.config.ultrasound_index)
            camera_dict["Ultrasound"] = self.ultrasound_camera

        self.force_sensor = None
        if self.config.use_force:
            self.force_sensor = ForceSensorMTCP(ip=self.config.force_ip)
            self.force_sensor.connect()

        self.env = RobotEnv(
            self.robot_client,
            control_rate_hz=self.config.hz,
            camera_dict=camera_dict,
            force_sensor=self.force_sensor,
        )
        if self.config.agent_name == "gello":
            self.agent = GelloAgent(port=self._resolve_gello_port())
        elif self.config.agent_name in {"none", "dummy"}:
            self.agent = DummyAgent(num_dofs=self.robot_client.num_dofs())
        else:
            raise ValueError(f"Unsupported GUI agent: {self.config.agent_name}")

    def _resolve_gello_port(self) -> str:
        if self.config.gello_port is not None:
            return self.config.gello_port
        ports = glob.glob("/dev/serial/by-id/*")
        if not ports:
            raise ValueError("No GELLO port found; pass --gello-port")
        return ports[0]

    @property
    def connected(self) -> bool:
        return self.env is not None and self.agent is not None and self.camera is not None

    @property
    def wrist_camera_name(self) -> str:
        return self.config.wrist_camera

    def close(self) -> None:
        with self._io_lock:
            if self.camera is not None and hasattr(self.camera, "pipeline"):
                try:
                    self.camera.pipeline.stop()
                except Exception:
                    pass
            if self.ultrasound_camera is not None and hasattr(self.ultrasound_camera, "close"):
                try:
                    self.ultrasound_camera.close()
                except Exception:
                    pass
            elif self.ultrasound_camera is not None and hasattr(self.ultrasound_camera, "cap"):
                try:
                    self.ultrasound_camera.cap.release()
                except Exception:
                    pass
            self.camera = None
            self.ultrasound_camera = None
            self.force_sensor = None
            self.env = None
            self.agent = None
            self.robot_client = None

    def get_obs(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        with self._io_lock:
            return self.env.get_obs()

    def latest_realsense_frames(self):
        if self.camera is None:
            raise RuntimeError("D405 is not connected")
        if not hasattr(self.camera, "latest_frames"):
            raise RuntimeError("D405 camera does not expose latest_frames()")
        return self.camera.latest_frames()

    def camera_intrinsics(self) -> dict[str, float]:
        if self.camera is None:
            raise RuntimeError("Wrist camera is not connected")
        if not hasattr(self.camera, "latest_intrinsics"):
            raise RuntimeError(f"{self.wrist_camera_name} does not expose latest_intrinsics()")
        return self.camera.latest_intrinsics()

    def camera_depth_scale_m_per_unit(self) -> float:
        if self.camera is None:
            raise RuntimeError("Wrist camera is not connected")
        if hasattr(self.camera, "depth_scale_m_per_unit"):
            return float(self.camera.depth_scale_m_per_unit())
        metadata = getattr(self.camera, "last_metadata", {}) or {}
        scale = metadata.get("depth_scale_m_per_unit")
        if scale is None:
            raise RuntimeError(f"{self.wrist_camera_name} does not expose depth scale")
        return float(scale)

    def read_action(self, obs: dict[str, Any]) -> np.ndarray:
        if self.agent is None:
            raise RuntimeError("Agent is not connected")
        with self._io_lock:
            action = np.asarray(self.agent.act(obs), dtype=float)
        robot_dof = int(len(obs["joint_positions"]))
        if action.shape[0] > robot_dof:
            action = action[:robot_dof]
        return action

    def clamp_action(self, obs: dict[str, Any], action: np.ndarray) -> np.ndarray:
        current = np.asarray(obs["joint_positions"], dtype=float)
        command = np.asarray(action, dtype=float)
        limit = float(self.config.max_joint_step_rad)
        if limit <= 0.0:
            return command
        delta = command - current
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_delta > limit:
            delta = delta / max_delta * limit
            command = current + delta
        return command

    def step_agent(self, obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, int]]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        with self._io_lock:
            agent_act_start = time.monotonic_ns()
            raw_action = self.read_action(obs)
            agent_act_end = time.monotonic_ns()
            action = self.clamp_action(obs, raw_action)
            current_obs_meta = dict(getattr(self.env, "last_obs_meta", {}) or {})
            next_obs = self.env.step(action)
        action_timing = {
            "agent_act_start_mono_ns": agent_act_start,
            "agent_act_end_mono_ns": agent_act_end,
        }
        return next_obs, action, current_obs_meta, action_timing

    def gello_action_to_tcp_pose(self, action: np.ndarray) -> np.ndarray:
        return gello_ur5_joint_action_to_tcp_pose(action)

    def read_gello_tcp_pose(self, obs: dict[str, Any]) -> np.ndarray:
        return self.gello_action_to_tcp_pose(self.read_action(obs))

    def step_surface_cartesian_teleop(
        self,
        controller,
        obs: dict[str, Any],
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, int]]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        with self._io_lock:
            agent_act_start = time.monotonic_ns()
            gello_tcp_pose = self.read_gello_tcp_pose(obs)
            agent_act_end = time.monotonic_ns()
            current_obs_meta = dict(getattr(self.env, "last_obs_meta", {}) or {})
            target_tcp_pose = controller.update(
                gello_tcp_pose=gello_tcp_pose,
                ur_tcp_pose=np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6),
            )
            send_start = time.monotonic_ns()
            next_obs = self.env.step_tcp_pose(target_tcp_pose)
            send_end = time.monotonic_ns()
        action_timing = {
            "agent_act_start_mono_ns": agent_act_start,
            "agent_act_end_mono_ns": agent_act_end,
            "tcp_pose_send_start_mono_ns": send_start,
            "tcp_pose_send_end_mono_ns": send_end,
            "action_mode": "tcp_pose",
        }
        return next_obs, target_tcp_pose, current_obs_meta, action_timing

    def step_tcp_pose(self, tcp_pose: np.ndarray) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, int]]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        with self._io_lock:
            action = np.asarray(tcp_pose, dtype=float).reshape(-1)
            current_obs_meta = dict(getattr(self.env, "last_obs_meta", {}) or {})
            send_start = time.monotonic_ns()
            next_obs = self.env.step_tcp_pose(action)
            send_end = time.monotonic_ns()
        action_timing = {
            "tcp_pose_send_start_mono_ns": send_start,
            "tcp_pose_send_end_mono_ns": send_end,
            "action_mode": "tcp_pose",
        }
        return next_obs, action, current_obs_meta, action_timing

    def move_tcp_pose_linear(
        self,
        tcp_pose: np.ndarray,
        *,
        max_position_step_m: float = 0.01,
        max_rotation_step_rad: float = 0.05,
        position_tolerance_m: float = 0.002,
        rotation_tolerance_rad: float = 0.03,
        timeout_s: float = 10.0,
        waypoint_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        target = np.asarray(tcp_pose, dtype=float).reshape(6)
        obs = self.get_obs()
        current = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
        deadline = time.monotonic() + float(timeout_s)

        waypoints = interpolate_tcp_poses(
            current,
            target,
            max_position_step_m=max_position_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
        )
        for index, waypoint in enumerate(waypoints, start=1):
            before = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
            obs, _action, _obs_meta, _timing = self.step_tcp_pose(waypoint)
            actual = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
            if waypoint_callback is not None:
                waypoint_callback(
                    {
                        "kind": "waypoint",
                        "index": index,
                        "count": len(waypoints),
                        "target_tcp_pose": waypoint.tolist(),
                        "actual_before_tcp_pose": before.tolist(),
                        "actual_after_tcp_pose": actual.tolist(),
                        "position_error_m": float(np.linalg.norm(actual[:3] - waypoint[:3])),
                        "rotation_error_rad": float(np.linalg.norm(actual[3:] - waypoint[3:])),
                    }
                )
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out while moving TCP along interpolated approach path")

        while True:
            actual = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
            position_error = float(np.linalg.norm(actual[:3] - target[:3]))
            rotation_error = float(np.linalg.norm(actual[3:] - target[3:]))
            if waypoint_callback is not None:
                waypoint_callback(
                    {
                        "kind": "settle",
                        "target_tcp_pose": target.tolist(),
                        "actual_tcp_pose": actual.tolist(),
                        "position_error_m": position_error,
                        "rotation_error_rad": rotation_error,
                    }
                )
            if position_error <= position_tolerance_m and rotation_error <= rotation_tolerance_rad:
                return obs
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for TCP target; "
                    f"position_error={position_error:.6f} m, rotation_error={rotation_error:.6f} rad"
                )
            obs, _action, _obs_meta, _timing = self.step_tcp_pose(target)
