from __future__ import annotations

import glob
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


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
