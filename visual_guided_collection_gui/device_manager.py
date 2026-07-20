from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from visual_guided_collection_gui.force_gravity import (
    DemoForceGravityCompensation,
    ForceGravityCalibration,
    demo1_force_gravity_compensation,
    load_force_gravity_calibration,
)
from visual_guided_collection_gui.gello_kinematics import gello_ur5_joint_action_to_tcp_pose
from visual_guided_collection_gui.surface_teleop import interpolate_tcp_poses


def _interpolate_joint_positions(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_joint_step_rad: float,
) -> list[np.ndarray]:
    start = np.asarray(current, dtype=float).reshape(-1)
    goal = np.asarray(target, dtype=float).reshape(-1)
    if start.shape != goal.shape:
        raise ValueError(f"current/target joint shapes must match, got {start.shape} and {goal.shape}")
    step = abs(float(max_joint_step_rad))
    if step <= 0.0:
        return [goal.copy()]
    delta = goal - start
    count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / step)))
    return [start + delta * (index / count) for index in range(1, count + 1)]


def _add_force_derived_observations(obs: dict[str, Any]) -> dict[str, Any]:
    force = np.asarray(obs.get("force", np.zeros(6)), dtype=float).reshape(-1)
    raw = np.asarray(obs.get("force_raw", force), dtype=float).reshape(-1)
    if force.size >= 6:
        obs["force_normal_z_n"] = float(force[2])
        obs["force_pressure_n"] = float(-force[2])
        obs["force_tangential_norm_n"] = float(np.linalg.norm(force[:2]))
        obs["force_shear_ratio"] = float(np.linalg.norm(force[:2]) / (abs(float(force[2])) + 1e-6))
        obs["torque_axial_z_nm"] = float(force[5])
        obs["torque_tangential_norm_nm"] = float(np.linalg.norm(force[3:5]))
    if raw.size >= 6:
        obs["force_raw_normal_z_n"] = float(raw[2])
        obs["force_raw_pressure_n"] = float(-raw[2])
        obs["force_raw_tangential_norm_n"] = float(np.linalg.norm(raw[:2]))
        obs["force_raw_shear_ratio"] = float(np.linalg.norm(raw[:2]) / (abs(float(raw[2])) + 1e-6))
        obs["torque_raw_axial_z_nm"] = float(raw[5])
        obs["torque_raw_tangential_norm_nm"] = float(np.linalg.norm(raw[3:5]))
    return obs


@dataclass
class DeviceConfig:
    hostname: str = "127.0.0.1"
    robot_port: int = 6001
    hz: float = 50.0
    agent_name: str = "gello"
    gello_port: str | None = None
    force_ip: str = "192.168.1.100"
    use_force: bool = True
    force_zero_on_connect: bool = True
    force_zero_samples: int = 20
    force_zero_sample_interval_s: float = 0.005
    force_gravity_calib_path: str | None = None
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
        self.force_bias: np.ndarray | None = None
        self.force_gravity_calibration: ForceGravityCalibration | DemoForceGravityCompensation | None = None
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
        self.force_bias = None
        self.force_gravity_calibration = None
        if self.config.use_force:
            self.force_sensor = ForceSensorMTCP(ip=self.config.force_ip)
            self.force_sensor.connect()
            if self.config.force_gravity_calib_path:
                self.force_gravity_calibration = load_force_gravity_calibration(
                    self.config.force_gravity_calib_path
                )
                self.force_bias = self.force_gravity_calibration.bias_wrench
                print(
                    "[系统] 已加载力传感器重力标定: "
                    f"{self.config.force_gravity_calib_path}, "
                    f"samples={self.force_gravity_calibration.sample_count}, "
                    f"gravity_base={self.force_gravity_calibration.gravity_base.tolist()}"
                )
            else:
                self.force_gravity_calibration = demo1_force_gravity_compensation()
                print(
                    "[系统] 使用 demo1 硬编码力传感器重力模型: "
                    f"mass={self.force_gravity_calibration.mass_kg}, "
                    f"com={self.force_gravity_calibration.com_sensor_m.tolist()}"
                )
                if self.config.force_zero_on_connect:
                    try:
                        bias = self.zero_force_with_gravity(
                            samples=self.config.force_zero_samples,
                            sample_interval_s=self.config.force_zero_sample_interval_s,
                        )
                        print(f"[系统] 力传感器悬空重力补偿归零完成: {bias.tolist()}")
                    except Exception as exc:
                        self.force_bias = self.force_gravity_calibration.bias_wrench
                        print(f"[警告] 力传感器重力补偿归零失败，将使用零 bias: {exc}")
            if self.force_gravity_calibration is None and self.config.force_zero_on_connect:
                try:
                    bias = self.zero_force(
                        samples=self.config.force_zero_samples,
                        sample_interval_s=self.config.force_zero_sample_interval_s,
                    )
                    print(f"[系统] 力传感器单姿态归零完成: {bias.tolist()}")
                except Exception as exc:
                    print(f"[警告] 力传感器单姿态归零失败，将使用原始力数据: {exc}")

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
            self.force_bias = None
            self.force_gravity_calibration = None
            self.env = None
            self.agent = None
            self.robot_client = None

    def get_obs(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        with self._io_lock:
            return self._apply_force_compensation(self.env.get_obs())

    def get_force_obs(self) -> dict[str, Any]:
        if self.force_sensor is None:
            return {
                "force": None,
                "force_sensor_valid": False,
                "force_sensor_error": "force sensor is not connected",
            }
        with self._io_lock:
            tcp_pose = None
            if self.robot_client is not None:
                robot_obs = self.robot_client.get_observations()
                if "ee_pos_rotvec" in robot_obs:
                    tcp_pose = np.asarray(robot_obs["ee_pos_rotvec"], dtype=float).reshape(6)
            values = self.force_sensor.read_values()
            metadata = dict(getattr(self.force_sensor, "last_metadata", {}) or {})
            if values is None:
                return {
                    "ee_pos_rotvec": tcp_pose,
                    "force": None,
                    "force_sensor_valid": False,
                    "force_sensor_error": metadata.get("error", "force read failed"),
                }
            force = np.asarray(values, dtype=float).reshape(-1)
            if force.size < 6 or not np.all(np.isfinite(force[:6])):
                return {
                    "ee_pos_rotvec": tcp_pose,
                    "force": None,
                    "force_raw": force.copy(),
                    "force_sensor_valid": False,
                    "force_sensor_error": "invalid force vector",
                }
            obs = {
                "force": force[:6].copy(),
                "ee_pos_rotvec": tcp_pose,
                "force_sensor_valid": bool(metadata.get("valid", True)),
                "force_sensor_error": metadata.get("error"),
            }
            if tcp_pose is not None:
                return self._apply_force_compensation(obs)
            obs["force_raw"] = force[:6].copy()
            obs["force_bias"] = np.zeros(6, dtype=float)
            obs["force_gravity"] = np.zeros(6, dtype=float)
            obs["force_gravity_calibrated"] = False
            obs["force_zeroed"] = self.force_bias is not None
            return _add_force_derived_observations(obs)

    def zero_force(self, *, samples: int = 20, sample_interval_s: float = 0.005) -> np.ndarray:
        if self.force_sensor is None:
            raise RuntimeError("Force sensor is not connected")
        force_samples = []
        for _ in range(max(1, int(samples))):
            values = self.force_sensor.read_values()
            if values is not None:
                force = np.asarray(values, dtype=float).reshape(-1)
                if force.size >= 6 and np.all(np.isfinite(force[:6])):
                    force_samples.append(force[:6].copy())
            if sample_interval_s > 0:
                time.sleep(float(sample_interval_s))
        if not force_samples:
            raise RuntimeError("No valid force samples were read")
        self.force_bias = np.mean(np.vstack(force_samples), axis=0)
        return self.force_bias.copy()

    def zero_force_with_gravity(self, *, samples: int = 20, sample_interval_s: float = 0.005) -> np.ndarray:
        if self.force_sensor is None:
            raise RuntimeError("Force sensor is not connected")
        if self.robot_client is None:
            raise RuntimeError("Robot client is not connected")
        if not isinstance(self.force_gravity_calibration, DemoForceGravityCompensation):
            raise RuntimeError("Demo gravity model is not active")
        robot_obs = self.robot_client.get_observations()
        if "ee_pos_rotvec" not in robot_obs:
            raise RuntimeError("Robot observation does not include ee_pos_rotvec")
        tcp_pose = np.asarray(robot_obs["ee_pos_rotvec"], dtype=float).reshape(6)
        gravity = self.force_gravity_calibration.gravity_wrench_sensor(tcp_pose)
        force_samples = []
        for _ in range(max(1, int(samples))):
            values = self.force_sensor.read_values()
            if values is not None:
                force = np.asarray(values, dtype=float).reshape(-1)
                if force.size >= 6 and np.all(np.isfinite(force[:6])):
                    force_samples.append(force[:6].copy())
            if sample_interval_s > 0:
                time.sleep(float(sample_interval_s))
        if not force_samples:
            raise RuntimeError("No valid force samples were read")
        raw_mean = np.mean(np.vstack(force_samples), axis=0)
        bias = raw_mean - gravity
        self.force_gravity_calibration = self.force_gravity_calibration.with_bias(bias)
        self.force_bias = self.force_gravity_calibration.bias_wrench
        return self.force_bias.copy()

    def _apply_force_compensation(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "force" not in obs:
            return obs
        compensated = dict(obs)
        raw = np.asarray(obs["force"], dtype=float).reshape(-1)
        compensated["force_raw"] = raw.copy()
        force_meta = {}
        if self.env is not None:
            obs_meta = getattr(self.env, "last_obs_meta", {}) or {}
            force_meta = dict((obs_meta.get("modalities", {}) or {}).get("force", {}) or {})
        if force_meta.get("valid") is False:
            compensated["force"] = raw.copy()
            compensated["force_bias"] = np.zeros_like(raw)
            compensated["force_gravity"] = np.zeros_like(raw)
            compensated["force_gravity_calibrated"] = False
            compensated["force_zeroed"] = False
            return _add_force_derived_observations(compensated)
        if self.force_gravity_calibration is not None:
            if "ee_pos_rotvec" not in obs:
                raise RuntimeError("Force gravity compensation requires ee_pos_rotvec in observations")
            force, bias, gravity = self.force_gravity_calibration.compensate(
                raw_wrench=raw,
                tcp_pose=np.asarray(obs["ee_pos_rotvec"], dtype=float),
            )
            if force.shape != raw.shape:
                raise RuntimeError(f"Compensated force shape {force.shape} does not match force shape {raw.shape}")
            compensated["force"] = force
            compensated["force_bias"] = bias
            compensated["force_gravity"] = gravity
            compensated["force_gravity_calibrated"] = True
            compensated["force_zeroed"] = True
            return _add_force_derived_observations(compensated)
        if self.force_bias is None:
            compensated["force"] = raw.copy()
            compensated["force_bias"] = np.zeros_like(raw)
            compensated["force_gravity"] = np.zeros_like(raw)
            compensated["force_gravity_calibrated"] = False
            compensated["force_zeroed"] = False
            return _add_force_derived_observations(compensated)
        bias = np.asarray(self.force_bias, dtype=float).reshape(-1)
        if bias.shape != raw.shape:
            raise RuntimeError(f"Force bias shape {bias.shape} does not match force shape {raw.shape}")
        compensated["force"] = raw - bias
        compensated["force_bias"] = bias.copy()
        compensated["force_gravity"] = np.zeros_like(raw)
        compensated["force_gravity_calibrated"] = False
        compensated["force_zeroed"] = True
        return _add_force_derived_observations(compensated)

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

    def joint_handover_error(
        self,
        obs: dict[str, Any],
    ) -> tuple[float, np.ndarray]:
        target = self.read_action(obs)
        current = np.asarray(obs["joint_positions"], dtype=float).reshape(-1)
        if target.shape != current.shape:
            raise ValueError(
                f"GELLO target shape {target.shape} does not match robot joints "
                f"{current.shape}"
            )
        error = float(np.max(np.abs(target - current))) if current.size else 0.0
        return error, target

    def step_agent(self, obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, int]]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        with self._io_lock:
            agent_act_start = time.monotonic_ns()
            raw_action = self.read_action(obs)
            agent_act_end = time.monotonic_ns()
            action = self.clamp_action(obs, raw_action)
            current_obs_meta = dict(getattr(self.env, "last_obs_meta", {}) or {})
            next_obs = self._apply_force_compensation(self.env.step(action))
        action_timing = {
            "agent_act_start_mono_ns": agent_act_start,
            "agent_act_end_mono_ns": agent_act_end,
            "action_mode": "joint_position",
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
            next_obs = self._apply_force_compensation(self.env.step_tcp_pose(target_tcp_pose))
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
            next_obs = self._apply_force_compensation(self.env.step_tcp_pose(action))
            send_end = time.monotonic_ns()
        action_timing = {
            "tcp_pose_send_start_mono_ns": send_start,
            "tcp_pose_send_end_mono_ns": send_end,
            "action_mode": "tcp_pose",
        }
        return next_obs, action, current_obs_meta, action_timing

    def move_joint_positions_linear(
        self,
        joint_positions: np.ndarray,
        *,
        max_joint_step_rad: float = 0.01,
        joint_tolerance_rad: float = 0.01,
        timeout_s: float = 30.0,
        waypoint_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("DeviceManager.connect() must be called first")
        target = np.asarray(joint_positions, dtype=float).reshape(-1)
        obs = self.get_obs()
        current = np.asarray(obs["joint_positions"], dtype=float).reshape(-1)
        if current.shape[0] < target.shape[0]:
            raise ValueError(f"target has {target.shape[0]} joints but robot observation has {current.shape[0]}")
        current = current[: target.shape[0]]
        waypoints = _interpolate_joint_positions(
            current,
            target,
            max_joint_step_rad=max_joint_step_rad,
        )
        deadline = time.monotonic() + float(timeout_s)
        for index, waypoint in enumerate(waypoints, start=1):
            with self._io_lock:
                obs = self._apply_force_compensation(self.env.step(waypoint))
            if waypoint_callback is not None:
                waypoint_callback(
                    {
                        "kind": "joint_waypoint",
                        "index": index,
                        "count": len(waypoints),
                        "target_joint_positions": waypoint.tolist(),
                        "actual_joint_positions": np.asarray(obs["joint_positions"], dtype=float).reshape(-1).tolist(),
                    }
                )
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out while moving joints along interpolated safe retreat path")

        tolerance = abs(float(joint_tolerance_rad))
        while True:
            actual = np.asarray(obs["joint_positions"], dtype=float).reshape(-1)[: target.shape[0]]
            max_error = float(np.max(np.abs(actual - target)))
            if waypoint_callback is not None:
                waypoint_callback(
                    {
                        "kind": "joint_settle",
                        "target_joint_positions": target.tolist(),
                        "actual_joint_positions": actual.tolist(),
                        "max_joint_error_rad": max_error,
                    }
                )
            if max_error <= tolerance:
                with self._io_lock:
                    robot = self.env.robot()
                    stop_servo = getattr(robot, "stop_servo", None)
                    if callable(stop_servo):
                        stop_servo()
                return obs
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for safe joint target; "
                    f"max_joint_error={max_error:.6f} rad"
                )
            with self._io_lock:
                obs = self._apply_force_compensation(self.env.step(target))

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
