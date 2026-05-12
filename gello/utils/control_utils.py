"""Shared utilities for robot control loops."""

import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Agent
from gello.env import RobotEnv

DEFAULT_MAX_JOINT_DELTA = 1.0


def build_time_alignment_meta(
    *,
    sample_index: int,
    episode_id: Optional[str],
    control_loop_hz_config: Optional[float],
    wall_time: datetime.datetime,
    obs_meta: Optional[Dict[str, Any]],
    action_timing: Dict[str, int],
    step_timing: Dict[str, int],
) -> Dict[str, Any]:
    obs_meta = obs_meta or {}
    timing: Dict[str, Any] = {}
    for key in ("obs_read_start_mono_ns", "obs_read_end_mono_ns"):
        if key in obs_meta:
            timing[key] = obs_meta[key]
    timing.update(action_timing)
    timing.update(step_timing)

    sample_mono_ns = timing.get("action_send_start_mono_ns")
    if sample_mono_ns is None:
        sample_mono_ns = timing.get("agent_act_start_mono_ns")

    return {
        "schema_version": "time_alignment_v1",
        "episode_id": episode_id,
        "sample_index": sample_index,
        "wall_time_iso": wall_time.isoformat(),
        "sample_mono_ns": sample_mono_ns,
        "control_loop_hz_config": control_loop_hz_config,
        "sample_semantics": "obs_t_to_action_t",
        "timing": timing,
        "modalities": dict(obs_meta.get("modalities", {})),
    }


def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually.

    Args:
        env: Robot environment
        agent: Agent that provides target position
        max_delta: Maximum joint delta per step
        steps: Number of steps for gradual movement

    Returns:
        bool: True if successful, False if position too far
    """
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = DEFAULT_MAX_JOINT_DELTA
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return False

    # print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    # assert len(start_pos) == len(
    #     joints
    # ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    # for _ in range(steps):
    #     obs = env.get_obs()
    #     command_joints = agent.act(obs)
    #     current_joints = obs["joint_positions"]
    #     delta = command_joints - current_joints
    #     max_joint_delta = np.abs(delta).max()
    #     if max_joint_delta > max_delta:
    #         delta = delta / max_joint_delta * max_delta
    #     env.step(current_joints + delta)

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    
    # 1. 修改断言：允许手柄数据比机器人多 (>=)
    assert len(start_pos) >= len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]

        # 2. 【关键】自动裁剪：如果手柄数据多，就只取前几个匹配机器人的
        if len(command_joints) > len(current_joints):
            command_joints = command_joints[:len(current_joints)]

        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    return True


class SaveInterface:
    """Handles keyboard-based data saving interface."""

    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        """Initialize save interface.

        Args:
            data_dir: Base directory for saving data
            agent_name: Name of agent (used for subdirectory)
            expand_user: Whether to expand ~ in data_dir path
        """
        from gello.data_utils.keyboard_interface import KBReset

        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None

        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording")
        print("  Q: Stop recording")

    def _start_recording(self) -> None:
        dt_time = datetime.datetime.now()
        self.save_path = (
            self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
        )
        self.save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {self.save_path}")

    def poll(self) -> str:
        state = self.kb_interface.update()

        if state == "start":
            self._start_recording()
        elif state == "normal":
            self.save_path = None
        elif state == "quit":
            print("\nExiting.")
        elif state != "save":
            raise ValueError(f"Invalid state {state}")

        return state

    def save(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime.datetime] = None,
    ) -> None:
        if self.save_path is None:
            return

        from gello.data_utils.format_obs import save_frame

        save_frame(
            self.save_path,
            timestamp or datetime.datetime.now(),
            obs,
            action,
            meta=meta,
        )

    def update(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Update save interface and handle saving.

        Args:
            obs: Current observations
            action: Current action
            meta: Optional time-alignment metadata

        Returns:
            Optional[str]: "quit" if user wants to exit, None otherwise
        """
        state = self.poll()

        if state == "save":
            self.save(obs, action, meta=meta)
        elif state == "quit":
            return "quit"

        return None


def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

    start_time = time.time()
    obs = env.get_obs()
    sample_index = 0

    # while True:
    #     if print_timing:
    #         num = time.time() - start_time
    #         message = f"\rTime passed: {round(num, 2)}          "

    #         if colors_available:
    #             print(
    #                 colored(message, color="white", attrs=["bold"]), end="", flush=True
    #             )
    #         else:
    #             print(message, end="", flush=True)

    #     action = agent.act(obs)

    #     # Handle save interface
    #     if save_interface is not None:
    #         result = save_interface.update(obs, action)
    #         if result == "quit":
    #             break

    #     obs = env.step(action)

    while True:
        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            if colors_available:
                print(colored(message, color="white", attrs=["bold"]), end="", flush=True)
            else:
                print(message, end="", flush=True)

        # 1. 获取手柄原始数据 (可能是 7 维)
        agent_act_start = time.monotonic_ns()
        action = agent.act(obs)
        agent_act_end = time.monotonic_ns()
        
        # 2. 获取机器人实际关节数 (例如 6 维)
        robot_dof = len(obs["joint_positions"])

        # 3. 【关键】自动裁剪：如果 action 是 7 维，这里切成 6 维
        if len(action) > robot_dof:
            action = action[:robot_dof]

        save_state = None
        if save_interface is not None:
            save_state = save_interface.poll()
            if save_state == "quit":
                break
            if save_state == "start":
                sample_index = 0

        # 4. 现在 action 肯定是 6 维了，传给 env.step 就不会报错了
        current_obs = obs
        current_obs_meta = dict(getattr(env, "last_obs_meta", {}) or {})
        obs = env.step(action)

        if (
            save_interface is not None
            and save_state == "save"
            and save_interface.save_path is not None
        ):
            timestamp = datetime.datetime.now()
            meta = build_time_alignment_meta(
                sample_index=sample_index,
                episode_id=save_interface.save_path.name,
                control_loop_hz_config=getattr(env, "control_rate_hz", None),
                wall_time=timestamp,
                obs_meta=current_obs_meta,
                action_timing={
                    "agent_act_start_mono_ns": agent_act_start,
                    "agent_act_end_mono_ns": agent_act_end,
                },
                step_timing=dict(getattr(env, "last_step_timing", {}) or {}),
            )
            save_interface.save(current_obs, action, meta=meta, timestamp=timestamp)
            sample_index += 1
