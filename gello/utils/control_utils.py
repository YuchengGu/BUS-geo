"""Shared utilities for robot control loops."""

import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Agent
from gello.env import RobotEnv

DEFAULT_MAX_JOINT_DELTA = 1.0


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

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Update save interface and handle saving.

        Args:
            obs: Current observations
            action: Current action

        Returns:
            Optional[str]: "quit" if user wants to exit, None otherwise
        """
        from gello.data_utils.format_obs import save_frame

        dt = datetime.datetime.now()
        state = self.kb_interface.update()

        if state == "start":
            dt_time = datetime.datetime.now()
            self.save_path = (
                self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {self.save_path}")
        elif state == "save":
            if self.save_path is not None:
                save_frame(self.save_path, dt, obs, action)
        elif state == "normal":
            self.save_path = None
        elif state == "quit":
            print("\nExiting.")
            return "quit"
        else:
            raise ValueError(f"Invalid state {state}")

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
        action = agent.act(obs)
        
        # 2. 获取机器人实际关节数 (例如 6 维)
        robot_dof = len(obs["joint_positions"])

        # 3. 【关键】自动裁剪：如果 action 是 7 维，这里切成 6 维
        if len(action) > robot_dof:
            action = action[:robot_dof]

        # Handle save interface
        if save_interface is not None:
            # 注意：保存的数据也建议是裁剪后的，否则回放时会报错
            result = save_interface.update(obs, action)
            if result == "quit":
                break

        # 4. 现在 action 肯定是 6 维了，传给 env.step 就不会报错了
        obs = env.step(action)
