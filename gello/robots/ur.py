from typing import Dict

import numpy as np

from gello.robots.robot import Robot


def _rotvec_to_wxyz_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a [qw, qx, qy, qz] quaternion."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = rotvec / angle
    half_angle = angle / 2.0
    return np.concatenate([[np.cos(half_angle)], axis * np.sin(half_angle)])


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.15", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = np.asarray(self.r_inter.getActualQ(), dtype=float)
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        accepted = self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)
        if accepted is False:
            raise RuntimeError("RTDE servoJ command rejected")

    def command_tcp_pose(self, tcp_pose: np.ndarray) -> None:
        """Command the UR TCP to a Cartesian pose vector [x, y, z, rx, ry, rz]."""
        velocity = 0.25
        acceleration = 0.25
        dt = 1.0 / 500
        lookahead_time = 0.2
        gain = 100

        pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
        if pose.shape[0] != 6:
            raise ValueError(f"tcp_pose must have 6 values [x, y, z, rx, ry, rz], got {pose.shape[0]}")
        t_start = self.robot.initPeriod()
        accepted = self.robot.servoL(
            pose, velocity, acceleration, dt, lookahead_time, gain
        )
        self.robot.waitPeriod(t_start)
        if accepted is False:
            raise RuntimeError("RTDE servoL command rejected")

    def stop_servo(self) -> None:
        """Exit the active servoJ/servoL mode before changing motion modes."""
        accepted = self.robot.servoStop()
        if accepted is False:
            raise RuntimeError("RTDE servoStop command rejected")

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        robot_joints = np.asarray(self.r_inter.getActualQ(), dtype=float)
        robot_joint_velocities = np.asarray(self.r_inter.getActualQd(), dtype=float)
        tcp_pose = np.asarray(self.r_inter.getActualTCPPose(), dtype=float)
        ee_pos_quat = np.concatenate(
            [tcp_pose[:3], _rotvec_to_wxyz_quat(tcp_pose[3:])]
        )

        observations = {
            "joint_positions": robot_joints,
            "joint_velocities": robot_joint_velocities,
            "ee_pos_rotvec": tcp_pose,
            "ee_pos_quat": ee_pos_quat,
        }

        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            observations["joint_positions"] = np.append(robot_joints, gripper_pos)
            observations["joint_velocities"] = np.append(robot_joint_velocities, 0.0)
            observations["gripper_position"] = np.array([gripper_pos])

        return observations


def main():
    robot_ip = "192.168.1.15"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
