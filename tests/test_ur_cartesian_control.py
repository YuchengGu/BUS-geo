import numpy as np

from gello.robots.ur import URRobot


class FakeRTDEControl:
    def __init__(self):
        self.servo_l_calls = []
        self.servo_j_calls = []
        self.waited = []

    def initPeriod(self):
        return "period"

    def waitPeriod(self, period):
        self.waited.append(period)

    def servoL(self, pose, velocity, acceleration, dt, lookahead_time, gain):
        self.servo_l_calls.append((pose, velocity, acceleration, dt, lookahead_time, gain))

    def servoJ(self, joints, velocity, acceleration, dt, lookahead_time, gain):
        self.servo_j_calls.append((joints, velocity, acceleration, dt, lookahead_time, gain))


def test_ur_robot_command_tcp_pose_uses_servo_l_without_changing_joint_command():
    robot = object.__new__(URRobot)
    robot.robot = FakeRTDEControl()
    robot._use_gripper = False

    tcp_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 1.57])
    robot.command_tcp_pose(tcp_pose)

    assert len(robot.robot.servo_l_calls) == 1
    assert robot.robot.servo_j_calls == []
    np.testing.assert_allclose(robot.robot.servo_l_calls[0][0], tcp_pose)

