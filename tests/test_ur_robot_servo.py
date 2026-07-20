from __future__ import annotations

import numpy as np
import pytest

from gello.robots.ur import URRobot


class FakeRTDEControl:
    def __init__(self, *, servo_result=True, stop_result=True):
        self.servo_result = servo_result
        self.stop_result = stop_result
        self.waited = False

    def initPeriod(self):
        return 123

    def servoJ(self, *_args):
        return self.servo_result

    def servoL(self, *_args):
        return self.servo_result

    def waitPeriod(self, _token):
        self.waited = True

    def servoStop(self):
        return self.stop_result


def _ur(control: FakeRTDEControl) -> URRobot:
    robot = URRobot.__new__(URRobot)
    robot.robot = control
    robot._use_gripper = False
    return robot


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("command_joint_state", np.zeros(6)),
        ("command_tcp_pose", np.zeros(6)),
    ],
)
def test_ur_robot_raises_when_rtde_rejects_servo_command(method, value):
    robot = _ur(FakeRTDEControl(servo_result=False))

    with pytest.raises(RuntimeError, match="rejected"):
        getattr(robot, method)(value)


def test_ur_robot_exposes_servo_stop_and_checks_result():
    accepted = _ur(FakeRTDEControl(stop_result=True))
    rejected = _ur(FakeRTDEControl(stop_result=False))

    accepted.stop_servo()
    with pytest.raises(RuntimeError, match="servoStop"):
        rejected.stop_servo()
