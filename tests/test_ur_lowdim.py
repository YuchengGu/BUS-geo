import numpy as np

from gello.robots.ur import URRobot


class FakeRTDEReceive:
    def getActualQ(self):
        return [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]

    def getActualQd(self):
        return [1.1, -1.2, 1.3, -1.4, 1.5, -1.6]

    def getActualTCPPose(self):
        return [0.4, -0.1, 0.2, 0.0, 0.0, np.pi / 2]


def make_ur_without_gripper():
    robot = URRobot.__new__(URRobot)
    robot.r_inter = FakeRTDEReceive()
    robot._use_gripper = False
    return robot


def test_ur_lowdim_uses_actual_joint_velocity_and_tcp_pose():
    robot = make_ur_without_gripper()

    obs = robot.get_observations()

    np.testing.assert_allclose(
        obs["joint_positions"],
        np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
    )
    np.testing.assert_allclose(
        obs["joint_velocities"],
        np.array([1.1, -1.2, 1.3, -1.4, 1.5, -1.6]),
    )
    assert not np.allclose(obs["joint_velocities"], obs["joint_positions"])
    np.testing.assert_allclose(
        obs["ee_pos_rotvec"],
        np.array([0.4, -0.1, 0.2, 0.0, 0.0, np.pi / 2]),
    )


def test_ur_lowdim_converts_tcp_rotvec_to_wxyz_quaternion():
    robot = make_ur_without_gripper()

    obs = robot.get_observations()

    np.testing.assert_allclose(obs["ee_pos_quat"][:3], np.array([0.4, -0.1, 0.2]))
    np.testing.assert_allclose(
        obs["ee_pos_quat"][3:],
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    )


def test_ur_lowdim_omits_gripper_position_when_gripper_is_disabled():
    robot = make_ur_without_gripper()

    obs = robot.get_observations()

    assert "gripper_position" not in obs
