import datetime
import pickle

import numpy as np

from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv


class FakeRobotWithoutGripper:
    def num_dofs(self):
        return 6

    def get_joint_state(self):
        return np.zeros(6)

    def command_joint_state(self, joint_state):
        self.last_command = np.asarray(joint_state)

    def get_observations(self):
        return {
            "joint_positions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "joint_velocities": np.array([0.0, 0.1, 0.0, -0.1, 0.2, -0.2]),
            "ee_pos_quat": np.array([0.4, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0]),
            "ee_pos_rotvec": np.array([0.4, -0.1, 0.2, 0.0, 0.0, 0.0]),
        }


def test_env_accepts_lowdim_observations_without_gripper_position():
    env = RobotEnv(FakeRobotWithoutGripper(), control_rate_hz=100)

    obs = env.get_obs()

    assert "gripper_position" not in obs
    np.testing.assert_allclose(obs["joint_positions"], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    np.testing.assert_allclose(obs["joint_velocities"], np.array([0.0, 0.1, 0.0, -0.1, 0.2, -0.2]))
    np.testing.assert_allclose(obs["ee_pos_rotvec"], np.array([0.4, -0.1, 0.2, 0.0, 0.0, 0.0]))


def test_save_frame_does_not_add_fake_gripper_position(tmp_path):
    env = RobotEnv(FakeRobotWithoutGripper(), control_rate_hz=100)
    obs = env.get_obs()
    action = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    timestamp = datetime.datetime(2026, 5, 11, 10, 0, 0)

    save_frame(tmp_path, timestamp, obs, action)

    saved_path = tmp_path / (timestamp.isoformat() + ".pkl")
    with open(saved_path, "rb") as f:
        saved = pickle.load(f)

    assert "gripper_position" not in saved
    np.testing.assert_allclose(saved["control"], action)
