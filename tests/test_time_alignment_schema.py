import datetime
import pickle

import numpy as np

from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv


class FakeRobot:
    def __init__(self):
        self.last_command = None

    def num_dofs(self):
        return 6

    def command_joint_state(self, joint_state):
        self.last_command = np.asarray(joint_state)

    def get_observations(self):
        return {
            "joint_positions": np.arange(6, dtype=float),
            "joint_velocities": np.ones(6, dtype=float),
            "ee_pos_quat": np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
        }


class FakeCamera:
    def __init__(self):
        self.last_metadata = {
            "valid": True,
            "frame_new": False,
            "frame_id": 7,
            "cache_age_ms": 12.5,
        }

    def read(self, img_size=None):
        return (
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros((2, 3, 1), dtype=np.uint16),
        )


class FailingForce:
    def __init__(self):
        self.last_metadata = {"valid": False, "error": "timeout"}

    def read_values(self):
        return None


def test_save_frame_preserves_legacy_fields_and_writes_meta(tmp_path):
    timestamp = datetime.datetime(2026, 5, 12, 12, 0, 0)
    obs = {"joint_positions": np.array([1.0, 2.0])}
    action = np.array([0.1, 0.2])
    meta = {
        "schema_version": "time_alignment_v1",
        "sample_index": 3,
        "sample_semantics": "obs_t_to_action_t",
    }

    save_frame(tmp_path, timestamp, obs, action, meta=meta)

    with open(tmp_path / f"{timestamp.isoformat()}.pkl", "rb") as f:
        saved = pickle.load(f)

    np.testing.assert_allclose(saved["joint_positions"], obs["joint_positions"])
    np.testing.assert_allclose(saved["control"], action)
    assert saved["meta"] == meta


def test_robot_env_records_obs_metadata_without_changing_obs_shape():
    env = RobotEnv(
        FakeRobot(),
        control_rate_hz=1_000_000,
        camera_dict={"D405": FakeCamera()},
        force_sensor=FailingForce(),
    )

    obs = env.get_obs()
    meta = env.last_obs_meta

    assert set(obs) >= {
        "D405_rgb",
        "D405_depth",
        "joint_positions",
        "joint_velocities",
        "ee_pos_quat",
        "force",
    }
    assert "meta" not in obs
    assert meta["obs_read_start_mono_ns"] <= meta["obs_read_end_mono_ns"]
    assert meta["modalities"]["D405"]["frame_new"] is False
    assert meta["modalities"]["D405"]["frame_id"] == 7
    assert meta["modalities"]["D405"]["cache_age_ms"] == 12.5
    assert meta["modalities"]["robot"]["valid"] is True
    assert meta["modalities"]["force"]["valid"] is False
    assert meta["modalities"]["force"]["error"] == "timeout"
    np.testing.assert_allclose(obs["force"], np.zeros(6))


def test_robot_env_step_records_action_send_timing():
    robot = FakeRobot()
    env = RobotEnv(robot, control_rate_hz=1_000_000)
    action = np.arange(6, dtype=float)

    env.step(action)

    timing = env.last_step_timing
    np.testing.assert_allclose(robot.last_command, action)
    assert timing["action_send_start_mono_ns"] <= timing["action_send_end_mono_ns"]
    assert timing["step_start_mono_ns"] <= timing["action_send_start_mono_ns"]
    assert timing["action_send_end_mono_ns"] <= timing["step_end_mono_ns"]


def test_episode_summary_reports_legacy_and_camera_cache_ratio(tmp_path):
    from chack_data import summarize_episode

    legacy_ts = datetime.datetime(2026, 5, 12, 12, 0, 0)
    save_frame(tmp_path, legacy_ts, {"joint_positions": np.zeros(1)}, np.zeros(1))

    meta_ts = datetime.datetime(2026, 5, 12, 12, 0, 1)
    save_frame(
        tmp_path,
        meta_ts,
        {"joint_positions": np.ones(1)},
        np.ones(1),
        meta={
            "schema_version": "time_alignment_v1",
            "sample_index": 1,
            "timing": {"action_send_start_mono_ns": 2_000_000_000},
            "modalities": {"D405": {"frame_new": False, "valid": True}},
        },
    )

    summary = summarize_episode(tmp_path)

    assert summary["num_frames"] == 2
    assert summary["legacy_frames"] == 1
    assert summary["camera_cache_ratio"]["D405"] == 1.0
