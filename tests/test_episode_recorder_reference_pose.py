import datetime
import pickle

import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.episode_recorder import EpisodeRecorder


def test_episode_recorder_saves_path_reference_tcp_pose_sequence(tmp_path):
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
    )
    recorder = EpisodeRecorder(
        data_dir=tmp_path,
        agent_name="gello",
        planned_path=path,
        probe_tip_offset_m=0.2,
    )
    recorder.start(episode_id="episode_reference")

    obs = {
        "joint_positions": np.zeros(6),
        "joint_velocities": np.zeros(6),
        "ee_pos_quat": np.zeros(7),
        "tcp_position_base": np.array([0.11, 0.0, 0.2]),
        "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
        "tcp_y_axis_base": np.array([0.0, -1.0, 0.0]),
        "tcp_z_axis_base": np.array([0.0, 0.0, -1.0]),
    }
    timestamp = datetime.datetime(2026, 5, 20, 12, 0, 0)

    enriched = recorder.save_sample(obs, np.zeros(6), timestamp=timestamp)

    saved_file = tmp_path / "gello" / "episode_reference" / f"{timestamp.isoformat()}.pkl"
    with open(saved_file, "rb") as f:
        frame = pickle.load(f)

    np.testing.assert_allclose(enriched["path_residuals_base"][0], [-0.01, 0.0, 0.0])
    np.testing.assert_allclose(frame["path_reference_tcp_positions_base"][0], [0.1, 0.0, 0.2])
    np.testing.assert_allclose(frame["path_reference_tcp_poses_base"][0], [0.1, 0.0, 0.2, 0.0, np.pi, 0.0], atol=1e-8)
    assert "path_tcp_frames_base" not in frame
    assert "path_darboux_frames_base" not in frame
