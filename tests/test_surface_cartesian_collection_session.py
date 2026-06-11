import pickle
import time

import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.collection_session import LoopConfig, TeleopLoop
from visual_guided_collection_gui.episode_recorder import EpisodeRecorder


class FakeSurfaceDevices:
    class Env:
        control_rate_hz = 50.0
        last_step_timing = {"action_mode": "tcp_pose"}

    def __init__(self):
        self.env = self.Env()
        self.count = 0

    def get_obs(self):
        return {
            "joint_positions": np.zeros(6),
            "joint_velocities": np.zeros(6),
            "ee_pos_quat": np.zeros(7),
            "tcp_position_base": np.array([0.0, 0.0, 0.2]),
            "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
            "tcp_y_axis_base": np.array([0.0, -1.0, 0.0]),
            "tcp_z_axis_base": np.array([0.0, 0.0, -1.0]),
            "ee_pos_rotvec": np.zeros(6),
        }

    def step_surface_cartesian_teleop(self, _controller, obs):
        self.count += 1
        action = np.array([0.01 * self.count, 0.0, 0.2, np.pi, 0.0, 0.0])
        next_obs = dict(obs)
        next_obs["ee_pos_rotvec"] = action.copy()
        return next_obs, action, {}, {"action_mode": "tcp_pose"}


def test_surface_cartesian_recording_saves_tcp_pose_control(tmp_path):
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    recorder = EpisodeRecorder(
        data_dir=tmp_path,
        agent_name="gello",
        planned_path=path,
        probe_tip_offset_m=0.2,
    )
    recorder.start("surface_episode")
    loop = TeleopLoop(
        devices=FakeSurfaceDevices(),
        config=LoopConfig(idle_sleep_s=0.001),
    )
    samples = []

    loop.start_surface_recording(
        controller=object(),
        recorder=recorder,
        on_sample=samples.append,
    )
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline and not samples:
        time.sleep(0.001)
    loop.stop()

    saved = sorted((tmp_path / "gello" / "surface_episode").glob("*.pkl"))
    assert saved
    with open(saved[0], "rb") as f:
        frame = pickle.load(f)
    np.testing.assert_allclose(frame["control"], [0.01, 0.0, 0.2, np.pi, 0.0, 0.0])
    assert frame["meta"]["timing"]["action_mode"] == "tcp_pose"
