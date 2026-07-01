import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.episode_recorder import EpisodeRecorder
from visual_guided_collection_gui.main import build_parser
from visual_guided_collection_gui.state import GuiStage, enabled_actions_for_stage
from visual_guided_collection_gui.surface_random_local import random_local_start_target


def test_random_local_start_never_selects_last_path_point():
    path = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ]
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1)),
    )
    rng = np.random.default_rng(123)

    indices = [
        random_local_start_target(path, tip_height_m=0.15, probe_length_m=0.2, rng=rng).index
        for _ in range(100)
    ]

    assert max(indices) == 2
    assert 3 not in indices


def test_random_local_start_places_tip_at_requested_normal_height():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )

    target = random_local_start_target(
        path,
        tip_height_m=0.12,
        probe_length_m=0.2,
        index=0,
    )

    np.testing.assert_allclose(target.tip_position_base, [0.0, 0.0, 0.12])
    np.testing.assert_allclose(target.tcp_pose_base[:3], [0.0, 0.0, 0.32])
    assert target.meta["random_start_index"] == 0
    assert target.meta["episode_mode"] == "random_local"
    assert target.meta["random_start_tip_height_m"] == 0.12


def test_random_local_start_button_is_surface_teleop_action():
    actions = enabled_actions_for_stage(GuiStage.TELEOP_READY)

    assert "surface_random_local_start" in actions


def test_random_local_episodes_are_opt_in_from_cli():
    defaults = build_parser().parse_args([])
    enabled = build_parser().parse_args(["--surface-random-local-episodes"])

    assert defaults.surface_random_local_episodes is False
    assert enabled.surface_random_local_episodes is True


def test_confirm_path_skips_first_point_motion_only_in_random_local_mode():
    from pathlib import Path

    app_source = Path("visual_guided_collection_gui/app.py").read_text(encoding="utf-8")
    method_source = app_source.split("    def _on_confirm_path", 1)[1].split(
        "    def _surface_path_confirmed", 1
    )[0]

    assert "if self.args.control_tcp and self.args.surface_random_local_episodes:" in method_source
    assert "Path confirmed for random local episodes" in method_source


def test_episode_recorder_adds_episode_context_to_sample_meta(tmp_path):
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    recorder = EpisodeRecorder(
        data_dir=tmp_path,
        agent_name="gello",
        planned_path=path,
        probe_tip_offset_m=0.2,
        episode_context={"episode_mode": "random_local", "random_start_index": 1},
    )

    recorder.start("episode")
    obs = {
        "tcp_position_base": np.array([0.0, 0.0, 0.2]),
        "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
        "tcp_y_axis_base": np.array([0.0, 1.0, 0.0]),
        "tcp_z_axis_base": np.array([0.0, 0.0, 1.0]),
    }
    recorder.save_sample(obs, np.zeros(6), meta={"sample_semantics": "obs_t_to_action_t"})

    saved = list((tmp_path / "gello" / "episode").glob("*.pkl"))
    assert len(saved) == 1
    import pickle

    frame = pickle.loads(saved[0].read_bytes())
    assert frame["meta"]["episode_mode"] == "random_local"
    assert frame["meta"]["random_start_index"] == 1
    assert frame["meta"]["sample_semantics"] == "obs_t_to_action_t"


def test_surface_recalibration_stops_existing_surface_loop_before_restart():
    from pathlib import Path

    app_source = Path("visual_guided_collection_gui/app.py").read_text(encoding="utf-8")
    method_source = app_source.split("    def _on_surface_calibrate_z", 1)[1].split(
        "    def _on_surface_recenter", 1
    )[0]

    assert "self.teleop_loop.stop()" in method_source
