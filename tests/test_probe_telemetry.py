import numpy as np

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.probe_telemetry import obs_from_tcp_pose_rotvec, probe_path_telemetry_lines


def test_probe_path_telemetry_lines_show_tip_pose_and_nearest_path_distance():
    path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
    )
    obs = {
        "tcp_position_base": np.array([0.1, 0.0, 0.2]),
        "tcp_x_axis_base": np.array([1.0, 0.0, 0.0]),
        "tcp_y_axis_base": np.array([0.0, 1.0, 0.0]),
        "tcp_z_axis_base": np.array([0.0, 0.0, -1.0]),
    }

    lines, nearest = probe_path_telemetry_lines(obs, path, probe_tip_offset_m=0.2)

    assert nearest == 1
    assert "Probe tip: 0.100, 0.000, 0.000 m" in lines
    assert "Probe x: 1.000, 0.000, 0.000" in lines
    assert "Probe y: 0.000, 1.000, 0.000" in lines
    assert "Probe z: 0.000, 0.000, -1.000" in lines
    assert "Nearest path point: 1/1, distance: 0.0 mm" in lines


def test_obs_from_tcp_pose_rotvec_exposes_tcp_axes():
    obs = obs_from_tcp_pose_rotvec(np.zeros(6))

    np.testing.assert_allclose(obs["tcp_position_base"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["tcp_x_axis_base"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["tcp_y_axis_base"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(obs["tcp_z_axis_base"], [0.0, 0.0, 1.0])
