from EXPERIMENT import demo_geodesic_annealing
import numpy as np


def test_demo_geodesic_annealing_requires_configured_input_dir():
    try:
        demo_geodesic_annealing.resolve_input_dir(configured_dir=None)
    except ValueError as exc:
        assert "Set INPUT_DIR" in str(exc)
    else:
        raise AssertionError("INPUT_DIR=None should be rejected")


def test_demo_geodesic_annealing_accepts_configured_dir(tmp_path):
    configured = tmp_path / "live_gui_configured"
    configured.mkdir()

    resolved = demo_geodesic_annealing.resolve_input_dir(configured_dir=str(configured))

    assert resolved == configured


def test_demo_geodesic_annealing_resolves_batch_input_dirs(tmp_path):
    valid = tmp_path / "live_gui_valid"
    valid.mkdir()
    (valid / "segmented_breast.ply").write_text("ply\n", encoding="utf-8")
    (valid / "planned_path_before_geodesic.json").write_text("{}", encoding="utf-8")
    invalid = tmp_path / "live_gui_invalid"
    invalid.mkdir()

    resolved = demo_geodesic_annealing.resolve_input_dirs(
        configured_dir=None,
        configured_dirs=[],
        input_glob=str(tmp_path / "live_gui_*"),
    )

    assert resolved == [valid]


def test_demo_geodesic_annealing_catmull_rom_path_preserves_endpoints_and_densifies():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    smoothed = demo_geodesic_annealing._catmull_rom_path(points, samples_per_segment=5)

    assert len(smoothed) > len(points)
    np.testing.assert_allclose(smoothed[0], points[0])
    np.testing.assert_allclose(smoothed[-1], points[-1])


def test_demo_geodesic_annealing_catmull_rom_returns_short_paths_unchanged():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

    smoothed = demo_geodesic_annealing._catmull_rom_path(points, samples_per_segment=5)

    np.testing.assert_allclose(smoothed, points)


def test_demo_geodesic_annealing_open3d_view_uses_cloud_center_lookat():
    cloud = demo_geodesic_annealing.PointCloud(
        points_base=np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], dtype=float)
    )

    view = demo_geodesic_annealing._open3d_view_kwargs(cloud, preset="front")

    np.testing.assert_allclose(view["lookat"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(view["front"], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(view["up"], [0.0, 0.0, 1.0])


def test_demo_geodesic_annealing_custom_open3d_view_requires_vectors():
    cloud = demo_geodesic_annealing.PointCloud(points_base=np.zeros((1, 3), dtype=float))

    try:
        demo_geodesic_annealing._open3d_view_kwargs(cloud, preset="custom")
    except ValueError as exc:
        assert "OPEN3D_VIEW_FRONT" in str(exc)
    else:
        raise AssertionError("custom view without front/up should fail")


def test_demo_geodesic_annealing_capture_paths_are_under_output_dir(tmp_path):
    png, pdf = demo_geodesic_annealing._open3d_capture_paths(tmp_path)

    assert png == tmp_path / "path_evolution_view.png"
    assert pdf == tmp_path / "path_evolution_view.pdf"
