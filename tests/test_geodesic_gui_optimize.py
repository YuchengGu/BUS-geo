from types import SimpleNamespace

import numpy as np

from breast_path_planning.path_io import PlannedPath, load_planned_path
from visual_guided_collection_gui.path_variants import (
    apply_b_spline_variant,
    apply_moving_average_variant,
    original_path_for_variant,
    path_variant_context,
)
from visual_guided_collection_gui.geodesic_optimize import (
    GUI_GEODESIC_PARAMS,
    GUI_GEODESIC_SURFACE_NORMAL_K_NEIGHBORS,
    optimize_gui_planned_path_geodesic,
)
from visual_guided_collection_gui.planning_session import PlanningSession


def _path(offset: float = 0.0) -> PlannedPath:
    return PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, offset],
                [0.01, 0.0, offset],
                [0.02, 0.0, offset],
            ],
            dtype=float,
        ),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (3, 1)),
        metadata={"planner": "test"},
    )


def test_gui_geodesic_defaults_match_current_demo_parameters():
    assert GUI_GEODESIC_PARAMS.max_iterations == 5000
    assert GUI_GEODESIC_PARAMS.fidelity_weight == 50000000
    assert GUI_GEODESIC_PARAMS.initial_temperature == 1
    assert GUI_GEODESIC_PARAMS.cooling_rate == 0.995
    assert GUI_GEODESIC_PARAMS.perturbation_radius_m == 0.01
    assert GUI_GEODESIC_PARAMS.max_candidate_step_m == 0.0075
    assert GUI_GEODESIC_PARAMS.corner_perturbation_scale == 0.1
    assert GUI_GEODESIC_PARAMS.random_seed == 0
    assert GUI_GEODESIC_PARAMS.energy_record_interval == 10
    assert GUI_GEODESIC_SURFACE_NORMAL_K_NEIGHBORS == 20


def test_optimize_gui_planned_path_geodesic_preserves_point_count_and_marks_metadata():
    source = _path()
    surface = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.01, 0.01, 0.0],
        ],
        dtype=float,
    )

    optimized = optimize_gui_planned_path_geodesic(source, surface)

    assert len(optimized) == len(source)
    assert optimized.metadata["geodesic_resample"] is True
    assert optimized.metadata["geodesic_trigger"] == "gui_optimize_geodesic"


def test_planning_session_replace_planned_path_saves_backup_and_current_path(tmp_path):
    old_path = _path()
    new_path = _path(offset=0.001)
    session = PlanningSession.__new__(PlanningSession)
    session.output_dir = tmp_path
    session.plan_result = SimpleNamespace(planned_path=old_path)

    output = session.replace_planned_path(new_path, backup_name="planned_path_before_geodesic.json")

    assert output == tmp_path / "planned_path.json"
    assert np.allclose(session.planned_path.positions_base, new_path.positions_base)
    assert np.allclose(
        load_planned_path(tmp_path / "planned_path_before_geodesic.json").positions_base,
        old_path.positions_base,
    )
    assert np.allclose(load_planned_path(tmp_path / "planned_path.json").positions_base, new_path.positions_base)


def test_path_variants_use_original_backup_instead_of_current_path(tmp_path):
    original = _path(offset=0.0)
    current = _path(offset=0.01)
    session = PlanningSession.__new__(PlanningSession)
    session.output_dir = tmp_path
    session.plan_result = SimpleNamespace(planned_path=current)
    from breast_path_planning.path_io import save_planned_path

    save_planned_path(original, tmp_path / "planned_path_before_geodesic.json")

    source = original_path_for_variant(session)
    np.testing.assert_allclose(source.positions_base, original.positions_base)


def test_moving_average_variant_marks_metadata_and_preserves_normals():
    source = PlannedPath(
        positions_base=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.02, 0.0],
                [0.02, -0.02, 0.0],
                [0.03, 0.02, 0.0],
                [0.04, 0.0, 0.0],
            ],
            dtype=float,
        ),
        normals_base=np.tile([[0.0, 0.0, 1.0]], (5, 1)),
        metadata={"planner": "test"},
    )

    smoothed = apply_moving_average_variant(source)

    assert smoothed.metadata["path_variant_method"] == "moving_average"
    np.testing.assert_allclose(smoothed.normals_base, source.normals_base)
    np.testing.assert_allclose(smoothed.positions_base[0], source.positions_base[0])
    np.testing.assert_allclose(smoothed.positions_base[-1], source.positions_base[-1])


def test_b_spline_variant_marks_metadata_and_preserves_normals():
    source = _path()

    smoothed = apply_b_spline_variant(source, smoothing_factor=1e-6)

    assert smoothed.metadata["path_variant_method"] == "b_spline"
    np.testing.assert_allclose(smoothed.normals_base, source.normals_base)
    np.testing.assert_allclose(smoothed.positions_base[0], source.positions_base[0])
    np.testing.assert_allclose(smoothed.positions_base[-1], source.positions_base[-1])


def test_path_variant_context_identifies_methods_for_episode_meta():
    original = _path()
    moving_average = apply_moving_average_variant(original)
    b_spline = apply_b_spline_variant(original, smoothing_factor=1e-6)
    geodesic = PlannedPath(
        positions_base=original.positions_base,
        normals_base=original.normals_base,
        metadata={"geodesic_trigger": "gui_optimize_geodesic", "geodesic_energy_final": 1.2},
    )

    assert path_variant_context(original)["path_variant_method"] == "original"
    assert path_variant_context(moving_average)["path_variant_method"] == "moving_average"
    assert path_variant_context(moving_average)["path_variant_moving_average_window"] == 5
    assert path_variant_context(b_spline)["path_variant_method"] == "b_spline"
    assert path_variant_context(b_spline)["path_variant_b_spline_smoothing_factor"] == 1e-6
    assert path_variant_context(geodesic)["path_variant_method"] == "geodesic"
    assert path_variant_context(geodesic)["path_variant_geodesic_energy_final"] == 1.2
