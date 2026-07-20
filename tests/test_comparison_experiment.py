import numpy as np
import pytest

from breast_path_planning.path_io import PlannedPath
from visual_guided_collection_gui.comparison_experiment import (
    ComparisonExperiment,
    generate_trial_pair,
)


def _path() -> PlannedPath:
    positions = np.column_stack(
        [
            np.linspace(0.0, 0.12, 13),
            np.zeros(13),
            np.zeros(13),
        ]
    )
    return PlannedPath(
        positions_base=positions,
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (len(positions), 1)),
    )


def test_generate_trial_pair_has_exact_five_centimeter_arclength():
    pair = generate_trial_pair(
        _path(),
        pair_number=1,
        length_m=0.05,
        rng=np.random.default_rng(3),
    )

    assert pair.pair_id == "pair_001"
    assert pair.length_m == pytest.approx(0.05)
    assert pair.end_arclength_m - pair.start_arclength_m == pytest.approx(0.05)
    np.testing.assert_allclose(
        pair.end_position_base,
        [pair.end_arclength_m, 0.0, 0.0],
    )


def test_generate_trial_pair_accepts_manual_pair_id():
    pair = generate_trial_pair(
        _path(),
        pair_number=1,
        pair_id="scan_A",
        length_m=0.05,
        rng=np.random.default_rng(3),
    )

    assert pair.pair_id == "scan_A"
    assert pair.context()["pair_id"] == "scan_A"


def test_pair_persists_when_participant_changes_and_modes_can_run_in_any_order():
    experiment = ComparisonExperiment()
    pair = generate_trial_pair(
        _path(),
        pair_number=1,
        length_m=0.05,
        rng=np.random.default_rng(0),
    )
    experiment.set_pair(pair)
    experiment.confirm_participant("anonymous_01")

    first = experiment.begin_trial("darboux")
    experiment.start_scan()
    experiment.finish_trial("manual")
    second = experiment.begin_trial("full_joint")
    experiment.start_scan()
    experiment.finish_trial("reached")

    assert first.sequence_index == 1
    assert second.sequence_index == 2
    assert experiment.completed_modes == {"darboux", "full_joint"}

    experiment.confirm_participant("anonymous_02")

    assert experiment.pair is pair
    assert experiment.completed_modes == set()
    assert experiment.participant_id == "anonymous_02"
    assert experiment.sequence_index == 0


def test_pair_can_be_renamed_before_trial_starts():
    experiment = ComparisonExperiment()
    pair = generate_trial_pair(
        _path(),
        pair_number=1,
        length_m=0.05,
        rng=np.random.default_rng(0),
    )
    experiment.set_pair(pair)

    experiment.rename_pair("scan_A")

    assert experiment.pair is not None
    assert experiment.pair.pair_id == "scan_A"
    assert experiment.pair.start_index == pair.start_index
    np.testing.assert_allclose(
        experiment.pair.end_position_base,
        pair.end_position_base,
    )


def test_endpoint_requires_position_and_forward_progress():
    experiment = ComparisonExperiment(
        endpoint_radius_m=0.005,
        progress_tolerance_m=0.003,
    )
    pair = generate_trial_pair(
        _path(),
        pair_number=1,
        length_m=0.05,
        rng=np.random.default_rng(1),
    )
    experiment.set_pair(pair)

    near_endpoint = pair.end_position_base + np.array([0.002, 0.0, 0.0])

    assert not experiment.endpoint_reached(
        near_endpoint,
        nearest_arclength_m=pair.end_arclength_m - 0.01,
    )
    assert not experiment.endpoint_reached(
        pair.end_position_base + np.array([0.008, 0.0, 0.0]),
        nearest_arclength_m=pair.end_arclength_m,
    )
    assert experiment.endpoint_reached(
        near_endpoint,
        nearest_arclength_m=pair.end_arclength_m - 0.002,
    )
    assert experiment.endpoint_reached(
        pair.end_position_base + 0.02 * pair.end_normal_base,
        nearest_arclength_m=pair.end_arclength_m,
        reference_height_m=0.02,
    )


def test_endpoint_uses_tangent_plane_distance_not_normal_height_error():
    experiment = ComparisonExperiment(
        endpoint_radius_m=0.005,
        progress_tolerance_m=0.003,
    )
    pair = generate_trial_pair(
        _path(),
        pair_number=1,
        length_m=0.05,
        rng=np.random.default_rng(1),
    )
    experiment.set_pair(pair)

    assert experiment.endpoint_reached(
        pair.end_position_base
        + 0.002 * np.array([1.0, 0.0, 0.0])
        + 0.03 * pair.end_normal_base,
        nearest_arclength_m=pair.end_arclength_m,
        reference_height_m=0.02,
    )
    assert not experiment.endpoint_reached(
        pair.end_position_base
        + 0.008 * np.array([1.0, 0.0, 0.0])
        + 0.02 * pair.end_normal_base,
        nearest_arclength_m=pair.end_arclength_m,
        reference_height_m=0.02,
    )


def test_trial_context_contains_pair_participant_mode_and_completion():
    experiment = ComparisonExperiment()
    experiment.set_pair(
        generate_trial_pair(
            _path(),
            pair_number=4,
            length_m=0.05,
            rng=np.random.default_rng(2),
        )
    )
    experiment.confirm_participant("anonymous")
    experiment.begin_trial("full_joint")
    experiment.set_approach_metrics(
        duration_s=4.2,
        position_error_m=0.006,
        orientation_error_rad=0.12,
    )
    experiment.start_scan()

    context = experiment.trial_context(action_mode="joint_position")

    assert context["operation_mode"] == "comparison"
    assert context["participant_id"] == "anonymous"
    assert context["pair_id"] == "pair_004"
    assert context["teleop_mode"] == "full_joint"
    assert context["trial_sequence_index"] == 1
    assert context["trial_phase"] == "scan"
    assert context["action_mode"] == "joint_position"
    assert context["approach_duration_s"] == pytest.approx(4.2)
    assert context["start_position_error_m"] == pytest.approx(0.006)


def test_finish_participant_keeps_pair_and_clears_participant_state():
    experiment = ComparisonExperiment()
    pair = generate_trial_pair(
        _path(),
        pair_number=2,
        length_m=0.05,
        rng=np.random.default_rng(0),
    )
    experiment.set_pair(pair)
    experiment.confirm_participant("anonymous")

    experiment.finish_participant()

    assert experiment.pair is pair
    assert experiment.participant_id is None
    assert experiment.completed_modes == set()


def test_abort_trial_does_not_mark_mode_completed():
    experiment = ComparisonExperiment()
    experiment.set_pair(
        generate_trial_pair(
            _path(),
            pair_number=3,
            length_m=0.05,
            rng=np.random.default_rng(0),
        )
    )
    experiment.confirm_participant("anonymous")
    experiment.begin_trial("darboux")

    experiment.abort_trial()

    assert experiment.active_trial is None
    assert experiment.completed_modes == set()


def test_completed_mode_cannot_be_started_twice_for_same_participant():
    experiment = ComparisonExperiment()
    experiment.set_pair(
        generate_trial_pair(
            _path(),
            pair_number=3,
            length_m=0.05,
            rng=np.random.default_rng(0),
        )
    )
    experiment.confirm_participant("anonymous")
    experiment.begin_trial("full_joint")
    experiment.start_scan()
    experiment.finish_trial("manual")

    with pytest.raises(RuntimeError, match="already completed"):
        experiment.begin_trial("full_joint")
