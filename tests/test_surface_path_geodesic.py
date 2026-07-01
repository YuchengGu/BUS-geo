import numpy as np

from breast_path_planning.geodesic_path import (
    GeodesicPathParams,
    _sample_bounded_perturbation,
    discrete_geodesic_curvatures,
    geodesic_path_energy,
    resample_path_with_surface_geodesics,
)
from breast_path_planning.path_io import PlannedPath


def test_discrete_geodesic_curvature_is_zero_for_straight_planar_path():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(positions), 1))

    kg = discrete_geodesic_curvatures(positions, normals)

    np.testing.assert_allclose(kg, np.zeros(2), atol=1e-12)


def test_geodesic_path_energy_uses_discrete_curvature_and_fidelity_terms():
    initial = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    candidate = initial.copy()
    candidate[1] = [1.0, 0.0, 0.0]
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(initial), 1))

    initial_energy = geodesic_path_energy(initial, normals, initial, fidelity_weight=0.01)
    candidate_energy = geodesic_path_energy(candidate, normals, initial, fidelity_weight=0.01)

    assert candidate_energy.total < initial_energy.total
    assert candidate_energy.curvature < initial_energy.curvature
    assert candidate_energy.fidelity > 0.0


def test_resample_path_with_surface_geodesics_uses_simulated_annealing_energy_optimization():
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    initial_path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
        metadata={"planner": "energy_test"},
    )

    geodesic = resample_path_with_surface_geodesics(
        initial_path,
        surface_points,
        surface_normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (len(surface_points), 1)),
        params=GeodesicPathParams(
            max_iterations=600,
            fidelity_weight=0.01,
            initial_temperature=1.0,
            cooling_rate=0.995,
            perturbation_radius_m=1.5,
            random_seed=7,
        ),
    )

    np.testing.assert_allclose(geodesic.positions_base[0], initial_path.positions_base[0])
    np.testing.assert_allclose(geodesic.positions_base[-1], initial_path.positions_base[-1])
    np.testing.assert_allclose(geodesic.positions_base[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(geodesic.normals_base, np.tile([0.0, 0.0, 1.0], (3, 1)))
    assert geodesic.metadata["geodesic_resample"] is True
    assert geodesic.metadata["planner"] == "geodesic_energy_sa_v1"
    assert geodesic.metadata["geodesic_source_path_points"] == 3
    assert geodesic.metadata["geodesic_energy_final"] < geodesic.metadata["geodesic_energy_initial"]


def test_resample_path_with_surface_geodesics_reports_progress_records():
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    initial_path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
    )
    records = []

    resample_path_with_surface_geodesics(
        initial_path,
        surface_points,
        surface_normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (len(surface_points), 1)),
        params=GeodesicPathParams(
            max_iterations=20,
            energy_record_interval=5,
            fidelity_weight=0.01,
            perturbation_radius_m=1.0,
            random_seed=4,
        ),
        progress_callback=records.append,
    )

    assert records[0]["iteration"] == 0
    assert records[-1]["iteration"] == 20
    assert {"total", "curvature", "fidelity", "temperature", "accepted_moves"} <= set(records[-1])


def test_resample_path_with_surface_geodesics_reports_path_snapshots_as_copies():
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    initial_path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
    )
    snapshots = []

    resample_path_with_surface_geodesics(
        initial_path,
        surface_points,
        surface_normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (len(surface_points), 1)),
        params=GeodesicPathParams(
            max_iterations=20,
            energy_record_interval=5,
            fidelity_weight=0.01,
            perturbation_radius_m=1.0,
            random_seed=4,
        ),
        path_snapshot_callback=lambda record, positions: snapshots.append((dict(record), positions)),
    )

    assert snapshots[0][0]["iteration"] == 0
    assert snapshots[-1][0]["iteration"] == 20
    snapshots[0][1][1] = [99.0, 99.0, 99.0]
    assert not np.allclose(snapshots[-1][1][1], [99.0, 99.0, 99.0])


def test_bounded_perturbation_stays_inside_configured_radius():
    rng = np.random.default_rng(2)
    radius = 0.008

    samples = np.asarray([_sample_bounded_perturbation(rng, radius) for _ in range(200)])

    assert np.max(np.linalg.norm(samples, axis=1)) <= radius + 1e-12


def test_resample_path_rejects_projected_candidates_that_jump_too_far():
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.25, 0.0, 0.0],
        ],
        dtype=float,
    )
    initial_path = PlannedPath(
        positions_base=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
    )

    geodesic = resample_path_with_surface_geodesics(
        initial_path,
        surface_points,
        surface_normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (len(surface_points), 1)),
        params=GeodesicPathParams(
            max_iterations=80,
            perturbation_radius_m=1.0,
            max_candidate_step_m=0.01,
            random_seed=5,
        ),
    )

    np.testing.assert_allclose(geodesic.positions_base[1], [1.0, 0.0, 0.0])
    assert geodesic.metadata["geodesic_rejected_large_steps"] > 0
