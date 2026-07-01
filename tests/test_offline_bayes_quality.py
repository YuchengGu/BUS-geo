import numpy as np

from scripts.bo_ask_tell_demo import resolve_total_trials
from scripts.view_ultrasound_frame import render_frame
from visual_guided_collection_gui.offline_bayes import LocalBayesOptimizer, LocalBOConfig
from visual_guided_collection_gui.ultrasound_quality import (
    QualityNormalization,
    UltrasoundQualityScorer,
    extract_quality_features,
)


def test_ultrasound_quality_scores_single_frame_with_fixed_normalization():
    frame = np.zeros((32, 32), dtype=np.uint8)
    frame[6:18, 8:24] = 160
    frame[18:28, 10:22] = 80

    scorer = UltrasoundQualityScorer(
        normalization=QualityNormalization(
            d_min=0.0,
            d_max=255.0,
            e_min=0.0,
            e_max=8.0,
            c_min=0.0,
            c_max=128.0,
            c_target=64.0,
            s_min=0.0,
            s_max=2.0,
        )
    )

    result = scorer.score_frame(frame)

    assert result.features.D > 0.0
    assert result.features.E > 0.0
    assert result.features.C > 0.0
    assert result.features.S >= 0.0
    assert 0.0 <= result.Q <= 1.0
    assert result.weighted_vector.shape == (4,)


def test_extract_quality_features_accepts_rgb_ultrasound_frame():
    gray = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    features = extract_quality_features(rgb)

    assert features.D > 0.0
    assert features.E > 0.0
    assert features.C > 0.0
    assert np.isfinite(features.S)


def test_ultrasound_quality_can_score_downsampled_large_frame():
    frame = np.zeros((240, 320), dtype=np.uint8)
    frame[30:130, 40:280] = 180
    frame[130:220, 80:240] = 70

    scorer = UltrasoundQualityScorer(max_size=96, confidence_method="fast")
    result = scorer.score_frame(frame)

    assert 0.0 <= result.Q <= 1.0
    assert result.features.D > 0.0


def test_speckle_can_use_full_resolution_when_other_features_are_downsampled():
    frame = np.zeros((96, 128), dtype=np.uint8)
    checker = ((np.indices((96, 128)).sum(axis=0) % 2) * 80).astype(np.uint8)
    frame[20:76, 24:104] = 80 + checker[20:76, 24:104]

    downsampled = UltrasoundQualityScorer(max_size=32, speckle_max_size=32).score_frame(frame)
    full_speckle = UltrasoundQualityScorer(max_size=32, speckle_max_size=None).score_frame(frame)

    assert downsampled.features.S != full_speckle.features.S
    assert full_speckle.features.D == downsampled.features.D


def test_confidence_can_be_downsampled_while_entropy_contrast_speckle_use_full_resolution():
    frame = np.zeros((96, 128), dtype=np.uint8)
    checker = ((np.indices((96, 128)).sum(axis=0) % 2) * 80).astype(np.uint8)
    frame[20:76, 24:104] = 80 + checker[20:76, 24:104]

    full = UltrasoundQualityScorer(
        max_size=None,
        speckle_max_size=None,
        confidence_max_size=None,
        confidence_method="fast",
    ).score_frame(frame)
    confidence_downsampled = UltrasoundQualityScorer(
        max_size=None,
        speckle_max_size=None,
        confidence_max_size=32,
        confidence_method="fast",
    ).score_frame(frame)

    assert confidence_downsampled.features.D != full.features.D
    assert confidence_downsampled.features.E == full.features.E
    assert confidence_downsampled.features.C == full.features.C
    assert confidence_downsampled.features.S == full.features.S


def test_ultrasound_quality_supports_random_walker_confidence_method():
    frame = np.zeros((48, 64), dtype=np.uint8)
    frame[8:28, 10:54] = 180

    scorer = UltrasoundQualityScorer(max_size=64, confidence_method="random_walker")
    result = scorer.score_frame(frame)

    assert 0.0 <= result.Q <= 1.0
    assert result.features.D > 0.0


def test_blank_ultrasound_frame_gets_low_quality_score():
    frame = np.zeros((128, 160), dtype=np.uint8)

    scorer = UltrasoundQualityScorer(max_size=128, confidence_method="fast")
    result = scorer.score_frame(frame)

    assert result.features.D == 0.0
    assert result.Q < 0.35


def test_view_ultrasound_frame_can_save_png(tmp_path):
    frame = np.zeros((24, 32), dtype=np.uint8)
    frame[4:20, 8:24] = 120
    output = tmp_path / "frame.png"

    render_frame(frame, title="test frame", save_path=output, show=False)

    assert output.exists()
    assert output.stat().st_size > 0


def test_local_bayes_optimizer_asks_three_initial_points_before_ei():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-0.005, 0.005), (-5.0, 5.0)],
            n_initial=3,
            max_trials=12,
            random_state=7,
        )
    )

    initial = [optimizer.ask() for _ in range(3)]

    assert len(initial) == 3
    assert all(point.shape == (2,) for point in initial)
    assert all(-0.005 <= point[0] <= 0.005 and -5.0 <= point[1] <= 5.0 for point in initial)
    assert len({tuple(point.round(8)) for point in initial}) == 3


def test_bo_cli_budget_counts_initial_plus_ei_trials():
    assert resolve_total_trials(n_initial=3, n_ei=12) == 15


def test_local_bayes_optimizer_uses_ei_after_initial_observations():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=3,
            max_trials=12,
            random_state=4,
            candidate_count=512,
        )
    )

    for x, y in [([-1.0], 1.0), ([0.0], 0.0), ([1.0], 1.0)]:
        optimizer.tell(np.array(x, dtype=float), y)

    suggestion = optimizer.ask()

    assert suggestion.shape == (1,)
    assert -1.0 <= suggestion[0] <= 1.0
    assert not any(np.allclose(suggestion, obs, atol=1e-8) for obs in optimizer.x_observed)


def test_local_bayes_optimizer_stops_after_flat_recent_best_values():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=3,
            max_trials=15,
            convergence_window=3,
            min_improvement=1e-3,
            random_state=1,
        )
    )

    for idx, y in enumerate([1.0, 0.5, 0.7, 0.4998, 0.4997, 0.4997]):
        optimizer.tell(np.array([idx / 10.0], dtype=float), y)

    assert optimizer.should_stop()


def test_local_bayes_optimizer_does_not_stop_if_recent_best_change_is_large():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=3,
            max_trials=15,
            convergence_window=3,
            min_improvement=1e-3,
            random_state=1,
        )
    )

    for idx, y in enumerate([1.0, 0.5, 0.7, 0.49, 0.4899, 0.48]):
        optimizer.tell(np.array([idx / 10.0], dtype=float), y)

    assert not optimizer.should_stop()


def test_local_bayes_optimizer_does_not_converge_during_initial_trials():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=5,
            max_trials=8,
            convergence_window=2,
            min_improvement=1e-3,
            random_state=1,
        )
    )

    for idx, y in enumerate([1.0, 1.0, 1.0, 1.0]):
        optimizer.tell(np.array([idx / 10.0], dtype=float), y)

    assert not optimizer.should_stop()


def test_local_bayes_optimizer_requires_full_ei_window_before_early_stop():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=3,
            max_trials=15,
            convergence_window=3,
            min_improvement=1e-3,
            random_state=1,
        )
    )

    for idx, y in enumerate([3.0, 2.0, 4.0, 2.1, 2.32]):
        optimizer.tell(np.array([idx / 10.0], dtype=float), y)

    assert not optimizer.should_stop()


def test_local_bayes_optimizer_boosts_exploration_after_three_stagnant_ei_trials():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=3,
            max_trials=15,
            stagnation_window=3,
            convergence_window=7,
            min_improvement=1e-3,
            random_state=1,
        )
    )

    for idx, y in enumerate([3.0, 2.0, 4.0, 2.1, 2.32]):
        optimizer.tell(np.array([idx / 10.0], dtype=float), y)

    assert not optimizer.should_boost_exploration()

    optimizer.tell(np.array([0.5], dtype=float), 2.2)

    assert optimizer.should_boost_exploration()
    assert not optimizer.should_stop()


def test_local_bayes_optimizer_does_not_boost_after_recent_improvement():
    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=[(-1.0, 1.0)],
            n_initial=3,
            max_trials=15,
            stagnation_window=3,
            convergence_window=7,
            min_improvement=1e-3,
            random_state=1,
        )
    )

    for idx, y in enumerate([3.0, 2.0, 4.0, 2.1, 2.32, 1.9]):
        optimizer.tell(np.array([idx / 10.0], dtype=float), y)

    assert not optimizer.should_boost_exploration()
