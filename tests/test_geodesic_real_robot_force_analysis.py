from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from EXPERIMENT.geodesic_real_robot.plot_case_wrench_progress import plot_wrench_progress
from EXPERIMENT.geodesic_real_robot.plot_pressure_metrics import plot_pressure_metrics
from EXPERIMENT.geodesic_real_robot.plot_rescue_metrics import plot_rescue_metrics
from EXPERIMENT.geodesic_real_robot.plot_tangential_force_metrics import (
    plot_tangential_force_metrics,
)
from EXPERIMENT.geodesic_real_robot.plot_torque_metrics import plot_torque_metrics
from EXPERIMENT.geodesic_real_robot.force_analysis import (
    GROUPS,
    METHOD_ORDER,
    TrialSignals,
    command_progress,
    compute_force_metrics,
    hard_lift_events,
    resample_by_progress,
)


def _signals() -> TrialSignals:
    return TrialSignals(
        episode="synthetic",
        method="geodesic",
        time_s=np.array([0.0, 1.0, 2.0, 3.0]),
        progress=np.array([0.0, 0.3, 0.7, 1.0]),
        force=np.array(
            [
                [0.0, 0.0, -2.0, 0.0, 0.0, 0.1],
                [3.0, 4.0, -3.5, 0.1, 0.2, 0.2],
                [6.0, 8.0, -5.0, 0.2, 0.3, 0.3],
                [0.0, 0.0, -9.0, 0.3, 0.4, 0.4],
            ]
        ),
        force_raw=np.zeros((4, 6)),
        force_gravity=np.zeros((4, 6)),
        force_bias=np.zeros((4, 6)),
        force_valid=np.ones(4, dtype=bool),
        hard_lift_active=np.array([False, True, True, False]),
        hard_lift_reason=np.array(["", "pressure", "pressure", ""], dtype=str),
        force_offset_m=np.array([0.0, 0.001, 0.002, 0.001]),
        command_offset_m=np.array([0.0, 0.001, 0.002, 0.001]),
        delta_offset_m=np.array([0.0, 0.001, 0.001, -0.001]),
        tcp_position_m=np.zeros((4, 3)),
        tcp_rotvec_rad=np.zeros((4, 3)),
        control_pose=np.zeros((4, 6)),
        path_distance_m=np.zeros(4),
        pose_index=np.array([0, 0, 1, 1]),
        pose_count=np.full(4, 2),
        waypoint_index=np.array([1, 2, 1, 2]),
        waypoint_count=np.full(4, 2),
    )


def test_method_order_keeps_geodesic_last_for_consistent_figures():
    assert METHOD_ORDER == ("original", "moving_average", "b_spline", "geodesic")


def test_groups_include_all_eight_real_robot_cases():
    assert tuple(sorted(GROUPS)) == tuple(range(1, 9))
    for group in range(4, 9):
        assert len(GROUPS[group]["episodes"]) == 4
        assert GROUPS[group]["coupling"] == "gel"
        assert GROUPS[group]["label"] == f"Path {group} (coupling gel)"


def test_analysis_sets_a_writable_matplotlib_cache_location():
    env = dict(os.environ)
    env.pop("MPLCONFIGDIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import EXPERIMENT.geodesic_real_robot.force_analysis; "
                "print(os.environ.get('MPLCONFIGDIR', ''))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip().startswith("/tmp/")


def test_command_progress_uses_pose_and_waypoint_indices():
    progress = command_progress(
        pose_index=np.array([0, 0, 1, 1]),
        pose_count=np.array([2, 2, 2, 2]),
        waypoint_index=np.array([1, 2, 1, 2]),
        waypoint_count=np.array([2, 2, 2, 2]),
    )

    np.testing.assert_allclose(progress, [0.25, 0.5, 0.75, 1.0])


def test_force_metrics_include_pressure_tangential_torque_and_rescue_terms():
    metrics = compute_force_metrics(_signals())

    assert metrics["pressure_mean_n"] == 4.875
    assert metrics["pressure_over_4_ratio"] == 0.5
    assert metrics["pressure_over_8_ratio"] == 1.0 / 6.0
    assert metrics["pressure_exposure_4_ns"] == 3.5
    assert metrics["tangential_force_p95_n"] > 5.0
    assert metrics["tangential_force_max_n"] == 10.0
    assert metrics["torque_tangential_max_nm"] == 0.5
    assert metrics["hard_lift_event_count"] == 1
    assert metrics["hard_lift_ratio"] == 2.0 / 3.0
    assert metrics["force_offset_max_outward_mm"] == 2.0
    assert metrics["force_offset_outward_motion_mm"] == 2.0


def test_hard_lift_events_count_transitions_and_recovery():
    events = hard_lift_events(_signals())

    assert len(events) == 1
    assert events[0]["start_index"] == 1
    assert events[0]["end_index"] == 3
    assert events[0]["duration_s"] == 2.0
    assert events[0]["reason"] == "pressure"


def test_resample_by_progress_averages_duplicate_progress_before_interpolation():
    grid, values = resample_by_progress(
        np.array([0.0, 0.5, 0.5, 1.0]),
        np.array([0.0, 1.0, 3.0, 4.0]),
        points=5,
    )

    np.testing.assert_allclose(grid, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(values, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_all_five_plot_modules_write_pdf_and_png(tmp_path):
    signals = _signals()
    trials = {method: signals for method in METHOD_ORDER}
    plotters = (
        plot_wrench_progress,
        plot_pressure_metrics,
        plot_tangential_force_metrics,
        plot_torque_metrics,
        plot_rescue_metrics,
    )

    for index, plotter in enumerate(plotters):
        stem = tmp_path / f"figure_{index}"
        pdf_path, png_path = plotter(trials, stem, group_label="Synthetic")
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        assert png_path.exists()
        assert png_path.stat().st_size > 0


def test_two_row_summary_figure_writes_pdf_svg_and_png(tmp_path):
    from EXPERIMENT.geodesic_real_robot.plot_geodesic_summary_figure import (
        plot_geodesic_summary_figure,
    )

    trials = {method: _signals() for method in METHOD_ORDER}
    cohort = {group: trials for group in range(2, 9)}

    outputs = plot_geodesic_summary_figure(
        trials,
        cohort,
        tmp_path,
        best_group=3,
    )

    assert set(outputs) == {
        "combined",
        "process_tangential_force",
        "process_tangential_torque",
        "box_offset_variation",
        "box_outward_correction",
        "box_tangential_force_p95",
        "box_tangential_torque_p95",
    }
    for paths in outputs.values():
        assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 0


def test_summary_metric_boxplot_has_no_p_value_annotation():
    import matplotlib.pyplot as plt

    from EXPERIMENT.geodesic_real_robot.plot_geodesic_summary_figure import (
        _cohort_metrics,
        _draw_metric_boxplot,
    )

    trials = {method: _signals() for method in METHOD_ORDER}
    metrics = _cohort_metrics({group: trials for group in range(2, 9)})
    figure, axis = plt.subplots()
    _draw_metric_boxplot(
        axis,
        metrics,
        name="test",
        key="force_offset_variation_mm",
        ylabel="$V_o$",
    )

    assert len(axis.texts) == 0
    assert len(axis.patches) == 3
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Moving\naverage",
        "B-spline",
        "Geodesic\n(Ours)",
    ]
    plotted_values = np.concatenate(
        [
            collection.get_offsets()[:, 1]
            for collection in axis.collections
        ]
    )
    lower, upper = axis.get_ylim()
    assert lower > 0.0
    assert lower < np.min(plotted_values)
    assert upper > np.max(plotted_values)
    assert np.isclose(
        0.5 * (lower + upper),
        0.5 * (np.min(plotted_values) + np.max(plotted_values)),
    )
    plt.close(figure)


def test_summary_process_plots_have_no_titles_or_force_threshold():
    import matplotlib.pyplot as plt

    from EXPERIMENT.geodesic_real_robot.plot_geodesic_summary_figure import (
        _draw_tangential_force,
        _draw_tangential_torque,
    )

    trials = {method: _signals() for method in METHOD_ORDER}
    figure, axes = plt.subplots(1, 2)
    _draw_tangential_force(
        axes[0],
        trials,
        best_group=7,
        show_legend=False,
    )
    _draw_tangential_torque(
        axes[1],
        trials,
        show_legend=False,
    )

    assert axes[0].get_title() == ""
    assert axes[1].get_title() == ""
    assert not any(
        line.get_linestyle() == "--"
        and np.allclose(np.asarray(line.get_ydata(), dtype=float), 8.0)
        for line in axes[0].lines
    )
    assert axes[1].get_ylabel() == "$\\tau_t$\n$(\\mathrm{N\\cdot m})$"
    assert axes[1].yaxis.get_label().get_position()[0] == -0.08
    plt.close(figure)


def test_offset_variation_box_uses_total_variation_axis_label():
    from EXPERIMENT.geodesic_real_robot.plot_geodesic_summary_figure import (
        _box_specs,
    )

    assert _box_specs()[0]["ylabel"] == "$\\mathrm{TV}(o)$\n$(\\mathrm{mm})$"
