from __future__ import annotations

import numpy as np

from EXPERIMENT.comparison_teleop.plot_comparison_teleop_figure import (
    MODE_ORDER,
    TrialData,
    plot_process_path_error,
)


def _trial(mode: str) -> TrialData:
    t = np.array([0.0, 2.0, 4.0], dtype=float)
    progress = np.array([0.0, 0.5, 1.0], dtype=float)
    values = np.array([1.0, 3.0, 2.0], dtype=float)
    return TrialData(
        episode=f"synthetic_{mode}",
        participant="gyc",
        pair_id="01",
        mode=mode,
        t=t,
        progress=progress,
        tip=np.zeros((3, 3), dtype=float),
        ref_tip=np.zeros((3, 3), dtype=float),
        ref_normals=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (3, 1)),
        ref_path=np.zeros((3, 3), dtype=float),
        path_error_mm=values,
        normal_offset_mm=np.zeros(3, dtype=float),
        orientation_error_deg=values,
        speed_mm_s=values,
        accel_mm_s2=values,
        force_n=values,
        force_rate_abs_n_s=values,
        summary={
            "path_rmse_mm": 1.0,
            "orientation_rmse_deg": 1.0,
            "accel_p95_mm_s2": 1.0,
            "force_rate_p95_n_s": 1.0,
        },
    )


def test_process_plots_use_elapsed_time_seconds_for_x_axis():
    import matplotlib.pyplot as plt

    modes = {mode: _trial(mode) for mode in MODE_ORDER}
    fig, ax = plt.subplots()

    plot_process_path_error(ax, modes)

    assert ax.get_xlabel() == "Time (s)"
    raw_lines = ax.lines[::2]
    assert all(len(line.get_xdata()) == 3 for line in raw_lines)
    assert all(np.isclose(line.get_alpha(), 0.45) for line in raw_lines)
    assert all(line.get_marker() == "." for line in raw_lines)
    line_x_max = max(float(np.nanmax(line.get_xdata())) for line in raw_lines)
    assert np.isclose(line_x_max, 4.0)
    plt.close(fig)
