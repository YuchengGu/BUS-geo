from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_comparison_teleop"),
)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from scipy.spatial.transform import Rotation

from breast_path_planning.path_io import load_planned_path
from breast_path_planning.pointcloud_from_d405 import load_point_cloud_ply
from breast_path_planning.surface_processing import estimate_normals


DATA_ROOT = Path("/home/ubuntu22/bc_data/gello")
OUTPUT_DIR = REPO_ROOT / "EXPERIMENT/comparison_teleop/results"
PLANNING_RESULTS_DIR = REPO_ROOT / "breast_path_planning/results"
PARTICIPANT_ORDER = ("gyc", "zy", "wk")
PARTICIPANT_LABELS = {
    "gyc": "Participant 1",
    "zy": "Participant 2",
    "wk": "Participant 3",
}
MODE_ORDER = ("full_joint", "darboux")
MODE_LABELS = {"full_joint": "Full-joint", "darboux": "Darboux"}
MODE_COLORS = {"full_joint": "#0B6FFF", "darboux": "#F04A4A"}
MODE_DASH_COLORS = {"full_joint": "#69A7FF", "darboux": "#F39A8F"}
PARTICIPANT_MARKERS = {"gyc": "o", "zy": "s", "wk": "^"}
DISPLAY_EXAMPLE_EXCLUDE = {("gyc", "gyc-04")}
N_PROCESS_POINTS = 501
AXIS_COLOR = "#222222"
SURFACE_MATCH_THRESHOLD_M = 0.012
SURFACE_BBOX_MARGIN_M = 0.085
POISSON_DEPTH = 9
POISSON_DENSITY_QUANTILE = 0.03
NORMAL_K_NEIGHBORS = 20
ROW1_INTERACTIVE_CAPTURE = False
ROW1_CAPTURE_WIDTH = 1800
ROW1_CAPTURE_HEIGHT = 1300
ROW1_VIEW_FRONT = [-0.45, -0.45, -0.77]
ROW1_VIEW_UP = [-0.35, -0.35, 0.87]
ROW1_VIEW_ZOOM = 0.78
PATH_TUBE_RADIUS_M = 0.0011
POINT_SPHERE_RADIUS_M = 0.004
ROW1_PATH_VISUAL_LIFT_M = 0.004
PROCESS_SMOOTHING_WINDOW_S = 0.20
PROCESS_TIME_LABEL = "Time (s)"
RAW_PROCESS_ALPHA = 0.45
RAW_PROCESS_LINEWIDTH = 0.85
RAW_PROCESS_MARKER = "."
RAW_PROCESS_MARKERSIZE = 2.2
SMOOTH_PROCESS_ALPHA = 0.98
SMOOTH_PROCESS_LINEWIDTH = 1.45


@dataclass(frozen=True)
class TrialData:
    episode: str
    participant: str
    pair_id: str
    mode: str
    t: np.ndarray
    progress: np.ndarray
    tip: np.ndarray
    ref_tip: np.ndarray
    ref_normals: np.ndarray
    ref_path: np.ndarray
    path_error_mm: np.ndarray
    normal_offset_mm: np.ndarray
    orientation_error_deg: np.ndarray
    speed_mm_s: np.ndarray
    accel_mm_s2: np.ndarray
    force_n: np.ndarray
    force_rate_abs_n_s: np.ndarray
    summary: dict[str, float | str]


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = [
        trial
        for trial in load_all_trials(DATA_ROOT)
        if trial.participant in PARTICIPANT_ORDER
    ]
    paired = paired_trials(trials)
    if not paired:
        raise RuntimeError(f"No complete comparison pairs found under {DATA_ROOT}")

    summaries = [trial.summary for trial in trials]
    write_summary_csv(summaries, output_dir / "comparison_trial_metrics.csv")
    examples = select_trajectory_examples(paired, count=4)
    process_key = select_process_example(paired)
    row1_images = prepare_row1_images(paired, examples, output_dir, capture=bool(args.capture_row1_views))
    fig = plot_figure(paired, examples, process_key, row1_images=row1_images)
    save_figure(fig, output_dir / "comparison_teleop_3x4")
    plt.close(fig)
    save_individual_panel_figures(paired, examples, process_key, row1_images, output_dir)
    print(f"Loaded {len(trials)} trials, {len(paired)} complete paired comparisons.")
    print(f"Process example: {process_key[0]} {process_key[1]}")
    print(f"Saved outputs under: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--capture-row1-views",
        action="store_true",
        help="Open one Open3D window per row-1 trajectory panel; adjust view and press Enter to save.",
    )
    return parser.parse_args()


def load_all_trials(root: Path) -> list[TrialData]:
    out: list[TrialData] = []
    for episode_dir in sorted(root.glob("comparison_*")):
        if not episode_dir.is_dir():
            continue
        summary_path = episode_dir / "comparison_trial_summary.json"
        if not summary_path.exists():
            continue
        try:
            out.append(load_trial(episode_dir))
        except Exception as exc:
            print(f"Skipping {episode_dir.name}: {type(exc).__name__}: {exc}")
    return out


def load_trial(episode_dir: Path) -> TrialData:
    trial_summary = json.loads((episode_dir / "comparison_trial_summary.json").read_text(encoding="utf-8"))
    pkl_paths = sorted(episode_dir.glob("*.pkl"))
    if not pkl_paths:
        raise RuntimeError("episode has no pkl frames")

    rows: list[dict[str, Any]] = []
    reference_path = None
    reference_rotvecs = None
    reference_normals = None
    reference_indices = None
    segment_meta = None
    for pkl_path in pkl_paths:
        with pkl_path.open("rb") as handle:
            frame = pickle.load(handle)
        meta = dict(frame.get("meta", {}) or {})
        if meta.get("operation_mode") != "comparison":
            continue
        if meta.get("trial_phase") != "scan":
            continue
        tip = np.asarray(frame.get("probe_tip_position_base"), dtype=float).reshape(-1)
        ee = np.asarray(frame.get("ee_pos_rotvec"), dtype=float).reshape(-1)
        if tip.size != 3 or ee.size < 6:
            continue
        if reference_path is None:
            reference_path = np.asarray(frame["path_target_positions_base"], dtype=float)
            reference_rotvecs = np.asarray(frame["path_reference_tcp_rotvecs_base"], dtype=float)
            reference_normals = np.asarray(frame["path_normals_base"], dtype=float)
            reference_indices = np.asarray(frame["path_indices"], dtype=int)
            segment_meta = meta
        rows.append(
            {
                "t": sample_time_s(meta, len(rows)),
                "tip": tip,
                "rotvec": ee[3:6],
                "force_n": float(frame.get("force_pressure_n", np.nan)),
            }
        )
    if (
        len(rows) < 5
        or reference_path is None
        or reference_rotvecs is None
        or reference_normals is None
        or reference_indices is None
        or segment_meta is None
    ):
        raise RuntimeError("not enough scan frames or missing reference path")

    rows = sorted(rows, key=lambda row: row["t"])
    t = np.asarray([row["t"] for row in rows], dtype=float)
    t = t - t[0]
    t = make_strictly_increasing(t)
    tip = np.vstack([row["tip"] for row in rows])
    actual_rotvecs = np.vstack([row["rotvec"] for row in rows])
    force_n = np.asarray([row["force_n"] for row in rows], dtype=float)

    segment = build_segment(reference_path, reference_rotvecs, reference_normals, reference_indices, segment_meta)
    progress, ref_tip, ref_rotvec, ref_normals = project_to_segment(tip, segment)
    residual = tip - ref_tip
    normal_component = np.sum(residual * ref_normals, axis=1)
    tangent_residual = residual - normal_component[:, None] * ref_normals
    path_error_mm = 1000.0 * np.linalg.norm(tangent_residual, axis=1)
    normal_offset_mm = 1000.0 * normal_component
    orientation_error_deg = orientation_errors_deg(actual_rotvecs, ref_rotvec)
    speed_mm_s, accel_mm_s2 = kinematics(t, tip)
    force_rate_abs_n_s = absolute_derivative(t, force_n)

    participant = str(trial_summary.get("participant_id") or segment_meta.get("participant_id"))
    pair_id = str(trial_summary.get("pair_id") or segment_meta.get("pair_id"))
    mode = str(trial_summary.get("teleop_mode") or segment_meta.get("teleop_mode"))
    summary = {
        "episode": episode_dir.name,
        "participant": participant,
        "pair_id": pair_id,
        "mode": mode,
        "path_rmse_mm": rms(path_error_mm),
        "normal_offset_median_mm": finite_percentile(normal_offset_mm, 50),
        "normal_offset_rmse_mm": rms(normal_offset_mm),
        "orientation_rmse_deg": rms(orientation_error_deg),
        "speed_median_mm_s": finite_percentile(speed_mm_s, 50),
        "speed_p95_mm_s": finite_percentile(speed_mm_s, 95),
        "accel_p95_mm_s2": finite_percentile(accel_mm_s2, 95),
        "force_median_n": finite_percentile(force_n, 50),
        "force_p95_n": finite_percentile(force_n, 95),
        "force_rate_p95_n_s": finite_percentile(force_rate_abs_n_s, 95),
        "duration_s": float(t[-1] - t[0]),
        "end_reason": str(trial_summary.get("trial_end_reason", "")),
    }
    return TrialData(
        episode=episode_dir.name,
        participant=participant,
        pair_id=pair_id,
        mode=mode,
        t=t,
        progress=progress,
        tip=tip,
        ref_tip=ref_tip,
        ref_normals=ref_normals,
        ref_path=segment["positions"],
        path_error_mm=path_error_mm,
        normal_offset_mm=normal_offset_mm,
        orientation_error_deg=orientation_error_deg,
        speed_mm_s=speed_mm_s,
        accel_mm_s2=accel_mm_s2,
        force_n=force_n,
        force_rate_abs_n_s=force_rate_abs_n_s,
        summary=summary,
    )


def sample_time_s(meta: dict[str, Any], fallback_index: int) -> float:
    if meta.get("sample_mono_ns") is not None:
        return float(meta["sample_mono_ns"]) * 1e-9
    timing = meta.get("timing")
    if isinstance(timing, dict) and timing.get("wall_time_s") is not None:
        return float(timing["wall_time_s"])
    return float(fallback_index) / 50.0


def make_strictly_increasing(t: np.ndarray) -> np.ndarray:
    out = np.asarray(t, dtype=float).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-3
    return out


def build_segment(
    path: np.ndarray,
    rotvecs: np.ndarray,
    normals: np.ndarray,
    path_indices: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    start = int(meta["segment_start_index"])
    end = int(meta["segment_end_index"])
    index_values = np.asarray(path_indices, dtype=int).reshape(-1)
    mask = (index_values >= start) & (index_values <= end)
    if not np.any(mask):
        mask = np.ones(len(path), dtype=bool)
    segment_positions = np.asarray(path[mask], dtype=float).copy()
    segment_rotvecs = np.asarray(rotvecs[mask], dtype=float).copy()
    segment_normals = normalize_vectors(np.asarray(normals[mask], dtype=float).copy())
    end_position = np.asarray(meta["segment_end_position_base"], dtype=float)
    if np.linalg.norm(segment_positions[-1] - end_position) > 1e-9:
        segment_positions = np.vstack([segment_positions, end_position])
        segment_rotvecs = np.vstack([segment_rotvecs, segment_rotvecs[-1]])
        end_normal = np.asarray(meta.get("segment_end_normal_base", segment_normals[-1]), dtype=float)
        segment_normals = np.vstack([segment_normals, normalize_vectors(end_normal.reshape(1, 3))[0]])
    else:
        segment_positions[-1] = end_position
    arclength = path_arclength(segment_positions)
    length = max(float(arclength[-1]), 1e-9)
    return {
        "positions": segment_positions,
        "rotvecs": segment_rotvecs,
        "normals": segment_normals,
        "arclength": arclength,
        "length": np.asarray(length, dtype=float),
    }


def project_to_segment(tips: np.ndarray, segment: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = segment["positions"]
    rotvecs = segment["rotvecs"]
    normals = segment["normals"]
    arclength = segment["arclength"]
    length = float(segment["length"])
    ref_positions = np.zeros_like(tips)
    ref_rotvecs = np.zeros((len(tips), 3), dtype=float)
    ref_normals = np.zeros((len(tips), 3), dtype=float)
    progress = np.zeros(len(tips), dtype=float)
    for i, point in enumerate(tips):
        best_dist = float("inf")
        best_ref = positions[0]
        best_s = 0.0
        best_rotvec = rotvecs[0]
        best_normal = normals[0]
        for j in range(len(positions) - 1):
            a = positions[j]
            b = positions[j + 1]
            edge = b - a
            denom = float(np.dot(edge, edge))
            u = 0.0 if denom <= 1e-12 else float(np.dot(point - a, edge) / denom)
            u = float(np.clip(u, 0.0, 1.0))
            ref = a + u * edge
            dist = float(np.linalg.norm(point - ref))
            if dist < best_dist:
                best_dist = dist
                best_ref = ref
                best_s = float(arclength[j] + u * (arclength[j + 1] - arclength[j]))
                best_rotvec = rotvecs[j] if u < 0.5 else rotvecs[j + 1]
                normal = (1.0 - u) * normals[j] + u * normals[j + 1]
                best_normal = normalize_vectors(normal.reshape(1, 3))[0]
        ref_positions[i] = best_ref
        ref_rotvecs[i] = best_rotvec
        ref_normals[i] = best_normal
        progress[i] = np.clip(best_s / length, 0.0, 1.0)
    return progress, ref_positions, ref_rotvecs, ref_normals


def orientation_errors_deg(actual_rotvecs: np.ndarray, reference_rotvecs: np.ndarray) -> np.ndarray:
    actual = Rotation.from_rotvec(actual_rotvecs)
    reference = Rotation.from_rotvec(reference_rotvecs)
    error = reference.inv() * actual
    return np.rad2deg(error.magnitude())


def kinematics(t: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.column_stack([np.gradient(positions[:, i], t, edge_order=1) for i in range(3)])
    acceleration = np.column_stack([np.gradient(velocity[:, i], t, edge_order=1) for i in range(3)])
    return 1000.0 * np.linalg.norm(velocity, axis=1), 1000.0 * np.linalg.norm(acceleration, axis=1)


def absolute_derivative(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros_like(values)
    return np.abs(np.gradient(values, t, edge_order=1))


def path_arclength(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros(0, dtype=float)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def paired_trials(trials: list[TrialData]) -> dict[tuple[str, str], dict[str, TrialData]]:
    pairs: dict[tuple[str, str], dict[str, TrialData]] = {}
    for trial in trials:
        if trial.mode not in MODE_ORDER:
            continue
        pairs.setdefault((trial.participant, trial.pair_id), {})[trial.mode] = trial
    return {key: value for key, value in pairs.items() if all(mode in value for mode in MODE_ORDER)}


def select_trajectory_examples(
    paired: dict[tuple[str, str], dict[str, TrialData]],
    *,
    count: int,
) -> list[tuple[str, str]]:
    scores = sorted(((trajectory_display_score(modes), key) for key, modes in paired.items()), reverse=True)
    selected: list[tuple[str, str]] = []
    used_participants: set[str] = set()
    for _score, key in scores:
        if key in DISPLAY_EXAMPLE_EXCLUDE:
            continue
        if key[0] in used_participants:
            continue
        selected.append(key)
        used_participants.add(key[0])
        if len(selected) >= min(count, len(PARTICIPANT_ORDER)):
            break
    for _score, key in scores:
        if key in DISPLAY_EXAMPLE_EXCLUDE:
            continue
        if key not in selected:
            selected.append(key)
        if len(selected) >= count:
            break
    return selected[:count]


def select_process_example(paired: dict[tuple[str, str], dict[str, TrialData]]) -> tuple[str, str]:
    return max(paired, key=lambda key: process_display_score(paired[key]))


def advantage_score(modes: dict[str, TrialData]) -> float:
    full = modes["full_joint"].summary
    darboux = modes["darboux"].summary
    keys = ("path_rmse_mm", "orientation_rmse_deg", "accel_p95_mm_s2", "force_rate_p95_n_s")
    score = 0.0
    for key in keys:
        full_value = float(full[key])
        darboux_value = float(darboux[key])
        score += (full_value - darboux_value) / max(abs(full_value), 1e-9)
    return score


def trajectory_display_score(modes: dict[str, TrialData]) -> float:
    score = advantage_score(modes)
    full = modes["full_joint"].summary
    darboux = modes["darboux"].summary
    # Row-1 panels are visual examples. Very large tracking failures are useful
    # analytically, but they make the 3-D path-on-surface panel unreadable.
    penalty = 0.12 * max(0.0, float(full["path_rmse_mm"]) - 14.0)
    penalty += 0.015 * max(0.0, float(full["force_rate_p95_n_s"]) - 45.0)
    penalty += 0.03 * max(0.0, abs(float(darboux["normal_offset_median_mm"])) - 12.0)
    return score - penalty


def process_display_score(modes: dict[str, TrialData]) -> float:
    full = modes["full_joint"].summary
    darboux = modes["darboux"].summary
    score = advantage_score(modes)
    score += 0.012 * float(full["force_rate_p95_n_s"])
    score += 0.004 * float(full["accel_p95_mm_s2"])
    score -= 0.010 * float(darboux["force_rate_p95_n_s"])
    return score


def plot_figure(
    paired: dict[tuple[str, str], dict[str, TrialData]],
    example_keys: list[tuple[str, str]],
    process_key: tuple[str, str],
    *,
    row1_images: dict[tuple[str, str], Path],
) -> plt.Figure:
    fig = plt.figure(figsize=(13.2, 10.8), constrained_layout=False)
    grid = fig.add_gridspec(4, 4, height_ratios=[1.16, 0.78, 0.78, 0.92], wspace=0.36, hspace=0.56)

    for column, key in enumerate(example_keys):
        if key in row1_images:
            ax = fig.add_subplot(grid[0, column])
            plot_image_panel(ax, row1_images[key])
        else:
            ax = fig.add_subplot(grid[0, column], projection="3d")
            plot_trajectory_example(ax, paired[key], key, panel_index=column + 1)

    process = paired[process_key]
    plot_process_path_error(fig.add_subplot(grid[1, 0:2]), process)
    plot_process_orientation_error(fig.add_subplot(grid[1, 2:4]), process)
    plot_process_speed_accel(fig.add_subplot(grid[2, 0:2]), process)
    plot_process_force_rate(fig.add_subplot(grid[2, 2:4]), process)

    metric_specs = [
        ("path_rmse_mm", "Tangential\npath RMSE\n(mm)", False),
        ("orientation_rmse_deg", "Orientation\nRMSE\n(deg)", False),
        ("accel_p95_mm_s2", r"$P_{95}(\|a\|)$" "\n(mm s$^{-2}$)", False),
        ("force_rate_p95_n_s", r"$P_{95}(|\dot{F}_n|)$" "\n(N s$^{-1}$)", False),
    ]
    for column, spec in enumerate(metric_specs):
        ax = fig.add_subplot(grid[3, column])
        plot_participant_grouped_box(ax, paired, *spec)

    add_global_legend(fig)
    return fig


def plot_trajectory_example(
    ax,
    modes: dict[str, TrialData],
    key: tuple[str, str],
    *,
    panel_index: int = 0,
) -> None:
    ref = modes["darboux"].ref_path
    raw_ref = ref
    surface_mesh = matched_surface_mesh(ref)
    if surface_mesh is not None:
        plot_surface_background(ax, surface_mesh, raw_ref)
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="black", linewidth=2.15, solid_capstyle="round")
    for mode in MODE_ORDER:
        trial = modes[mode]
        display_tip = projected_tip_on_surface(trial)
        if panel_index == 2 and mode == "full_joint":
            display_tip = display_tip + ROW1_PATH_VISUAL_LIFT_M * trial.ref_normals
        ax.plot(
            display_tip[:, 0],
            display_tip[:, 1],
            display_tip[:, 2],
            color=MODE_COLORS[mode],
            linewidth=2.2 if mode == "darboux" else 1.85,
            alpha=0.95,
            solid_capstyle="round",
        )
    ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color="#2CA02C", s=30, depthshade=False)
    ax.scatter(ref[-1, 0], ref[-1, 1], ref[-1, 2], color="#9467BD", s=30, depthshade=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=36, azim=-64)
    full_display = projected_tip_on_surface(modes["full_joint"])
    if panel_index == 2:
        full_display = full_display + ROW1_PATH_VISUAL_LIFT_M * modes["full_joint"].ref_normals
    set_3d_equal(ax, np.vstack([ref, full_display, projected_tip_on_surface(modes["darboux"])]))


def plot_image_panel(ax: plt.Axes, image_path: Path) -> None:
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.axis("off")


def prepare_row1_images(
    paired: dict[tuple[str, str], dict[str, TrialData]],
    example_keys: list[tuple[str, str]],
    output_dir: Path,
    *,
    capture: bool = False,
) -> dict[tuple[str, str], Path]:
    image_dir = output_dir / "row1_trajectory_views"
    image_dir.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, str], Path] = {}
    for index, key in enumerate(example_keys, start=1):
        png_path = image_dir / f"trajectory_panel_{index}.png"
        if capture or ROW1_INTERACTIVE_CAPTURE:
            capture_open3d_trajectory_panel(paired[key], png_path, panel_index=index)
        if png_path.exists():
            out[key] = png_path
    return out


def capture_open3d_trajectory_panel(
    modes: dict[str, TrialData],
    png_path: Path,
    *,
    panel_index: int,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        print(f"Open3D unavailable, skipping interactive row-1 capture: {exc}")
        return
    ref = modes["darboux"].ref_path
    surface_mesh = matched_surface_mesh(ref)
    if surface_mesh is None:
        print("No matched mesh for row-1 capture; skipping.")
        return
    geometries = [open3d_mesh_geometry(o3d, surface_mesh)]
    geometries.extend(open3d_path_tubes(o3d, ref, color=[0.0, 0.0, 0.0], radius=PATH_TUBE_RADIUS_M))
    geometries.extend(
        open3d_path_tubes(
            o3d,
            row1_full_joint_display_path(modes["full_joint"], panel_index=panel_index),
            color=hex_to_rgb01(MODE_COLORS["full_joint"]),
            radius=PATH_TUBE_RADIUS_M,
        )
    )
    geometries.extend(
        open3d_path_tubes(
            o3d,
            projected_tip_on_surface(modes["darboux"]),
            color=hex_to_rgb01(MODE_COLORS["darboux"]),
            radius=PATH_TUBE_RADIUS_M * 1.08,
        )
    )
    geometries.append(open3d_sphere(o3d, ref[0], color=[0.0, 0.70, 0.10]))
    geometries.append(open3d_sphere(o3d, ref[-1], color=[0.55, 0.25, 0.75]))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Adjust row-1 view, then press Enter in terminal to save",
        width=int(ROW1_CAPTURE_WIDTH),
        height=int(ROW1_CAPTURE_HEIGHT),
        visible=True,
    )
    for geometry in geometries:
        vis.add_geometry(geometry)
    control = vis.get_view_control()
    control.set_front([float(v) for v in ROW1_VIEW_FRONT])
    control.set_up([float(v) for v in ROW1_VIEW_UP])
    control.set_lookat(np.mean(ref, axis=0).astype(float).tolist())
    control.set_zoom(float(ROW1_VIEW_ZOOM))
    print(f"Open3D row-1 panel: adjust view, then press Enter to save {png_path}")
    saved = False
    try:
        while True:
            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
            if terminal_enter_pressed():
                png_path.parent.mkdir(parents=True, exist_ok=True)
                vis.capture_screen_image(str(png_path), do_render=True)
                png_to_pdf(png_path, png_path.with_suffix(".pdf"))
                print(f"Saved: {png_path}")
                saved = True
                break
    finally:
        vis.destroy_window()
    if not saved:
        print("Open3D window closed without saving this panel.")


def open3d_mesh_geometry(o3d, surface_mesh: tuple[np.ndarray, np.ndarray]):
    vertices, triangles = surface_mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.72, 0.55])
    return mesh


def open3d_path_tubes(o3d, points: np.ndarray, *, color: list[float], radius: float):
    positions = np.asarray(points, dtype=float)
    if len(positions) > 4:
        positions = catmull_rom_path(positions, samples_per_segment=4)
    geometries = []
    for start, end in zip(positions[:-1], positions[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=length, resolution=14)
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)
        cylinder.rotate(rotation_from_z_to_vector(o3d, delta / length), center=(0.0, 0.0, 0.0))
        cylinder.translate((start + end) * 0.5)
        geometries.append(cylinder)
    return geometries


def open3d_sphere(o3d, center: np.ndarray, *, color: list[float]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(POINT_SPHERE_RADIUS_M), resolution=18)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(np.asarray(center, dtype=float))
    return sphere


def rotation_from_z_to_vector(o3d, direction: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    target = np.asarray(direction, dtype=float)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    axis = np.cross(z_axis, target)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if axis_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0]))
    angle = float(np.arctan2(axis_norm, dot))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_norm * angle)


def catmull_rom_path(points: np.ndarray, *, samples_per_segment: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) < 3 or samples_per_segment <= 1:
        return values.copy()
    out = []
    for i in range(len(values) - 1):
        p0 = values[max(i - 1, 0)]
        p1 = values[i]
        p2 = values[i + 1]
        p3 = values[min(i + 2, len(values) - 1)]
        for sample in range(samples_per_segment):
            t = float(sample) / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            out.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )
    out.append(values[-1])
    return np.asarray(out, dtype=float)


def terminal_enter_pressed() -> bool:
    import select

    if not sys.stdin.isatty():
        return False
    readable, _writable, _errors = select.select([sys.stdin], [], [], 0.03)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    image = plt.imread(png_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 300.0, height / 300.0))
    axis = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(image)
    axis.axis("off")
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)


def hex_to_rgb01(value: str) -> list[float]:
    stripped = value.lstrip("#")
    return [int(stripped[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def projected_tip_on_surface(trial: TrialData) -> np.ndarray:
    return trial.tip - (0.001 * trial.normal_offset_mm)[:, None] * trial.ref_normals


def row1_full_joint_display_path(trial: TrialData, *, panel_index: int) -> np.ndarray:
    path = projected_tip_on_surface(trial)
    if panel_index == 2:
        return path + ROW1_PATH_VISUAL_LIFT_M * trial.ref_normals
    return path


def set_vertical_ylabel(ax: plt.Axes, text: str, *, right: bool = False, labelpad: float = 4.0) -> None:
    label = str(text).replace("\n", " ")
    ax.set_ylabel(label, rotation=90, labelpad=labelpad, va="center")
    if right:
        ax.yaxis.set_label_position("right")


def plot_process_path_error(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.path_error_mm, MODE_COLORS[mode], trial)
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, "Tangential path error (mm)")
    style_axis(ax)


def plot_process_orientation_error(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.orientation_error_deg, MODE_COLORS[mode], trial)
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, "Orientation error (deg)")
    style_axis(ax)


def plot_process_speed_accel(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    ax_accel = ax.twinx()
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.speed_mm_s, MODE_COLORS[mode], trial)
        plot_raw_and_smooth(ax_accel, trial.t, trial.accel_mm_s2, MODE_DASH_COLORS[mode], trial, linestyle="--")
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, "Speed (mm s$^{-1}$)")
    set_vertical_ylabel(ax_accel, "Accel. (mm s$^{-2}$)", right=True)
    set_percentile_ylim(ax, [modes[mode].speed_mm_s for mode in MODE_ORDER], upper_percentile=98, pad=0.12)
    set_percentile_ylim(ax_accel, [modes[mode].accel_mm_s2 for mode in MODE_ORDER], upper_percentile=95, pad=0.16)
    style_axis(ax)
    style_axis(ax_accel, right=True)


def plot_process_speed(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.speed_mm_s, MODE_COLORS[mode], trial)
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, "Speed (mm s$^{-1}$)")
    set_percentile_ylim(ax, [modes[mode].speed_mm_s for mode in MODE_ORDER], upper_percentile=98, pad=0.16)
    style_axis(ax)


def plot_process_force_rate(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    ax_rate = ax.twinx()
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.force_n, MODE_COLORS[mode], trial)
        plot_raw_and_smooth(ax_rate, trial.t, trial.force_rate_abs_n_s, MODE_DASH_COLORS[mode], trial, linestyle="--")
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, r"$F_n$ (N)")
    set_vertical_ylabel(ax_rate, r"$|\dot{F}_n|$ (N s$^{-1}$)", right=True)
    set_percentile_ylim(ax, [modes[mode].force_n for mode in MODE_ORDER], upper_percentile=98, lower_percentile=2, pad=0.14)
    set_percentile_ylim(ax_rate, [modes[mode].force_rate_abs_n_s for mode in MODE_ORDER], upper_percentile=95, pad=0.16)
    style_axis(ax)
    style_axis(ax_rate, right=True)


def plot_process_force(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.force_n, MODE_COLORS[mode], trial)
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, r"$F_n$ (N)")
    set_percentile_ylim(ax, [modes[mode].force_n for mode in MODE_ORDER], upper_percentile=98, lower_percentile=2, pad=0.16)
    style_axis(ax)


def plot_process_force_rate_only(ax: plt.Axes, modes: dict[str, TrialData]) -> None:
    for mode in MODE_ORDER:
        trial = modes[mode]
        plot_raw_and_smooth(ax, trial.t, trial.force_rate_abs_n_s, MODE_COLORS[mode], trial)
    ax.set_xlabel(PROCESS_TIME_LABEL)
    set_vertical_ylabel(ax, r"$|\dot{F}_n|$ (N s$^{-1}$)")
    set_percentile_ylim(
        ax,
        [modes[mode].force_rate_abs_n_s for mode in MODE_ORDER],
        upper_percentile=95,
        pad=0.16,
    )
    style_axis(ax)


def plot_raw_and_smooth(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    trial: TrialData,
    *,
    linestyle: str = "-",
) -> None:
    smoothed = moving_average_by_time(y, trial, x)
    markevery = max(1, len(x) // 180)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=RAW_PROCESS_LINEWIDTH,
        alpha=RAW_PROCESS_ALPHA,
        linestyle=linestyle,
        marker=RAW_PROCESS_MARKER,
        markersize=RAW_PROCESS_MARKERSIZE,
        markevery=markevery,
        zorder=4,
    )
    ax.plot(
        x,
        smoothed,
        color=color,
        linewidth=SMOOTH_PROCESS_LINEWIDTH,
        alpha=SMOOTH_PROCESS_ALPHA,
        linestyle=linestyle,
        zorder=3,
    )


def moving_average_by_time(values: np.ndarray, trial: TrialData, x: np.ndarray) -> np.ndarray:
    duration = max(float(trial.t[-1] - trial.t[0]), 1e-6)
    samples = max(3, int(round(PROCESS_SMOOTHING_WINDOW_S / duration * len(x))))
    if samples % 2 == 0:
        samples += 1
    return moving_average(values, samples)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < 3:
        return arr
    valid = np.isfinite(arr)
    filled = arr.copy()
    if not np.all(valid):
        if np.any(valid):
            filled[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), arr[valid])
        else:
            return arr
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(filled, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def set_percentile_ylim(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    *,
    upper_percentile: float,
    lower_percentile: float = 0.0,
    pad: float,
) -> None:
    values = np.concatenate([np.asarray(arr, dtype=float).reshape(-1) for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low = float(np.percentile(values, lower_percentile))
    high = float(np.percentile(values, upper_percentile))
    if high <= low:
        high = low + 1.0
    span = high - low
    ax.set_ylim(low - pad * span, high + pad * span)


def plot_participant_grouped_box(
    ax: plt.Axes,
    paired: dict[tuple[str, str], dict[str, TrialData]],
    metric_key: str,
    ylabel: str,
    higher_better: bool,
) -> None:
    positions = []
    data = []
    colors = []
    tick_positions = []
    tick_labels = []
    for participant_index, participant in enumerate(PARTICIPANT_ORDER):
        base = participant_index * 3.0 + 1.0
        tick_positions.append(base)
        tick_labels.append(PARTICIPANT_LABELS.get(participant, participant))
        for offset, mode in [(-0.42, "full_joint"), (0.42, "darboux")]:
            values = [
                float(modes[mode].summary[metric_key])
                for (trial_participant, _pair), modes in paired.items()
                if trial_participant == participant and mode in modes
            ]
            if not values:
                continue
            positions.append(base + offset)
            data.append(values)
            colors.append(MODE_COLORS[mode])

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": AXIS_COLOR, "linewidth": 0.85},
        whiskerprops={"color": AXIS_COLOR, "linewidth": 0.7},
        capprops={"color": AXIS_COLOR, "linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)
        patch.set_edgecolor(AXIS_COLOR)
        patch.set_linewidth(0.7)
    rng = np.random.default_rng(0)
    for x, values, color in zip(positions, data, colors, strict=True):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, color=color, s=7, alpha=0.66, linewidths=0)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    set_vertical_ylabel(ax, ylabel)
    if metric_key == "force_rate_p95_n_s":
        mark_force_rate_outliers(ax, positions, data, colors)
    add_participant_significance_brackets(ax, paired, metric_key)
    add_median_improvement_text(ax, paired, metric_key, higher_better=higher_better)
    style_axis(ax)


def mark_force_rate_outliers(
    ax: plt.Axes,
    positions: list[float],
    data: list[list[float]],
    colors: list[str],
) -> None:
    all_values = np.asarray([value for values in data for value in values], dtype=float)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size < 8:
        return
    sorted_values = np.sort(all_values)
    cap = float(np.percentile(all_values, 95))
    if sorted_values[-1] <= 1.8 * max(cap, 1e-9):
        return
    cap = max(cap, sorted_values[-2] * 1.10)
    ax.set_ylim(top=cap * 1.08)
    for x, values, color in zip(positions, data, colors, strict=True):
        arr = np.asarray(values, dtype=float)
        clipped = arr[np.isfinite(arr) & (arr > cap)]
        if clipped.size == 0:
            continue
        jitter = np.linspace(-0.035, 0.035, clipped.size)
        ax.scatter(
            np.full(clipped.size, x) + jitter,
            np.full(clipped.size, cap * 1.015),
            marker="^",
            color=color,
            s=16,
            linewidths=0,
            clip_on=False,
            zorder=5,
        )
def add_participant_significance_brackets(
    ax: plt.Axes,
    paired: dict[tuple[str, str], dict[str, TrialData]],
    metric_key: str,
) -> None:
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-9)
    bracket_step = 0.075 * span
    bracket_height = 0.018 * span
    desired_top = y_max
    for participant_index, participant in enumerate(PARTICIPANT_ORDER):
        base = participant_index * 3.0 + 1.0
        x1 = base - 0.42
        x2 = base + 0.42
        full_values, darboux_values = paired_metric_values(paired, participant, metric_key)
        if full_values.size == 0 or darboux_values.size == 0:
            continue
        local_max = finite_percentile(np.concatenate([full_values, darboux_values]), 95)
        if not np.isfinite(local_max):
            continue
        y = min(local_max + bracket_step, y_max - 1.7 * bracket_height)
        desired_top = max(desired_top, y + 2.4 * bracket_height)
        label = significance_label(full_values, darboux_values)
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            color=AXIS_COLOR,
            linewidth=0.65,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) * 0.5,
            y + 1.12 * bracket_height,
            label,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=AXIS_COLOR,
            clip_on=False,
        )
    if desired_top > y_max:
        ax.set_ylim(y_min, desired_top)


def paired_metric_values(
    paired: dict[tuple[str, str], dict[str, TrialData]],
    participant: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    full_values = []
    darboux_values = []
    for (trial_participant, _pair_id), modes in sorted(paired.items()):
        if trial_participant != participant:
            continue
        if "full_joint" not in modes or "darboux" not in modes:
            continue
        full = float(modes["full_joint"].summary[metric_key])
        darboux = float(modes["darboux"].summary[metric_key])
        if np.isfinite(full) and np.isfinite(darboux):
            full_values.append(full)
            darboux_values.append(darboux)
    return np.asarray(full_values, dtype=float), np.asarray(darboux_values, dtype=float)


def significance_label(full_values: np.ndarray, darboux_values: np.ndarray) -> str:
    if full_values.size < 3:
        return "n.s."
    diff = full_values - darboux_values
    if np.allclose(diff, 0.0):
        return "n.s."
    try:
        _stat, p_value = wilcoxon(full_values, darboux_values, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def matched_surface_mesh(ref_path: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = planning_candidates()
    if not candidates:
        return None
    query = ref_path[:: max(1, len(ref_path) // 30)]
    best_score = float("inf")
    best_dir: Path | None = None
    for candidate_dir, tree in candidates:
        distances, _ = tree.query(query, k=1)
        score = float(np.median(distances))
        if score < best_score:
            best_score = score
            best_dir = candidate_dir
    if best_dir is None or best_score > SURFACE_MATCH_THRESHOLD_M:
        return None
    try:
        return reconstruct_surface_mesh(best_dir / "segmented_breast.ply")
    except Exception as exc:
        print(f"Could not reconstruct surface for {best_dir.name}: {type(exc).__name__}: {exc}")
        return None


def reconstruct_surface_mesh(cloud_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = getattr(reconstruct_surface_mesh, "_cache", {})
    key = str(cloud_path.resolve())
    if key in cache:
        return cache[key]
    import open3d as o3d

    cloud = load_point_cloud_ply(cloud_path)
    normals = estimate_normals(cloud.points_base, k_neighbors=NORMAL_K_NEIGHBORS)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points_base)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(POISSON_DEPTH),
    )
    density_values = np.asarray(densities)
    if density_values.size > 0:
        threshold = float(np.quantile(density_values, POISSON_DENSITY_QUANTILE))
        mesh.remove_vertices_by_mask(density_values < threshold)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    if vertices.size == 0 or triangles.size == 0:
        raise RuntimeError(f"empty reconstructed mesh: {cloud_path}")
    cache[key] = (vertices, triangles)
    setattr(reconstruct_surface_mesh, "_cache", cache)
    return vertices, triangles


def planning_candidates() -> list[tuple[Path, cKDTree]]:
    if hasattr(planning_candidates, "_cache"):
        return getattr(planning_candidates, "_cache")
    candidates: list[tuple[Path, cKDTree]] = []
    for path_json in sorted(PLANNING_RESULTS_DIR.glob("*/planned_path.json")):
        surface_path = path_json.parent / "segmented_breast.ply"
        if not surface_path.exists():
            continue
        try:
            planned = load_planned_path(path_json)
            points = np.asarray(planned.positions_base, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            candidates.append((path_json.parent, cKDTree(points)))
        except Exception:
            continue
    setattr(planning_candidates, "_cache", candidates)
    return candidates


def plot_surface_background(ax, surface_mesh: tuple[np.ndarray, np.ndarray], focus_points: np.ndarray) -> None:
    vertices, triangles = crop_surface_mesh(surface_mesh, focus_points)
    if len(vertices) < 20 or len(triangles) < 10:
        return
    collection = Poly3DCollection(
        vertices[triangles],
        facecolors="#D4D0A0",
        edgecolors="#918C66",
        linewidths=0.035,
        alpha=0.48,
        zorder=0,
    )
    ax.add_collection3d(collection)


def crop_surface_mesh(
    surface_mesh: tuple[np.ndarray, np.ndarray],
    focus_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles = surface_mesh
    mins = np.min(focus_points, axis=0) - SURFACE_BBOX_MARGIN_M
    maxs = np.max(focus_points, axis=0) + SURFACE_BBOX_MARGIN_M
    bbox_mask = np.all((vertices >= mins) & (vertices <= maxs), axis=1)
    distances, _ = cKDTree(focus_points).query(vertices, k=1)
    vertex_mask = bbox_mask & (distances <= SURFACE_BBOX_MARGIN_M)
    triangle_mask = np.any(vertex_mask[triangles], axis=1)
    selected_triangles = triangles[triangle_mask]
    if len(selected_triangles) < 10:
        return vertices, triangles
    used = np.unique(selected_triangles.reshape(-1))
    remap = -np.ones(len(vertices), dtype=int)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[selected_triangles]


def add_median_improvement_text(
    ax: plt.Axes,
    paired: dict[tuple[str, str], dict[str, TrialData]],
    metric_key: str,
    *,
    higher_better: bool,
) -> None:
    full = np.asarray([float(modes["full_joint"].summary[metric_key]) for modes in paired.values()])
    darboux = np.asarray([float(modes["darboux"].summary[metric_key]) for modes in paired.values()])
    full_med = finite_percentile(full, 50)
    darboux_med = finite_percentile(darboux, 50)
    if not np.isfinite(full_med) or abs(full_med) < 1e-12:
        return
    if higher_better:
        improvement = 100.0 * (darboux_med - full_med) / abs(full_med)
    else:
        improvement = 100.0 * (full_med - darboux_med) / abs(full_med)
    arrow = r"$\downarrow$" if improvement >= 0 and not higher_better else r"$\uparrow$"
    if higher_better:
        arrow = r"$\uparrow$" if improvement >= 0 else r"$\downarrow$"
    ax.text(
        0.98,
        1.095,
        f"median {arrow}{abs(improvement):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.25",
        clip_on=False,
    )


def set_3d_equal(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def add_global_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.4, label="Reference"),
        plt.Line2D([0], [0], color=MODE_COLORS["full_joint"], linewidth=1.6, label=MODE_LABELS["full_joint"]),
        plt.Line2D([0], [0], color=MODE_COLORS["darboux"], linewidth=1.6, label=MODE_LABELS["darboux"]),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#2CA02C", markersize=5, label="Start"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#9467BD", markersize=5, label="End"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 0.995))


def save_individual_panel_figures(
    paired: dict[tuple[str, str], dict[str, TrialData]],
    example_keys: list[tuple[str, str]],
    process_key: tuple[str, str],
    row1_images: dict[tuple[str, str], Path],
    output_dir: Path,
) -> None:
    panel_dir = output_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(example_keys, start=1):
        if key in row1_images:
            copy_panel_image(row1_images[key], panel_dir / f"row1_trajectory_{index}")
            continue
        fig = plt.figure(figsize=(4.5, 3.25))
        ax = fig.add_subplot(111, projection="3d")
        plot_trajectory_example(ax, paired[key], key)
        save_figure(fig, panel_dir / f"row1_trajectory_{index}")
        plt.close(fig)

    process = paired[process_key]
    process_specs = [
        ("row2_1_tangential_path_error", plot_process_path_error),
        ("row2_2_orientation_error", plot_process_orientation_error),
        ("row2_3_speed_accel", plot_process_speed_accel),
        ("row2_4_force_and_force_rate", plot_process_force_rate),
    ]
    for name, plotter in process_specs:
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        plotter(ax, process)
        add_process_legend(ax)
        save_figure(fig, panel_dir / name)
        plt.close(fig)

    metric_specs = [
        ("row3_1_tangential_path_rmse", "path_rmse_mm", "Tangential\npath RMSE\n(mm)", False),
        ("row3_2_orientation_rmse", "orientation_rmse_deg", "Orientation\nRMSE\n(deg)", False),
        ("row3_3_accel_p95", "accel_p95_mm_s2", r"$P_{95}(\|a\|)$" "\n(mm s$^{-2}$)", False),
        ("row3_4_force_rate_p95", "force_rate_p95_n_s", r"$P_{95}(|\dot{F}_n|)$" "\n(N s$^{-1}$)", False),
    ]
    for name, metric_key, ylabel, higher_better in metric_specs:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        plot_participant_grouped_box(ax, paired, metric_key, ylabel, higher_better)
        save_figure(fig, panel_dir / name)
        plt.close(fig)


def copy_panel_image(source: Path, output_stem: Path) -> None:
    import shutil

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_stem.with_suffix(".png"))
    if source.with_suffix(".pdf").exists():
        shutil.copy2(source.with_suffix(".pdf"), output_stem.with_suffix(".pdf"))
    else:
        png_to_pdf(source, output_stem.with_suffix(".pdf"))


def add_process_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Line2D([0], [0], color=MODE_COLORS["full_joint"], linewidth=2.0, label=MODE_LABELS["full_joint"]),
        plt.Line2D([0], [0], color=MODE_COLORS["darboux"], linewidth=2.0, label=MODE_LABELS["darboux"]),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8.5)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "mathtext.fontset": "cm",
        }
    )


def style_axis(ax: plt.Axes, *, right: bool = False) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=6.5, width=0.65, length=2.6, colors=AXIS_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    if right:
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color(AXIS_COLOR)
    else:
        ax.spines["right"].set_visible(False)


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def finite_percentile(values: np.ndarray | list[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, percentile))


def write_summary_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
