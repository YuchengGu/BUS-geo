from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_geodesic_real_robot"),
)

METHOD_ORDER = ("original", "moving_average", "b_spline", "geodesic")
METHOD_LABELS = {
    "original": "Original",
    "moving_average": "Moving average",
    "b_spline": "B-spline",
    "geodesic": "Geodesic",
}
METHOD_COLORS = {
    "original": "#D95F5F",
    "moving_average": "#7F7F7F",
    "b_spline": "#4C78A8",
    "geodesic": "#3A9D6F",
}
DEFAULT_DATA_ROOT = Path.home() / "bc_data" / "gello"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUPS = {
    1: {
        "label": "Path 1 (no coupling gel)",
        "coupling": "no_gel",
        "episodes": (
            "auto_scan_0630_174040",
            "auto_scan_0630_174314",
            "auto_scan_0630_174556",
            "auto_scan_0630_174853",
        ),
    },
    2: {
        "label": "Path 2 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0630_180145",
            "auto_scan_0630_180536",
            "auto_scan_0630_180839",
            "auto_scan_0630_181204",
        ),
    },
    3: {
        "label": "Path 3 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0630_182540",
            "auto_scan_0630_182849",
            "auto_scan_0630_183142",
            "auto_scan_0630_183431",
        ),
    },
    4: {
        "label": "Path 4 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0701_164336",
            "auto_scan_0701_164723",
            "auto_scan_0701_165020",
            "auto_scan_0701_165332",
        ),
    },
    5: {
        "label": "Path 5 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0701_170404",
            "auto_scan_0701_170648",
            "auto_scan_0701_170921",
            "auto_scan_0701_171154",
        ),
    },
    6: {
        "label": "Path 6 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0701_171632",
            "auto_scan_0701_171901",
            "auto_scan_0701_172113",
            "auto_scan_0701_172326",
        ),
    },
    7: {
        "label": "Path 7 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0701_172924",
            "auto_scan_0701_173210",
            "auto_scan_0701_173451",
            "auto_scan_0701_173749",
        ),
    },
    8: {
        "label": "Path 8 (coupling gel)",
        "coupling": "gel",
        "episodes": (
            "auto_scan_0701_174949",
            "auto_scan_0701_175211",
            "auto_scan_0701_175439",
            "auto_scan_0701_175654",
        ),
    },
}


@dataclass(frozen=True)
class TrialSignals:
    episode: str
    method: str
    time_s: np.ndarray
    progress: np.ndarray
    force: np.ndarray
    force_raw: np.ndarray
    force_gravity: np.ndarray
    force_bias: np.ndarray
    force_valid: np.ndarray
    hard_lift_active: np.ndarray
    hard_lift_reason: np.ndarray
    force_offset_m: np.ndarray
    command_offset_m: np.ndarray
    delta_offset_m: np.ndarray
    tcp_position_m: np.ndarray
    tcp_rotvec_rad: np.ndarray
    control_pose: np.ndarray
    path_distance_m: np.ndarray
    pose_index: np.ndarray
    pose_count: np.ndarray
    waypoint_index: np.ndarray
    waypoint_count: np.ndarray
    hard_lift_entry: np.ndarray | None = None
    hard_lift_limit_reached: np.ndarray | None = None
    filtered_pressure_n: np.ndarray | None = None
    servo_velocity_m_s: np.ndarray | None = None
    servo_acceleration_m_s2: np.ndarray | None = None
    inward_motion_blocked: np.ndarray | None = None

    @property
    def pressure_n(self) -> np.ndarray:
        return -np.asarray(self.force, dtype=float)[:, 2]

    @property
    def tangential_force_n(self) -> np.ndarray:
        values = np.asarray(self.force, dtype=float)
        return np.linalg.norm(values[:, :2], axis=1)

    @property
    def tangential_torque_nm(self) -> np.ndarray:
        values = np.asarray(self.force, dtype=float)
        return np.linalg.norm(values[:, 3:5], axis=1)

    @property
    def axial_torque_nm(self) -> np.ndarray:
        return np.abs(np.asarray(self.force, dtype=float)[:, 5])


def group_config(group: int) -> dict[str, Any]:
    try:
        return GROUPS[int(group)]
    except KeyError as exc:
        raise ValueError(f"group must be one of {sorted(GROUPS)}, got {group}") from exc


def command_progress(
    pose_index: np.ndarray,
    pose_count: np.ndarray,
    waypoint_index: np.ndarray,
    waypoint_count: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(pose_index, dtype=float)
    count = np.maximum(np.asarray(pose_count, dtype=float), 1.0)
    waypoint = np.asarray(waypoint_index, dtype=float)
    waypoint_total = np.maximum(np.asarray(waypoint_count, dtype=float), 1.0)
    return np.clip((pose + waypoint / waypoint_total) / count, 0.0, 1.0)


def resample_by_progress(
    progress: np.ndarray,
    values: np.ndarray,
    *,
    points: int = 501,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(progress, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    grid = np.linspace(0.0, 1.0, int(points))
    if x.size == 0:
        return grid, np.full_like(grid, np.nan)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    unique_y = sums / np.maximum(counts, 1)
    if unique_x.size == 1:
        return grid, np.full_like(grid, unique_y[0])
    return grid, np.interp(grid, unique_x, unique_y, left=unique_y[0], right=unique_y[-1])


def hard_lift_events(signals: TrialSignals) -> list[dict[str, Any]]:
    active = np.asarray(signals.hard_lift_active, dtype=bool)
    time_s = np.asarray(signals.time_s, dtype=float)
    reasons = np.asarray(signals.hard_lift_reason, dtype=str)
    pressure = signals.pressure_n
    lateral = signals.tangential_force_n
    offset = np.asarray(signals.force_offset_m, dtype=float)
    events: list[dict[str, Any]] = []
    index = 0
    while index < len(active):
        if not active[index] or (index > 0 and active[index - 1]):
            index += 1
            continue
        start = index
        end = start + 1
        while end < len(active) and active[end]:
            end += 1
        event_stop = min(end, len(active) - 1)
        event_slice = slice(start, max(end, start + 1))
        nonempty_reasons = [value for value in reasons[event_slice] if value]
        events.append(
            {
                "start_index": int(start),
                "end_index": int(event_stop),
                "start_time_s": float(time_s[start]),
                "end_time_s": float(time_s[event_stop]),
                "duration_s": float(max(time_s[event_stop] - time_s[start], 0.0)),
                "reason": nonempty_reasons[0] if nonempty_reasons else "unknown",
                "entry_pressure_n": float(pressure[start]),
                "entry_lateral_force_n": float(lateral[start]),
                "max_pressure_n": float(np.nanmax(pressure[event_slice])),
                "max_lateral_force_n": float(np.nanmax(lateral[event_slice])),
                "max_offset_m": float(np.nanmax(offset[event_slice])),
            }
        )
        index = max(end, index + 1)
    return events


def compute_force_metrics(signals: TrialSignals) -> dict[str, float | int | bool]:
    time_s = np.asarray(signals.time_s, dtype=float)
    force = np.asarray(signals.force, dtype=float)
    valid = np.asarray(signals.force_valid, dtype=bool)
    valid &= np.isfinite(time_s)
    valid &= np.all(np.isfinite(force), axis=1)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"{signals.episode} has fewer than two valid force samples")

    t = time_s[valid]
    wrench = force[valid]
    order = np.argsort(t, kind="stable")
    t = t[order]
    wrench = wrench[order]
    pressure = -wrench[:, 2]
    tangential = np.linalg.norm(wrench[:, :2], axis=1)
    torque_t = np.linalg.norm(wrench[:, 3:5], axis=1)
    torque_axial = np.abs(wrench[:, 5])
    torque_total = np.linalg.norm(wrench[:, 3:6], axis=1)
    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError(f"{signals.episode} has non-positive duration")

    hard_active = np.asarray(signals.hard_lift_active, dtype=bool)[valid][order]
    offset = np.asarray(signals.force_offset_m, dtype=float)[valid][order]
    shear_mask = pressure > 1.0
    shear_ratio = tangential[shear_mask] / pressure[shear_mask]
    cop_proxy = torque_t[shear_mask] / pressure[shear_mask]
    events = hard_lift_events(signals)
    event_reasons = [str(event["reason"]) for event in events]
    recovery_times = [float(event["duration_s"]) for event in events]
    hard_limit = _optional_bool_array(signals.hard_lift_limit_reached, len(time_s))

    metrics: dict[str, float | int | bool] = {
        "frame_count": int(len(time_s)),
        "valid_frame_count": int(np.count_nonzero(valid)),
        "force_valid_ratio": float(np.mean(valid)),
        "duration_s": duration,
        "pressure_mean_n": float(np.mean(pressure)),
        "pressure_median_n": float(np.median(pressure)),
        "pressure_std_n": float(np.std(pressure)),
        "pressure_p95_n": _percentile(pressure, 95),
        "pressure_p99_n": _percentile(pressure, 99),
        "pressure_max_n": float(np.max(pressure)),
        "pressure_target_band_ratio": _time_fraction((pressure >= 3.0) & (pressure <= 4.0), t),
        "pressure_under_3_ratio": _time_fraction(pressure < 3.0, t),
        "pressure_over_4_ratio": _time_fraction(pressure > 4.0, t),
        "pressure_over_8_ratio": _time_fraction(pressure > 8.0, t),
        "pressure_exposure_4_ns": _time_integral(np.maximum(pressure - 4.0, 0.0), t),
        "pressure_exposure_8_ns": _time_integral(np.maximum(pressure - 8.0, 0.0), t),
        "pressure_total_variation_n": float(np.sum(np.abs(np.diff(pressure)))),
        "pressure_variation_rate_n_s": float(np.sum(np.abs(np.diff(pressure))) / duration),
        "pressure_derivative_rms_n_s": _derivative_rms(pressure, t),
        "tangential_force_mean_n": float(np.mean(tangential)),
        "tangential_force_p95_n": _percentile(tangential, 95),
        "tangential_force_p99_n": _percentile(tangential, 99),
        "tangential_force_max_n": float(np.max(tangential)),
        "tangential_force_over_8_ratio": _time_fraction(tangential > 8.0, t),
        "tangential_exposure_8_ns": _time_integral(np.maximum(tangential - 8.0, 0.0), t),
        "tangential_impulse_ns": _time_integral(tangential, t),
        "tangential_variation_rate_n_s": float(np.sum(np.abs(np.diff(tangential))) / duration),
        "tangential_derivative_rms_n_s": _derivative_rms(tangential, t),
        "shear_ratio_p95": _percentile(shear_ratio, 95),
        "force_angle_p95_deg": _percentile(np.degrees(np.arctan2(tangential[shear_mask], pressure[shear_mask])), 95),
        "torque_tangential_mean_nm": float(np.mean(torque_t)),
        "torque_tangential_p95_nm": _percentile(torque_t, 95),
        "torque_tangential_p99_nm": _percentile(torque_t, 99),
        "torque_tangential_max_nm": float(np.max(torque_t)),
        "torque_axial_p95_nm": _percentile(torque_axial, 95),
        "torque_axial_max_nm": float(np.max(torque_axial)),
        "torque_total_p95_nm": _percentile(torque_total, 95),
        "torque_total_max_nm": float(np.max(torque_total)),
        "torque_variation_rate_nm_s": float(np.sum(np.abs(np.diff(torque_t))) / duration),
        "torque_derivative_rms_nm_s": _derivative_rms(torque_t, t),
        "cop_proxy_p95_mm": 1000.0 * _percentile(cop_proxy, 95),
        "cop_proxy_max_mm": 1000.0 * float(np.max(cop_proxy)) if cop_proxy.size else float("nan"),
        "hard_lift_event_count": int(len(events)),
        "hard_lift_pressure_event_count": int(sum("pressure" in reason for reason in event_reasons)),
        "hard_lift_lateral_event_count": int(sum("lateral" in reason for reason in event_reasons)),
        "hard_lift_ratio": _time_fraction(hard_active, t),
        "hard_lift_duration_s": _time_integral(hard_active.astype(float), t),
        "hard_lift_recovery_time_median_s": float(np.median(recovery_times)) if recovery_times else 0.0,
        "hard_lift_limit_reached": bool(np.any(hard_limit)),
        "force_offset_rms_mm": 1000.0 * float(np.sqrt(np.mean(offset * offset))),
        "force_offset_max_outward_mm": 1000.0 * float(np.max(offset)),
        "force_offset_max_inward_mm": 1000.0 * float(np.min(offset)),
        "force_offset_outward_integral_mm_s": 1000.0 * _time_integral(np.maximum(offset, 0.0), t),
        "force_offset_outward_motion_mm": 1000.0 * float(
            np.sum(np.maximum(np.diff(offset), 0.0))
        ),
        "force_offset_variation_mm": 1000.0 * float(np.sum(np.abs(np.diff(offset)))),
    }
    return metrics


def load_group(
    group: int,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    rebuild_cache: bool = False,
) -> dict[str, TrialSignals]:
    config = group_config(group)
    root = Path(data_root).expanduser()
    cache_root = Path(cache_dir).expanduser()
    trials: dict[str, TrialSignals] = {}
    for episode in config["episodes"]:
        episode_dir = root / episode
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
        cache_path = cache_root / f"{episode}.npz"
        trial = load_trial(episode_dir, cache_path=cache_path, rebuild_cache=rebuild_cache)
        if trial.method in trials:
            raise ValueError(f"Duplicate method {trial.method!r} in group {group}")
        trials[trial.method] = trial
    missing = [method for method in METHOD_ORDER if method not in trials]
    if missing:
        raise ValueError(f"Group {group} is missing methods: {missing}")
    return {method: trials[method] for method in METHOD_ORDER}


def load_trial(
    episode_dir: str | Path,
    *,
    cache_path: str | Path | None = None,
    rebuild_cache: bool = False,
) -> TrialSignals:
    episode_path = Path(episode_dir)
    target_cache = None if cache_path is None else Path(cache_path)
    if target_cache is not None and target_cache.exists() and not rebuild_cache:
        return _load_cache(target_cache)
    trial = _extract_trial(episode_path)
    if target_cache is not None:
        _save_cache(trial, target_cache)
    return trial


def _extract_trial(episode_dir: Path) -> TrialSignals:
    frame_paths = sorted(episode_dir.glob("*.pkl"))
    if not frame_paths:
        raise FileNotFoundError(f"No PKL frames found in {episode_dir}")
    rows: dict[str, list[Any]] = {
        "time_ns": [],
        "force": [],
        "force_raw": [],
        "force_gravity": [],
        "force_bias": [],
        "force_valid": [],
        "hard_lift_active": [],
        "hard_lift_reason": [],
        "force_offset_m": [],
        "command_offset_m": [],
        "delta_offset_m": [],
        "tcp_position_m": [],
        "tcp_rotvec_rad": [],
        "control_pose": [],
        "path_distance_m": [],
        "pose_index": [],
        "pose_count": [],
        "waypoint_index": [],
        "waypoint_count": [],
        "hard_lift_entry": [],
        "hard_lift_limit_reached": [],
        "filtered_pressure_n": [],
        "servo_velocity_m_s": [],
        "servo_acceleration_m_s2": [],
        "inward_motion_blocked": [],
    }
    method: str | None = None
    for frame_number, frame_path in enumerate(frame_paths, start=1):
        with frame_path.open("rb") as handle:
            sample = pickle.load(handle)
        meta = dict(sample.get("meta", {}))
        frame_method = str(meta.get("path_variant_method", "unknown"))
        if method is None:
            method = frame_method
        elif frame_method != method:
            raise ValueError(f"{episode_dir.name} contains mixed path methods: {method}, {frame_method}")
        modalities = dict(meta.get("modalities", {}))
        force_meta = dict(modalities.get("force", {}))
        rows["time_ns"].append(_float(meta.get("sample_mono_ns"), default=float(frame_number)))
        rows["force"].append(_vector(sample.get("force"), 6))
        rows["force_raw"].append(_vector(sample.get("force_raw"), 6))
        rows["force_gravity"].append(_vector(sample.get("force_gravity"), 6))
        rows["force_bias"].append(_vector(sample.get("force_bias"), 6))
        rows["force_valid"].append(bool(force_meta.get("valid", True)))
        rows["hard_lift_active"].append(bool(meta.get("auto_force_servo_hard_lift_active", False)))
        rows["hard_lift_reason"].append(str(meta.get("auto_force_servo_hard_lift_reason", "")))
        rows["force_offset_m"].append(_float(meta.get("auto_force_servo_offset_m")))
        rows["command_offset_m"].append(_float(meta.get("auto_force_servo_command_offset_m")))
        rows["delta_offset_m"].append(_float(meta.get("auto_force_servo_delta_offset_m")))
        rows["tcp_position_m"].append(_vector(sample.get("tcp_position_base"), 3))
        rows["tcp_rotvec_rad"].append(_vector(sample.get("ee_pos_rotvec"), 6)[3:])
        rows["control_pose"].append(_vector(sample.get("control"), 6))
        rows["path_distance_m"].append(_float(sample.get("path_distance_to_nearest_m")))
        rows["pose_index"].append(_int(meta.get("auto_scan_pose_index")))
        rows["pose_count"].append(_int(meta.get("auto_scan_pose_count"), default=1))
        rows["waypoint_index"].append(_int(meta.get("auto_scan_waypoint_index"), default=1))
        rows["waypoint_count"].append(_int(meta.get("auto_scan_waypoint_count"), default=1))
        rows["hard_lift_entry"].append(bool(meta.get("auto_force_servo_hard_lift_entry", False)))
        rows["hard_lift_limit_reached"].append(
            bool(meta.get("auto_force_servo_hard_lift_limit_reached", False))
        )
        rows["filtered_pressure_n"].append(_float(meta.get("auto_force_servo_filtered_pressure_n")))
        rows["servo_velocity_m_s"].append(_float(meta.get("auto_force_servo_velocity_m_s")))
        rows["servo_acceleration_m_s2"].append(_float(meta.get("auto_force_servo_acceleration_m_s2")))
        rows["inward_motion_blocked"].append(
            bool(meta.get("auto_force_servo_inward_motion_blocked", False))
        )
        if frame_number % 500 == 0 or frame_number == len(frame_paths):
            print(f"\rExtracting {episode_dir.name}: {frame_number}/{len(frame_paths)}", end="", flush=True)
    print()
    time_ns = np.asarray(rows["time_ns"], dtype=float)
    if np.nanmedian(time_ns) > 1e12:
        time_s = (time_ns - time_ns[0]) / 1e9
    else:
        time_s = time_ns - time_ns[0]
    pose_index = np.asarray(rows["pose_index"], dtype=int)
    pose_count = np.asarray(rows["pose_count"], dtype=int)
    waypoint_index = np.asarray(rows["waypoint_index"], dtype=int)
    waypoint_count = np.asarray(rows["waypoint_count"], dtype=int)
    progress = command_progress(pose_index, pose_count, waypoint_index, waypoint_count)
    return TrialSignals(
        episode=episode_dir.name,
        method=method or "unknown",
        time_s=time_s,
        progress=progress,
        force=np.asarray(rows["force"], dtype=float),
        force_raw=np.asarray(rows["force_raw"], dtype=float),
        force_gravity=np.asarray(rows["force_gravity"], dtype=float),
        force_bias=np.asarray(rows["force_bias"], dtype=float),
        force_valid=np.asarray(rows["force_valid"], dtype=bool),
        hard_lift_active=np.asarray(rows["hard_lift_active"], dtype=bool),
        hard_lift_reason=np.asarray(rows["hard_lift_reason"], dtype=str),
        force_offset_m=np.asarray(rows["force_offset_m"], dtype=float),
        command_offset_m=np.asarray(rows["command_offset_m"], dtype=float),
        delta_offset_m=np.asarray(rows["delta_offset_m"], dtype=float),
        tcp_position_m=np.asarray(rows["tcp_position_m"], dtype=float),
        tcp_rotvec_rad=np.asarray(rows["tcp_rotvec_rad"], dtype=float),
        control_pose=np.asarray(rows["control_pose"], dtype=float),
        path_distance_m=np.asarray(rows["path_distance_m"], dtype=float),
        pose_index=pose_index,
        pose_count=pose_count,
        waypoint_index=waypoint_index,
        waypoint_count=waypoint_count,
        hard_lift_entry=np.asarray(rows["hard_lift_entry"], dtype=bool),
        hard_lift_limit_reached=np.asarray(rows["hard_lift_limit_reached"], dtype=bool),
        filtered_pressure_n=np.asarray(rows["filtered_pressure_n"], dtype=float),
        servo_velocity_m_s=np.asarray(rows["servo_velocity_m_s"], dtype=float),
        servo_acceleration_m_s2=np.asarray(rows["servo_acceleration_m_s2"], dtype=float),
        inward_motion_blocked=np.asarray(rows["inward_motion_blocked"], dtype=bool),
    )


def _save_cache(trial: TrialSignals, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for field in fields(trial):
        value = getattr(trial, field.name)
        if value is None:
            continue
        payload[field.name] = np.asarray(value)
    np.savez_compressed(cache_path, **payload)


def _load_cache(cache_path: Path) -> TrialSignals:
    with np.load(cache_path, allow_pickle=False) as data:
        values: dict[str, Any] = {}
        for field in fields(TrialSignals):
            if field.name not in data:
                values[field.name] = None
                continue
            value = data[field.name]
            if field.name in {"episode", "method"}:
                values[field.name] = str(value.item())
            else:
                values[field.name] = value
    return TrialSignals(**values)


def ordered_trials(trials: dict[str, TrialSignals]) -> Iterable[tuple[str, TrialSignals]]:
    for method in METHOD_ORDER:
        yield method, trials[method]


def save_figure(fig: Any, output_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return pdf_path, png_path


def style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.8, labelsize=8)


def plot_metric_panels(
    trials: dict[str, TrialSignals],
    panels: Iterable[dict[str, Any]],
    output_stem: str | Path,
    *,
    group_label: str,
    figsize: tuple[float, float] = (8.2, 5.2),
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    panel_list = list(panels)
    columns = 3
    rows = int(np.ceil(len(panel_list) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    metrics = {method: compute_force_metrics(trial) for method, trial in ordered_trials(trials)}
    methods = list(METHOD_ORDER)
    x = np.arange(len(methods))
    colors = [METHOD_COLORS[method] for method in methods]
    labels = [METHOD_LABELS[method].replace(" ", "\n", 1) for method in methods]
    for axis, panel in zip(axes.flat, panel_list):
        values = []
        for method in methods:
            value = float(metrics[method][panel["key"]])
            if "divide_by" in panel:
                value /= max(float(metrics[method][panel["divide_by"]]), 1e-12)
            value *= float(panel.get("scale", 1.0))
            values.append(value)
        bars = axis.bar(x, values, color=colors, width=0.68, edgecolor="white", linewidth=0.6)
        axis.set_ylabel(str(panel["ylabel"]), fontsize=9)
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=7)
        if panel.get("zero", True):
            axis.set_ylim(bottom=0.0)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.annotate(
                    f"{value:.2g}",
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
        style_axis(axis)
    for axis in axes.flat[len(panel_list) :]:
        axis.remove()
    fig.suptitle(group_label, fontsize=11)
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def _time_integral(values: np.ndarray, time_s: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.trapezoid(y, t))


def _time_fraction(mask: np.ndarray, time_s: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0.0:
        return float("nan")
    return _time_integral(np.asarray(mask, dtype=float), time_s) / duration


def _derivative_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    dt = np.diff(np.asarray(time_s, dtype=float))
    dv = np.diff(np.asarray(values, dtype=float))
    valid = dt > 1e-9
    if not np.any(valid):
        return float("nan")
    derivative = dv[valid] / dt[valid]
    return float(np.sqrt(np.mean(derivative * derivative)))


def _percentile(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _optional_bool_array(values: np.ndarray | None, length: int) -> np.ndarray:
    if values is None:
        return np.zeros(length, dtype=bool)
    return np.asarray(values, dtype=bool)


def _vector(value: Any, size: int) -> np.ndarray:
    if value is None:
        return np.full(size, np.nan, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < size:
        return np.pad(array, (0, size - array.size), constant_values=np.nan)
    return array[:size]


def _float(value: Any, *, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
