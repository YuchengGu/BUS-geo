from __future__ import annotations

import datetime
import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from gello.utils.control_utils import build_time_alignment_meta
from visual_guided_collection_gui.surface_teleop import interpolate_tcp_poses


@dataclass(frozen=True)
class SurfaceAutoScanResult:
    completed: bool
    stopped: bool
    saved_samples: int
    last_pose_index: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class SurfaceForceServoConfig:
    enabled: bool = False
    pressure_min_n: float = 3.0
    pressure_max_n: float = 4.0
    max_offset_m: float = 0.005
    max_lift_offset_m: float | None = None
    max_step_m: float = 0.00025
    hard_lift_pressure_n: float = 8.0
    hard_lift_lateral_force_n: float = 8.0
    hard_lift_resume_pressure_n: float = 4.5
    hard_lift_lateral_resume_n: float = 4.5
    hard_lift_step_m: float = 0.0001
    hard_lift_max_m: float = 0.08
    lowpass_alpha: float = 0.25
    mass: float = 0.1
    damping: float = 30.0
    stiffness: float = 400.0
    pressure_gain: float = 1.0
    dt_s: float | None = None


def run_surface_auto_scan(
    *,
    devices,
    tcp_poses: Sequence[np.ndarray],
    normals_base: Sequence[np.ndarray] | None = None,
    recorder,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    paused_ack_event: threading.Event | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    max_position_step_m: float = 0.001,
    max_rotation_step_rad: float = 0.006,
    force_servo: SurfaceForceServoConfig | None = None,
) -> SurfaceAutoScanResult:
    stop = stop_event or threading.Event()
    pause = pause_event or threading.Event()
    saved = 0
    last_pose_index: int | None = None
    poses = [np.asarray(pose, dtype=float).reshape(6) for pose in tcp_poses]
    normals = None if normals_base is None else [_normalize_normal(n) for n in normals_base]
    if normals is not None and len(normals) != len(poses):
        raise ValueError(f"normals_base length {len(normals)} must match tcp_poses length {len(poses)}")
    servo = force_servo or SurfaceForceServoConfig()
    force_offset_m = 0.0
    force_velocity_m_s = 0.0
    filtered_pressure_n: float | None = None
    hard_lift_active = False

    try:
        obs = devices.get_obs()
        dt_s = _force_servo_dt_s(devices, servo)
        for pose_index, target in enumerate(poses):
            if stop.is_set():
                return SurfaceAutoScanResult(False, True, saved, last_pose_index)
            if on_status is not None:
                on_status(f"Auto scan pose {pose_index + 1}/{len(poses)}")
            current = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
            interpolation_start = current.copy()
            if servo.enabled and normals is not None:
                interpolation_start[:3] -= force_offset_m * normals[pose_index]
            waypoints = interpolate_tcp_poses(
                interpolation_start,
                target,
                max_position_step_m=max_position_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
            )
            waypoint_index = 1
            while waypoint_index <= len(waypoints):
                waypoint = waypoints[waypoint_index - 1]
                _wait_if_paused(pause, stop, paused_ack_event)
                if stop.is_set():
                    return SurfaceAutoScanResult(False, True, saved, last_pose_index)
                current_obs = obs
                servo_meta: dict[str, Any] = {"auto_force_servo_enabled": bool(servo.enabled)}
                command = np.asarray(waypoint, dtype=float).copy()
                if servo.enabled and normals is not None:
                    normal = normals[pose_index]
                    (
                        force_offset_m,
                        force_velocity_m_s,
                        filtered_pressure_n,
                        hard_lift_active,
                        servo_meta,
                    ) = _update_force_servo_command(
                        current_obs,
                        command,
                        normal,
                        reference_position=waypoint[:3],
                        force_offset_m=force_offset_m,
                        force_velocity_m_s=force_velocity_m_s,
                        filtered_pressure_n=filtered_pressure_n,
                        hard_lift_active=hard_lift_active,
                        dt_s=dt_s,
                        config=servo,
                    )
                obs, action, obs_meta, action_timing = devices.step_tcp_pose(command)
                timestamp = datetime.datetime.now()
                auto_meta = {
                    "operation_mode": "auto",
                    "auto_phase": "scan",
                    "auto_scan_pose_index": int(pose_index),
                    "auto_scan_pose_count": int(len(poses)),
                    "auto_scan_waypoint_index": int(waypoint_index),
                    "auto_scan_waypoint_count": int(len(waypoints)),
                    "auto_scan_target_tcp_pose": target.tolist(),
                    "auto_scan_waypoint_tcp_pose": np.asarray(waypoint, dtype=float).tolist(),
                }
                auto_meta.update(servo_meta)
                meta = build_time_alignment_meta(
                    sample_index=recorder.sample_index,
                    episode_id=None if recorder.episode_dir is None else recorder.episode_dir.name,
                    control_loop_hz_config=getattr(devices.env, "control_rate_hz", None),
                    wall_time=timestamp,
                    obs_meta=obs_meta,
                    action_timing=action_timing,
                    step_timing=dict(getattr(devices.env, "last_step_timing", {}) or {}),
                )
                meta.update(auto_meta)
                recorder.save_sample(current_obs, action, meta=meta, timestamp=timestamp)
                saved += 1
                last_pose_index = int(pose_index)
                if on_sample is not None:
                    on_sample({"obs": current_obs, "action": action, "meta": meta})
                if stop.is_set():
                    return SurfaceAutoScanResult(False, True, saved, last_pose_index)
                if bool(servo_meta.get("auto_force_servo_hard_lift_limit_reached", False)):
                    return SurfaceAutoScanResult(
                        False,
                        False,
                        saved,
                        last_pose_index,
                        error="hard lift limit reached before pressure recovered",
                    )
                if not hard_lift_active:
                    waypoint_index += 1
        return SurfaceAutoScanResult(True, False, saved, last_pose_index)
    except Exception as exc:
        return SurfaceAutoScanResult(False, stop.is_set(), saved, last_pose_index, error=str(exc))


def _wait_if_paused(
    pause_event: threading.Event,
    stop_event: threading.Event,
    paused_ack_event: threading.Event | None,
) -> None:
    if not pause_event.is_set():
        if paused_ack_event is not None:
            paused_ack_event.clear()
        return
    if paused_ack_event is not None:
        paused_ack_event.set()
    while pause_event.is_set() and not stop_event.is_set():
        stop_event.wait(0.01)
    if paused_ack_event is not None:
        paused_ack_event.clear()


def _normalize_normal(normal: np.ndarray) -> np.ndarray:
    vec = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("surface normal must be non-zero")
    return vec / norm


def _update_force_servo_command(
    obs: dict[str, Any],
    command: np.ndarray,
    normal: np.ndarray,
    *,
    reference_position: np.ndarray,
    force_offset_m: float,
    force_velocity_m_s: float,
    filtered_pressure_n: float | None,
    hard_lift_active: bool,
    dt_s: float,
    config: SurfaceForceServoConfig,
) -> tuple[float, float, float | None, bool, dict[str, Any]]:
    meta: dict[str, Any] = {
        "auto_force_servo_enabled": True,
        "auto_force_servo_offset_m": float(force_offset_m),
        "auto_force_servo_velocity_m_s": float(force_velocity_m_s),
        "auto_force_servo_direction": "hold",
        "auto_force_servo_hard_lift_active": bool(hard_lift_active),
    }
    force = obs.get("force")
    if force is None:
        meta["auto_force_servo_valid"] = False
        return force_offset_m, force_velocity_m_s, filtered_pressure_n, False, meta
    values = np.asarray(force, dtype=float).reshape(-1)
    if values.size < 3 or not np.all(np.isfinite(values[:3])):
        meta["auto_force_servo_valid"] = False
        return force_offset_m, force_velocity_m_s, filtered_pressure_n, False, meta

    force_x_n = float(values[0])
    force_y_n = float(values[1])
    force_z_n = float(values[2])
    lateral_force_n = float(np.linalg.norm(values[:2]))
    pressure_n = -force_z_n
    alpha = float(np.clip(config.lowpass_alpha, 0.0, 1.0))
    if filtered_pressure_n is None:
        filtered = pressure_n
    else:
        filtered = alpha * pressure_n + (1.0 - alpha) * filtered_pressure_n

    direction = "hold"
    pressure_error_n = 0.0
    direction_guard_applied = False
    inward_motion_blocked = False
    hard_lift_limit_reached = False
    hard_lift_pressure_entry = filtered > float(config.hard_lift_pressure_n)
    hard_lift_lateral_entry = lateral_force_n > float(config.hard_lift_lateral_force_n)
    hard_lift_entry = hard_lift_pressure_entry or hard_lift_lateral_entry
    hard_lift_reason = "none"
    if hard_lift_pressure_entry and hard_lift_lateral_entry:
        hard_lift_reason = "pressure+lateral"
    elif hard_lift_pressure_entry:
        hard_lift_reason = "pressure"
    elif hard_lift_lateral_entry:
        hard_lift_reason = "lateral"
    if hard_lift_active:
        hard_lift_active = (
            filtered > float(config.hard_lift_resume_pressure_n)
            or lateral_force_n > float(config.hard_lift_lateral_resume_n)
        )
        if hard_lift_active and hard_lift_reason == "none":
            if filtered > float(config.hard_lift_resume_pressure_n) and lateral_force_n > float(config.hard_lift_lateral_resume_n):
                hard_lift_reason = "pressure+lateral"
            elif filtered > float(config.hard_lift_resume_pressure_n):
                hard_lift_reason = "pressure"
            else:
                hard_lift_reason = "lateral"
    elif hard_lift_entry:
        hard_lift_active = True

    if hard_lift_active:
        pressure_error_n = max(
            0.0,
            filtered - float(config.pressure_max_n),
            lateral_force_n - float(config.hard_lift_lateral_force_n),
        )
        direction = "hard_lift"
    elif filtered > float(config.pressure_max_n):
        pressure_error_n = filtered - float(config.pressure_max_n)
        direction = "lift"
    elif filtered < float(config.pressure_min_n):
        pressure_error_n = filtered - float(config.pressure_min_n)
        direction = "press"

    max_step = abs(float(config.max_step_m))
    max_velocity = max_step / dt_s if dt_s > 0.0 else 0.0
    hard_lift_step = abs(float(config.hard_lift_step_m))
    hard_lift_step = min(hard_lift_step, max_step) if max_step > 0.0 else hard_lift_step
    mass = max(float(config.mass), 1e-9)
    damping = max(float(config.damping), 0.0)
    stiffness = max(float(config.stiffness), 0.0)
    drive = float(config.pressure_gain) * pressure_error_n
    acceleration = 0.0

    if hard_lift_active:
        delta_offset = 0.0
        force_velocity_m_s = 0.0
    else:
        acceleration = (drive - damping * force_velocity_m_s - stiffness * force_offset_m) / mass
        delta_offset = force_velocity_m_s * dt_s + 0.5 * acceleration * dt_s * dt_s
        delta_offset = float(np.clip(delta_offset, -max_step, max_step))
        force_velocity_m_s += acceleration * dt_s
        if dt_s > 0.0:
            force_velocity_m_s = float(np.clip(force_velocity_m_s, -max_velocity, max_velocity))

        if direction == "lift":
            if delta_offset < 0.0:
                delta_offset = 0.0
                direction_guard_applied = True
            if force_velocity_m_s < 0.0:
                force_velocity_m_s = 0.0
                direction_guard_applied = True
        elif direction == "press":
            if delta_offset > 0.0:
                delta_offset = 0.0
                direction_guard_applied = True
            if force_velocity_m_s > 0.0:
                force_velocity_m_s = 0.0
                direction_guard_applied = True

    previous_force_offset_m = float(force_offset_m)
    force_offset_m += delta_offset

    lower_limit, upper_limit = _force_offset_limits(
        config,
        hard_lift_active=hard_lift_active,
        current_offset_m=previous_force_offset_m,
    )
    force_offset_m = float(np.clip(force_offset_m, lower_limit, upper_limit))
    if (
        (force_offset_m <= lower_limit and force_velocity_m_s < 0.0)
        or (force_offset_m >= upper_limit and force_velocity_m_s > 0.0)
    ):
        force_velocity_m_s = 0.0
    reference = np.asarray(reference_position, dtype=float).reshape(3)

    if hard_lift_active and "ee_pos_rotvec" in obs:
        current_position = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)[:3]
        current_offset = float(np.dot(current_position - reference, normal))
        desired_offset = float(np.clip(max(force_offset_m, current_offset) + hard_lift_step, lower_limit, upper_limit))
        command[:3] = current_position + (desired_offset - current_offset) * normal
        force_offset_m = desired_offset
        inward_motion_blocked = True
        hard_lift_limit_reached = (
            force_offset_m >= upper_limit - 1e-12
            and (
                filtered > float(config.hard_lift_resume_pressure_n)
                or lateral_force_n > float(config.hard_lift_lateral_resume_n)
            )
        )
    else:
        command[:3] += force_offset_m * normal

    actual_offset = float(np.dot(command[:3] - reference, normal))
    clamped_offset = float(np.clip(actual_offset, lower_limit, upper_limit))
    command[:3] += (clamped_offset - actual_offset) * normal
    if direction == "lift" and "ee_pos_rotvec" in obs:
        current_position = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)[:3]
        command_normal_motion = float(np.dot(command[:3] - current_position, normal))
        min_lift_motion = max_step if force_offset_m < upper_limit - 1e-12 else 0.0
        if command_normal_motion < min_lift_motion:
            command[:3] += (min_lift_motion - command_normal_motion) * normal
            inward_motion_blocked = True
    final_command_offset = float(np.dot(command[:3] - reference, normal))
    meta.update(
        {
            "auto_force_servo_valid": True,
            "auto_force_servo_force_x_n": force_x_n,
            "auto_force_servo_force_y_n": force_y_n,
            "auto_force_servo_force_z_n": force_z_n,
            "auto_force_servo_lateral_force_n": lateral_force_n,
            "auto_force_servo_pressure_n": pressure_n,
            "auto_force_servo_filtered_pressure_n": float(filtered),
            "auto_force_servo_offset_m": force_offset_m,
            "auto_force_servo_velocity_m_s": float(force_velocity_m_s),
            "auto_force_servo_max_velocity_m_s": float(max_velocity),
            "auto_force_servo_command_offset_m": final_command_offset,
            "auto_force_servo_direction": direction,
            "auto_force_servo_hard_lift_active": bool(hard_lift_active),
            "auto_force_servo_hard_lift_entry": bool(hard_lift_entry),
            "auto_force_servo_hard_lift_pressure_entry": bool(hard_lift_pressure_entry),
            "auto_force_servo_hard_lift_lateral_entry": bool(hard_lift_lateral_entry),
            "auto_force_servo_hard_lift_reason": hard_lift_reason,
            "auto_force_servo_hard_lift_limit_reached": bool(hard_lift_limit_reached),
            "auto_force_servo_direction_guard_applied": direction_guard_applied,
            "auto_force_servo_inward_motion_blocked": inward_motion_blocked,
            "auto_force_servo_pressure_error_n": float(pressure_error_n),
            "auto_force_servo_drive": float(drive),
            "auto_force_servo_acceleration_m_s2": float(acceleration),
            "auto_force_servo_delta_offset_m": float(delta_offset),
            "auto_force_servo_max_step_m": float(config.max_step_m),
            "auto_force_servo_max_offset_m": float(config.max_offset_m),
            "auto_force_servo_max_lift_offset_m": float(upper_limit),
            "auto_force_servo_hard_lift_pressure_n": float(config.hard_lift_pressure_n),
            "auto_force_servo_hard_lift_lateral_force_n": float(config.hard_lift_lateral_force_n),
            "auto_force_servo_hard_lift_resume_pressure_n": float(config.hard_lift_resume_pressure_n),
            "auto_force_servo_hard_lift_lateral_resume_n": float(config.hard_lift_lateral_resume_n),
            "auto_force_servo_hard_lift_step_m": float(config.hard_lift_step_m),
            "auto_force_servo_hard_lift_max_m": float(config.hard_lift_max_m),
            "auto_force_servo_mass": float(config.mass),
            "auto_force_servo_damping": float(config.damping),
            "auto_force_servo_stiffness": float(config.stiffness),
            "auto_force_servo_pressure_gain": float(config.pressure_gain),
            "auto_force_servo_dt_s": float(dt_s),
            "auto_force_servo_pressure_min_n": float(config.pressure_min_n),
            "auto_force_servo_pressure_max_n": float(config.pressure_max_n),
        }
    )
    return force_offset_m, force_velocity_m_s, float(filtered), bool(hard_lift_active), meta


def _force_offset_limits(
    config: SurfaceForceServoConfig,
    *,
    hard_lift_active: bool = False,
    current_offset_m: float = 0.0,
) -> tuple[float, float]:
    downward = abs(float(config.max_offset_m))
    normal_upward = downward if config.max_lift_offset_m is None else abs(float(config.max_lift_offset_m))
    if hard_lift_active:
        upward = max(normal_upward, abs(float(config.hard_lift_max_m)))
    else:
        upward = max(normal_upward, float(current_offset_m))
    return -downward, upward


def _force_servo_dt_s(devices, config: SurfaceForceServoConfig) -> float:
    if config.dt_s is not None and float(config.dt_s) > 0.0:
        return float(config.dt_s)
    hz = getattr(getattr(devices, "env", None), "control_rate_hz", None)
    if hz is None or float(hz) <= 0.0:
        return 0.02
    return 1.0 / float(hz)
