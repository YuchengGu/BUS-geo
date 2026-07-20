from __future__ import annotations

import threading
import time
import json
import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from breast_path_planning.pointcloud_from_d405 import PointCloud
from breast_path_planning.geometry import rodrigues
from visual_guided_collection_gui.collection_session import TeleopLoop
from visual_guided_collection_gui.comparison_experiment import (
    ComparisonExperiment,
    generate_trial_pair,
)
from visual_guided_collection_gui.device_manager import DeviceConfig, DeviceManager
from visual_guided_collection_gui.episode_recorder import EpisodeRecorder, add_probe_tip_observation
from visual_guided_collection_gui.images import depth_to_display_rgb
from visual_guided_collection_gui.picking import pick_nearest_projected_point
from visual_guided_collection_gui.planning_session import PlanningSession
from visual_guided_collection_gui.probe_telemetry import obs_from_tcp_pose_rotvec, probe_path_telemetry_lines
from visual_guided_collection_gui.geodesic_optimize import optimize_gui_planned_path_geodesic
from visual_guided_collection_gui.path_variants import (
    apply_b_spline_variant,
    apply_moving_average_variant,
    apply_original_variant,
    original_path_for_variant,
    path_variant_context,
)
from visual_guided_collection_gui.surface_auto_scan import SurfaceForceServoConfig, run_surface_auto_scan
from visual_guided_collection_gui.surface_random_local import random_local_start_target
from visual_guided_collection_gui.surface_bo_point import (
    PathBOReference,
    select_random_path_bo_reference,
)
from visual_guided_collection_gui.surface_bayes import (
    SurfaceBOConfig,
    SurfaceBOStopSignal,
    parse_local_bounds,
    parse_ultrasound_crop,
    post_run_reset_tcp_targets,
    run_surface_bayes_optimization,
    save_surface_bo_run_artifacts,
    select_current_tcp_bo_reference,
)
from visual_guided_collection_gui.surface_teleop import (
    SurfaceTeleopState,
    SurfaceCartesianTeleopController,
    build_tcp_target,
    first_darboux_scan_line_tcp_poses,
    path_arclengths,
    path_start_tcp_targets,
    staged_surface_start_tcp_sequence,
)
from visual_guided_collection_gui.state import GuiStage, enabled_actions_for_stage


SURFACE_CONFIRM_POSITION_STEP_M = 0.001
SURFACE_CONFIRM_ROTATION_STEP_RAD = 0.006
SURFACE_AUTOSCAN_POSITION_STEP_M = 0.0005
SURFACE_AUTOSCAN_ROTATION_STEP_RAD = 0.003
SAFE_STOP_AUTO_WORKER_JOIN_TIMEOUT_S = 10.0
SURFACE_BO_EXPERIMENT_CONDITIONS: dict[str, tuple[str, str]] = {
    "surface_bo_run_full": ("bo", "full"),
    "surface_bo_run_no_penalty": ("bo", "no_penalty"),
    "surface_bo_run_force_only": ("bo", "force_only"),
    "surface_bo_run_torque_only": ("bo", "torque_only"),
    "surface_bo_run_random": ("random", "full"),
    "surface_bo_run_uniform": ("lhs", "full"),
}


def format_surface_bo_status_lines(meta: dict[str, Any] | None) -> list[str]:
    data = dict(meta or {})
    if not str(data.get("auto_phase", "")).startswith("bo"):
        return []

    trial = data.get("bo_trial_index")
    phase = data.get("bo_phase", data.get("auto_phase", "bo"))
    measured = bool(data.get("bo_is_measurement", False))
    if trial is None:
        header = f"BO {phase}"
    else:
        header = f"BO trial {int(trial) + 1} [{phase}]"
    header += " measured" if measured else " moving"
    lines = [header]

    x = data.get("bo_x")
    if x is not None:
        values = np.asarray(x, dtype=float).reshape(-1)
        if values.shape[0] >= 4:
            deg = np.degrees(values[1:4])
            lines.append(
                "x: "
                f"dn={values[0] * 1000.0:.1f}mm, "
                f"rdeg={deg[0]:.2f}, {deg[1]:.2f}, {deg[2]:.2f}"
            )

    target = data.get("bo_target_tcp_pose") or data.get("bo_best_target_tcp_pose")
    if target is not None:
        pose = np.asarray(target, dtype=float).reshape(-1)
        if pose.shape[0] >= 6:
            lines.append("target p: " + ", ".join(f"{v:.3f}" for v in pose[:3]))
            lines.append("target r: " + ", ".join(f"{v:.3f}" for v in pose[3:6]))

    if measured:
        for key in ("Q", "F"):
            if key in data:
                lines.append(f"{key}={float(data[key]):.4f}")
        if all(key in data for key in ("D", "E", "C", "S")):
            lines.append(
                "D/E/C/S="
                f"{float(data['D']):.3f}/{float(data['E']):.3f}/"
                f"{float(data['C']):.3f}/{float(data['S']):.3f}"
            )
        if "P_f" in data or "P_tau" in data:
            lines.append(
                f"Pf={float(data.get('P_f', 0.0)):.4f}, "
                f"Ptau={float(data.get('P_tau', 0.0)):.4f}"
            )
        if "force_valid" in data:
            lines.append(f"force valid={bool(data['force_valid'])}")
        if "best_F" in data and data["best_F"] is not None:
            lines.append(f"best F={float(data['best_F']):.4f}")

    return lines


def path_point_colors(
    path_length: int,
    *,
    nearest_index: int | None = None,
    future_count: int = 8,
) -> np.ndarray:
    colors = np.tile(np.array([[0.0, 0.8, 0.1]], dtype=float), (int(path_length), 1))
    if path_length <= 0:
        return colors
    colors[0] = [1.0, 0.05, 0.05]
    if nearest_index is not None:
        nearest = int(nearest_index)
        for idx in range(nearest + 1, min(path_length, nearest + future_count + 1)):
            colors[idx] = [0.0, 0.85, 1.0]
        if 0 <= nearest < path_length:
            colors[nearest] = [1.0, 0.55, 0.0]
    return colors


def path_display_color(path) -> np.ndarray:
    metadata = dict(getattr(path, "metadata", {}) or {})
    method = metadata.get("path_variant_method")
    if metadata.get("geodesic_trigger") == "gui_optimize_geodesic" or metadata.get("geodesic_resample"):
        return np.array([0.0, 0.8, 0.1], dtype=float)
    if method == "moving_average":
        return np.array([0.55, 0.55, 0.55], dtype=float)
    if method == "b_spline":
        return np.array([0.1, 0.35, 0.9], dtype=float)
    return np.array([1.0, 0.05, 0.05], dtype=float)


def path_preview_point_colors(path) -> np.ndarray:
    path_length = len(path)
    return np.tile(path_display_color(path), (int(path_length), 1))


def path_point_geometry_name(path) -> str:
    metadata = dict(getattr(path, "metadata", {}) or {})
    method = metadata.get("path_variant_method")
    if metadata.get("geodesic_trigger") == "gui_optimize_geodesic" or metadata.get("geodesic_resample"):
        return "optimized_path_points"
    if method == "moving_average":
        return "moving_average_path_points"
    if method == "b_spline":
        return "b_spline_path_points"
    return "planned_path_points"


def all_path_point_geometry_names() -> tuple[str, ...]:
    return (
        "planned_path_points",
        "optimized_path_points",
        "moving_average_path_points",
        "b_spline_path_points",
    )


def _force_vector_from_obs(force_or_obs: Any, key: str) -> np.ndarray | None:
    if isinstance(force_or_obs, dict):
        values = force_or_obs.get(key)
    else:
        values = force_or_obs if key == "force" else None
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 6:
        return None
    return array[:6]


def force_display_image(force: Any | None, width: int = 420, height: int = 220) -> np.ndarray:
    if isinstance(force, dict):
        width = max(width, 520)
        height = max(height, 260)
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    yellow = (255, 225, 0)
    muted = (170, 170, 170)
    cyan = (120, 220, 255)
    green = (90, 230, 120)
    cv2.putText(image, "FORCE", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, yellow, 3, cv2.LINE_AA)
    if force is None:
        cv2.putText(image, "waiting", (18, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.15, muted, 2, cv2.LINE_AA)
        return image

    values = _force_vector_from_obs(force, "force")
    if values is None:
        status = ""
        if isinstance(force, dict):
            status = f"valid={bool(force.get('force_sensor_valid', False))}"
            error = force.get("force_sensor_error")
            if error:
                status += f"  {error}"
        cv2.putText(image, status or "invalid", (18, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, muted, 2, cv2.LINE_AA)
        return image

    if not isinstance(force, dict):
        lines = [
            f"F {values[0]: .2f} {values[1]: .2f} {values[2]: .2f}",
            f"M {values[3]: .2f} {values[4]: .2f} {values[5]: .2f}",
        ]
        y = 106
        for line in lines:
            cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, yellow, 2, cv2.LINE_AA)
            y += 58
        return image

    raw = _force_vector_from_obs(force, "force_raw")
    gravity = _force_vector_from_obs(force, "force_gravity")
    if raw is None:
        raw = values
    if gravity is None:
        gravity = np.zeros(6, dtype=float)
    lines = [
        (f"comp F {values[0]: .2f} {values[1]: .2f} {values[2]: .2f}   M {values[3]: .2f} {values[4]: .2f} {values[5]: .2f}", yellow),
        (f"raw  F {raw[0]: .2f} {raw[1]: .2f} {raw[2]: .2f}   M {raw[3]: .2f} {raw[4]: .2f} {raw[5]: .2f}", cyan),
        (
            f"grav F {gravity[0]: .2f} {gravity[1]: .2f} {gravity[2]: .2f}   "
            f"M {gravity[3]: .2f} {gravity[4]: .2f} {gravity[5]: .2f}",
            green,
        ),
        (
            f"valid={bool(force.get('force_sensor_valid', True))}  "
            f"zero={bool(force.get('force_zeroed', False))}  "
            f"grav={bool(force.get('force_gravity_calibrated', False))}",
            muted,
        ),
    ]
    error = force.get("force_sensor_error")
    if error:
        lines.append((f"error={error}", muted))
    y = 88
    for line, color in lines:
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 38
    return image


class VisualGuidedCollectionApp:
    def __init__(self, args) -> None:
        self.args = args
        self.stage = GuiStage.DISCONNECTED
        self.devices = DeviceManager(
            DeviceConfig(
                hostname=args.hostname,
                robot_port=args.robot_port,
                hz=args.hz,
                agent_name=args.agent,
                gello_port=args.gello_port,
                force_ip=args.force_ip,
                force_gravity_calib_path=args.force_gravity_calib,
                use_force=not args.disable_force,
                use_ultrasound=not args.disable_ultrasound,
                ultrasound_index=args.ultrasound_index,
                max_joint_step_rad=args.max_joint_step_rad,
                wrist_camera=args.wrist_camera,
            )
        )
        self.planning = PlanningSession(
            t_tcp_camera_path=args.t_tcp_camera,
            output_root=args.planning_output_root,
            point_stride=args.point_stride,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            capture_settle_s=args.capture_settle_s,
        )
        self.teleop_loop = TeleopLoop(devices=self.devices)
        self.recorder: EpisodeRecorder | None = None
        self.surface_controller: SurfaceCartesianTeleopController | None = None
        self.surface_frame_axis_mode = "world-y"
        self.surface_random_local_context: dict[str, Any] | None = None
        self._surface_bo_stop_signal: SurfaceBOStopSignal | None = None
        self._surface_bo_running = False
        self._surface_bo_reference: PathBOReference | None = None
        self._surface_bo_rng = np.random.default_rng(args.surface_bo_random_seed)
        self._last_surface_bo_status_lines: list[str] = []
        self._auto_scan_stop_event: threading.Event | None = None
        self._auto_scan_pause_event: threading.Event | None = None
        self._auto_scan_paused_ack_event: threading.Event | None = None
        self._auto_scan_running = False
        self._auto_scan_thread: threading.Thread | None = None
        self._auto_scan_safe_position_reached = False
        self._geodesic_opt_running = False
        self._comparison_path_confirmed = False
        self.comparison_experiment = ComparisonExperiment(
            endpoint_radius_m=args.comparison_endpoint_radius_m,
            progress_tolerance_m=args.comparison_progress_tolerance_m,
            timeout_s=args.comparison_timeout_s,
        )
        self._comparison_pair_counter = 0
        self._comparison_rng = np.random.default_rng(args.comparison_random_seed)
        self._comparison_approach_started_monotonic: float | None = None
        self._comparison_scan_started_monotonic: float | None = None
        self._comparison_finish_pending = False
        self._comparison_direct_control_armed = False
        self._comparison_last_nearest_index: int | None = None
        self._comparison_episode_dir: Path | None = None
        self.seed_index: int | None = None
        self.latest_obs: dict[str, Any] | None = None
        self._last_gui_update_time = 0.0
        self._gui_update_period_s = 1.0 / float(args.gui_update_hz) if float(args.gui_update_hz) > 0.0 else 0.0
        self._last_scene_pose_update_time = 0.0
        self._scene_pose_update_period_s = 0.2
        self._force_monitor_stop_event = threading.Event()
        self._force_monitor_thread: threading.Thread | None = None
        self._force_monitor_period_s = 0.05
        self._lock = threading.Lock()

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("GELLO Breast Guided Collection", 2000, 1100)
        self._build_widgets()
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self._set_status("Disconnected. Start experiments/launch_nodes.py --robot ur first, then click Connect.")
        self._refresh_buttons()

    def _build_widgets(self) -> None:
        em = self.window.theme.font_size
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene.set_on_mouse(self._on_scene_mouse)
        self.window.add_child(self.scene)

        self.panel = gui.Widget()
        self.window.add_child(self.panel)

        self.d405_rgb_label = gui.Label(f"{self.args.wrist_camera} RGB")
        self.d405_depth_label = gui.Label(f"{self.args.wrist_camera} depth")
        self.ultrasound_label = gui.Label("Ultrasound")
        self.d405_rgb_widget = gui.ImageWidget()
        self.d405_depth_widget = gui.ImageWidget()
        self.ultrasound_widget = gui.ImageWidget()
        self.force_widget = gui.ImageWidget()
        self._set_placeholder_images()
        for widget in [
            self.d405_rgb_label,
            self.d405_rgb_widget,
            self.d405_depth_label,
            self.d405_depth_widget,
            self.ultrasound_label,
            self.ultrasound_widget,
            self.force_widget,
        ]:
            self.panel.add_child(widget)

        self.telemetry_label = gui.Label("Telemetry: not connected")
        self.control_mode_label = gui.Label(self._control_mode_text())
        self.status_label = gui.Label("")
        self.panel.add_child(self.telemetry_label)
        self.panel.add_child(self.control_mode_label)
        self.panel.add_child(self.status_label)

        self.comparison_participant_label = None
        self.comparison_participant_edit = None
        self.comparison_pair_label = None
        self.comparison_pair_edit = None
        if self.args.operation_mode == "comparison":
            self.comparison_pair_label = gui.Label("Path ID")
            self.comparison_pair_edit = gui.TextEdit()
            self.comparison_pair_edit.text_value = ""
            self.comparison_participant_label = gui.Label("Participant ID")
            self.comparison_participant_edit = gui.TextEdit()
            self.comparison_participant_edit.text_value = "anonymous"
            self.panel.add_child(self.comparison_pair_label)
            self.panel.add_child(self.comparison_pair_edit)
            self.panel.add_child(self.comparison_participant_label)
            self.panel.add_child(self.comparison_participant_edit)

        self.buttons: dict[str, gui.Button] = {}
        button_specs = [
            ("connect", "Connect devices", self._on_connect),
            ("start_photo_positioning", "Photo positioning", self._on_start_photo_positioning),
            ("start_gello_handover", "GELLO handover", self._on_start_gello_handover),
            ("capture_frame", "Freeze capture", self._on_capture_frame),
            ("resegment", "Re-pick seed", self._on_resegment),
            (
                "plan_path",
                "Plan path",
                self._on_plan_path,
            ),
            ("use_original_path", "Use original path", self._on_use_original_path),
            ("smooth_moving_average", "Smooth moving avg", self._on_smooth_moving_average),
            ("smooth_b_spline", "Smooth B-spline", self._on_smooth_b_spline),
            ("optimize_geodesic", "Optimize geodesic", self._on_optimize_geodesic),
            ("confirm_path", "Confirm path", self._on_confirm_path),
            ("surface_auto_scan_start", "Start auto scan", self._on_surface_auto_scan_start),
            ("surface_auto_scan_stop", "Stop auto scan", self._on_surface_auto_scan_stop),
            ("surface_bo_optimize", "Optimize local pose", self._on_surface_bo_optimize),
            ("surface_bo_stop", "Stop BO", self._on_surface_bo_stop),
            ("surface_bo_select_point", "Select path BO point", self._on_surface_bo_select_point),
            ("surface_bo_confirm_point", "Move to BO point", self._on_surface_bo_confirm_point),
            ("surface_random_local_start", "Random local start", self._on_surface_random_local_start),
            ("surface_set_neutral", "Set neutral", self._on_surface_set_neutral),
            ("surface_calibrate_x", "Calibrate +X", self._on_surface_calibrate_x),
            ("surface_calibrate_z", "Calibrate +Z", self._on_surface_calibrate_z),
            ("surface_recenter", "Clutch GELLO", self._on_surface_recenter),
            ("start_recording", "Start episode", self._on_start_recording),
            ("toggle_fine_scan", "Fine-scan flag", self._on_toggle_fine_scan),
            ("stop_recording", "Stop episode", self._on_stop_recording),
            ("safe_stop", "Safe stop", self._on_safe_stop),
        ]
        if self.args.operation_mode == "comparison":
            button_specs.extend(
                [
                    (
                        "comparison_generate_pair",
                        "Generate trial pair",
                        self._on_comparison_generate_pair,
                    ),
                    (
                        "comparison_confirm_participant",
                        "Confirm participant",
                        self._on_comparison_confirm_participant,
                    ),
                    (
                        "comparison_full_joint",
                        "Full-joint trial",
                        self._on_comparison_full_joint,
                    ),
                    (
                        "comparison_darboux",
                        "Darboux trial",
                        self._on_comparison_darboux,
                    ),
                    (
                        "comparison_arm_direct",
                        "Take direct control",
                        self._on_comparison_arm_direct,
                    ),
                    (
                        "comparison_start_scan",
                        "Start 5 cm scan",
                        self._on_comparison_start_scan,
                    ),
                    (
                        "comparison_finish_trial",
                        "Finish manually",
                        self._on_comparison_finish_trial,
                    ),
                    (
                        "comparison_finish_participant",
                        "Finish participant",
                        self._on_comparison_finish_participant,
                    ),
                ]
            )
        if self.args.operation_mode == "bo":
            button_specs.extend(
                [
                    (
                        "surface_bo_run_full",
                        "Run BO full",
                        lambda: self._on_surface_bo_optimize("bo", "full"),
                    ),
                    (
                        "surface_bo_run_no_penalty",
                        "Run BO no penalty",
                        lambda: self._on_surface_bo_optimize("bo", "no_penalty"),
                    ),
                    (
                        "surface_bo_run_force_only",
                        "Run BO force only",
                        lambda: self._on_surface_bo_optimize("bo", "force_only"),
                    ),
                    (
                        "surface_bo_run_torque_only",
                        "Run BO torque only",
                        lambda: self._on_surface_bo_optimize("bo", "torque_only"),
                    ),
                    (
                        "surface_bo_run_random",
                        "Run random full",
                        lambda: self._on_surface_bo_optimize("random", "full"),
                    ),
                    (
                        "surface_bo_run_uniform",
                        "Run LHS full",
                        lambda: self._on_surface_bo_optimize("lhs", "full"),
                    ),
                ]
            )
        for key, text, callback in button_specs:
            button = gui.Button(text)
            button.set_on_clicked(callback)
            self.buttons[key] = button
            self.panel.add_child(button)

    def _set_placeholder_images(self) -> None:
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        self._update_rgb(self.d405_rgb_widget, placeholder)
        self._update_rgb(self.d405_depth_widget, placeholder)
        self._update_rgb(self.ultrasound_widget, placeholder)
        self._update_rgb(self.force_widget, force_display_image(None))

    def _on_layout(self, layout_context) -> None:
        content = self.window.content_rect
        em = self.window.theme.font_size
        margin = int(0.6 * em)
        gap = int(0.5 * em)
        label_h = int(1.3 * em)
        panel_width = int(0.46 * content.width)
        self.scene.frame = gui.Rect(content.x, content.y, content.width - panel_width, content.height)
        panel_left = content.get_right() - panel_width
        self.panel.frame = gui.Rect(panel_left, content.y, panel_width, content.height)
        panel_x = panel_left + margin
        panel_y = content.y + margin
        inner_w = panel_width - 2 * margin

        half_w = int((inner_w - gap) / 2)
        top_h = int(half_w * 9 / 16)
        self.d405_rgb_label.frame = gui.Rect(panel_x, panel_y, half_w, label_h)
        self.d405_depth_label.frame = gui.Rect(panel_x + half_w + gap, panel_y, half_w, label_h)
        top_y = panel_y + label_h
        self.d405_rgb_widget.frame = gui.Rect(panel_x, top_y, half_w, top_h)
        self.d405_depth_widget.frame = gui.Rect(panel_x + half_w + gap, top_y, half_w, top_h)

        y = top_y + top_h + gap
        ultrasound_h = min(int(inner_w * 9 / 16), int(content.height * 0.27))
        self.ultrasound_label.frame = gui.Rect(panel_x, y, inner_w, label_h)
        y += label_h
        self.ultrasound_widget.frame = gui.Rect(panel_x, y, inner_w, ultrasound_h)

        y += ultrasound_h + gap
        telemetry_h = int(8.8 * em)
        force_w = int((inner_w - gap) * 0.44)
        telemetry_w = inner_w - force_w - gap
        self.force_widget.frame = gui.Rect(panel_x, y, force_w, telemetry_h)
        self.telemetry_label.frame = gui.Rect(panel_x + force_w + gap, y, telemetry_w, telemetry_h)
        y += telemetry_h + gap
        mode_h = int(1.4 * em)
        self.control_mode_label.frame = gui.Rect(panel_x, y, inner_w, mode_h)
        y += mode_h + gap
        status_h = int(2.8 * em)
        self.status_label.frame = gui.Rect(panel_x, y, inner_w, status_h)
        y += status_h + gap

        if (
            self.comparison_pair_label is not None
            and self.comparison_pair_edit is not None
            and self.comparison_participant_label is not None
            and self.comparison_participant_edit is not None
        ):
            participant_label_w = int(inner_w * 0.28)
            participant_h = int(1.8 * em)
            self.comparison_pair_label.frame = gui.Rect(
                panel_x,
                y,
                participant_label_w,
                participant_h,
            )
            self.comparison_pair_edit.frame = gui.Rect(
                panel_x + participant_label_w + gap,
                y,
                inner_w - participant_label_w - gap,
                participant_h,
            )
            y += participant_h + gap
            self.comparison_participant_label.frame = gui.Rect(
                panel_x,
                y,
                participant_label_w,
                participant_h,
            )
            self.comparison_participant_edit.frame = gui.Rect(
                panel_x + participant_label_w + gap,
                y,
                inner_w - participant_label_w - gap,
                participant_h,
            )
            y += participant_h + gap

        button_h = int(2.0 * em)
        button_w = int((inner_w - gap) / 2)
        buttons = list(self.buttons.items())
        if buttons:
            _key, first_button = buttons[0]
            first_button.frame = gui.Rect(panel_x, y, inner_w, button_h)
            y += button_h + gap
        for i, (_key, button) in enumerate(buttons[1:]):
            row = i // 2
            col = i % 2
            bx = panel_x + col * (button_w + gap)
            by = y + row * (button_h + gap)
            button.frame = gui.Rect(bx, by, button_w, button_h)

    def _set_stage(self, stage: GuiStage) -> None:
        self.stage = stage
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        enabled = enabled_actions_for_stage(self.stage)
        for key, button in self.buttons.items():
            button.enabled = key in enabled
            if key.startswith("surface_") and not self.args.control_tcp:
                button.enabled = False
            if key.startswith("comparison_"):
                button.enabled = (
                    button.enabled
                    and self.args.operation_mode == "comparison"
                )
            if key in {"surface_auto_scan_start", "surface_auto_scan_stop"}:
                button.enabled = (
                    button.enabled
                    and self.args.operation_mode == "auto"
                    and self.args.control_tcp
                )
            if key in {"surface_bo_optimize", "surface_bo_stop"} or key in SURFACE_BO_EXPERIMENT_CONDITIONS:
                button.enabled = (
                    button.enabled
                    and self.args.operation_mode in {"auto", "bo"}
                    and self.args.control_tcp
                    and not self.args.disable_ultrasound
                )
            if key in {"surface_bo_select_point", "surface_bo_confirm_point"}:
                button.enabled = (
                    button.enabled
                    and self.args.operation_mode == "bo"
                    and self.planning.planned_path is not None
                )
                if key == "surface_bo_confirm_point":
                    button.enabled = button.enabled and self._surface_bo_reference is not None
            elif self.args.operation_mode == "auto" and key in {
                "start_gello_handover",
                "surface_random_local_start",
                "surface_set_neutral",
                "surface_calibrate_x",
                "surface_calibrate_z",
                "surface_recenter",
                "start_recording",
                "toggle_fine_scan",
                "stop_recording",
            }:
                button.enabled = False
            elif self.args.operation_mode == "comparison" and key in {
                "start_gello_handover",
                "surface_random_local_start",
                "surface_recenter",
                "start_recording",
                "toggle_fine_scan",
                "stop_recording",
            }:
                button.enabled = False
            if (
                self.args.operation_mode == "comparison"
                and key
                in {
                    "surface_set_neutral",
                    "surface_calibrate_x",
                    "surface_calibrate_z",
                }
            ):
                active = self.comparison_experiment.active_trial
                button.enabled = (
                    button.enabled
                    and active is not None
                    and active.teleop_mode == "darboux"
                    and self.comparison_experiment.phase == "approach"
                )
            if key == "surface_bo_optimize":
                button.enabled = button.enabled and not self._surface_bo_running
            if key in SURFACE_BO_EXPERIMENT_CONDITIONS:
                button.enabled = (
                    button.enabled
                    and self.args.operation_mode == "bo"
                    and self._surface_bo_reference is not None
                    and not self._surface_bo_running
                )
            if key == "surface_bo_stop":
                button.enabled = button.enabled and self._surface_bo_running
            if key == "surface_auto_scan_start":
                button.enabled = button.enabled and not self._auto_scan_running
            if key == "surface_auto_scan_stop":
                button.enabled = button.enabled and self._auto_scan_running
            if key == "start_recording" and self.args.control_tcp:
                button.enabled = (
                    button.enabled
                    and self.surface_controller is not None
                    and self.surface_controller.input_axes_ready
                    and not self.surface_controller.clutch_enabled
                )
            if key == "surface_random_local_start":
                button.enabled = (
                    button.enabled
                    and self.args.surface_random_local_episodes
                    and self.surface_controller is not None
                    and self.surface_controller.input_axes_ready
                )
            if key.startswith("comparison_"):
                button.enabled = button.enabled and self._comparison_button_enabled(
                    key
                )
            if self._geodesic_opt_running and key in {
                "plan_path",
                "use_original_path",
                "smooth_moving_average",
                "smooth_b_spline",
                "optimize_geodesic",
                "confirm_path",
                "resegment",
            }:
                button.enabled = False

    def _comparison_button_enabled(self, key: str) -> bool:
        experiment = self.comparison_experiment
        active = experiment.active_trial
        if key == "comparison_generate_pair":
            return (
                self.stage == GuiStage.PATH_CONFIRMED
                and self._comparison_path_confirmed
                and active is None
                and not self._geodesic_opt_running
            )
        if key == "comparison_confirm_participant":
            return (
                self.stage == GuiStage.PATH_CONFIRMED
                and experiment.pair is not None
                and active is None
            )
        if key in {"comparison_full_joint", "comparison_darboux"}:
            mode = (
                "full_joint"
                if key == "comparison_full_joint"
                else "darboux"
            )
            return (
                self.stage == GuiStage.PATH_CONFIRMED
                and experiment.participant_id is not None
                and active is None
                and mode not in experiment.completed_modes
            )
        if key == "comparison_finish_participant":
            return (
                self.stage == GuiStage.PATH_CONFIRMED
                and experiment.participant_id is not None
                and active is None
            )
        if key == "comparison_arm_direct":
            return (
                self.stage == GuiStage.TELEOP_READY
                and active is not None
                and active.teleop_mode == "full_joint"
                and experiment.phase == "approach"
                and not self._comparison_direct_control_armed
            )
        if key == "comparison_start_scan":
            if (
                self.stage != GuiStage.TELEOP_READY
                or active is None
                or experiment.phase != "approach"
            ):
                return False
            if active.teleop_mode == "full_joint":
                return self._comparison_direct_control_armed
            return (
                self.surface_controller is not None
                and self.surface_controller.input_axes_ready
            )
        if key == "comparison_finish_trial":
            return (
                self.stage == GuiStage.RECORDING
                and active is not None
                and experiment.phase == "scan"
                and not self._comparison_finish_pending
            )
        return False

    def _set_status(self, text: str) -> None:
        self.status_label.text = text

    def _control_mode_text(self) -> str:
        if self.args.operation_mode == "bo":
            return "Control mode: Standalone local Bayesian optimization"
        if self.args.operation_mode == "auto":
            return "Control mode: Automatic surface scan + local Bayesian optimization"
        if self.args.operation_mode == "comparison":
            return "Control mode: Full-joint vs Darboux comparison experiment"
        if self.args.control_tcp:
            return "Control mode: Surface Cartesian Darboux TCP (servoL)"
        return "Control mode: Legacy GELLO joint mirroring (servoJ)"

    def _post_status(self, text: str) -> None:
        gui.Application.instance.post_to_main_thread(self.window, lambda: self._set_status(text))

    def _on_connect(self) -> None:
        self._set_status(f"Connecting UR node / {self.args.wrist_camera} / GELLO / force ...")

        def worker() -> None:
            try:
                self.devices.connect()
                gui.Application.instance.post_to_main_thread(self.window, self._connected)
            except Exception as exc:
                self._post_status(f"Connect failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _connected(self) -> None:
        self._set_stage(GuiStage.CONNECTED)
        self._start_force_monitor()
        self._set_status(f"Connected. Click Photo positioning, then move UR5 to a good {self.args.wrist_camera} view with GELLO.")

    def _start_force_monitor(self) -> None:
        if self.args.disable_force:
            return
        if self._force_monitor_thread is not None and self._force_monitor_thread.is_alive():
            return
        self._force_monitor_stop_event.clear()
        self._force_monitor_thread = threading.Thread(target=self._run_force_monitor, daemon=True)
        self._force_monitor_thread.start()

    def _stop_force_monitor(self) -> None:
        self._force_monitor_stop_event.set()
        if self._force_monitor_thread is not None:
            self._force_monitor_thread.join(timeout=1.0)
        self._force_monitor_thread = None

    def _run_force_monitor(self) -> None:
        while not self._force_monitor_stop_event.is_set():
            try:
                obs = self.devices.get_force_obs()
            except Exception as exc:
                obs = {
                    "force": None,
                    "force_sensor_valid": False,
                    "force_sensor_error": f"{type(exc).__name__}: {exc}",
                }
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda obs=obs: self._update_rgb(self.force_widget, force_display_image(obs)),
            )
            time.sleep(self._force_monitor_period_s)

    def _start_gello_control(self, next_stage: GuiStage, status: str) -> None:
        if not self.devices.connected:
            self._set_status("Devices are not connected.")
            return
        self._set_stage(next_stage)
        self.teleop_loop.start_positioning(on_sample=self._on_loop_sample, on_status=self._post_status)
        self._set_status(status)

    def _on_start_photo_positioning(self) -> None:
        self._start_gello_control(
            GuiStage.POSITIONING,
            "Photo positioning: GELLO is controlling UR5. Click Freeze capture at a good view.",
        )

    def _on_start_gello_handover(self) -> None:
        if self.args.control_tcp:
            path = self.planning.planned_path
            if path is None:
                self._set_status("Path is not confirmed yet.")
                return
            self.surface_controller = SurfaceCartesianTeleopController(
                path=path,
                probe_length_m=self.args.probe_tip_offset_m,
                translation_gains_xyz=np.full(3, float(self.args.surface_translation_gain), dtype=float),
                rotation_gains_xyz=np.full(3, float(self.args.surface_rotation_gain), dtype=float),
                frame_axis_mode=self.surface_frame_axis_mode,
                use_corner_frame_modes=True,
            )
            self._set_stage(GuiStage.TELEOP_READY)
            self._set_status(
                "Surface Cartesian calibration: move GELLO to a comfortable neutral pose, then click Set neutral."
            )
            return
        self._start_gello_control(
            GuiStage.TELEOP_READY,
            "GELLO handover active, not recording. After motion is stable, click Start episode.",
        )

    def _on_capture_frame(self) -> None:
        self.teleop_loop.stop()
        self._set_status(f"GELLO control stopped. Waiting {self.args.capture_settle_s:.2f}s, then freezing {self.args.wrist_camera} RGB-D and UR TCP ...")

        def worker() -> None:
            try:
                frozen = self.planning.capture(self.devices)
                gui.Application.instance.post_to_main_thread(self.window, lambda: self._show_frozen_frame(frozen))
            except Exception as exc:
                self._post_status(f"Capture failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _show_frozen_frame(self, frozen) -> None:
        self.seed_index = None
        self._update_rgb(self.d405_rgb_widget, frozen.rgb)
        self._update_depth(frozen.depth)
        self._show_cloud("raw_cloud", frozen.raw_cloud)
        self._set_stage(GuiStage.FRAME_CAPTURED)
        self._set_status("Cloud frozen. Shift + left click a breast seed point in the left 3D view.")

    def _on_resegment(self) -> None:
        self.seed_index = None
        if self.planning.frozen_frame is not None:
            self._show_cloud("raw_cloud", self.planning.frozen_frame.raw_cloud)
            self._set_stage(GuiStage.FRAME_CAPTURED)
        self._set_status("Current seed cleared. Shift + left click a new seed.")

    def _on_plan_path(self) -> None:
        if self.seed_index is None:
            self._set_status("No seed selected. Shift + left click the breast region first.")
            return
        self._set_status("Segmenting and planning path ...")

        def worker() -> None:
            try:
                result = self.planning.plan_from_seed(self.seed_index)
                gui.Application.instance.post_to_main_thread(self.window, lambda: self._show_plan_result(result))
            except Exception as exc:
                self._post_status(f"Planning failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_surface_bo_select_point(self) -> None:
        path = self.planning.planned_path
        if self.args.operation_mode != "bo" or path is None:
            self._set_status("Plan a path in --operation-mode bo first.")
            return
        self._surface_bo_reference = select_random_path_bo_reference(
            path,
            rng=self._surface_bo_rng,
        )
        self._show_seed_marker(self._surface_bo_reference.position_base)
        self._refresh_buttons()
        self._set_status(
            f"Selected BO path point {self._surface_bo_reference.path_index}/"
            f"{len(path) - 1}. Inspect it, then click Move to BO point or select again."
        )

    def _on_surface_bo_confirm_point(self) -> None:
        selected = self._surface_bo_reference
        if self.args.operation_mode != "bo" or selected is None:
            self._set_status("Select a random BO surface point first.")
            return
        if not self.devices.connected:
            self._set_status("Devices are not connected.")
            return
        self._set_status("Moving to the confirmed BO surface point ...")

        def worker() -> None:
            try:
                current = np.asarray(
                    self.devices.get_obs()["ee_pos_rotvec"],
                    dtype=float,
                ).reshape(6)
                current_x = rodrigues(current[3:])[:, 0]
                pre = build_tcp_target(
                    selected.position_base,
                    selected.tangent_base,
                    selected.normal_base,
                    SurfaceTeleopState(normal_offset_m=self.args.surface_approach_height_m),
                    probe_length_m=self.args.probe_tip_offset_m,
                    preferred_tcp_x_axis_base=current_x,
                )
                start = build_tcp_target(
                    selected.position_base,
                    selected.tangent_base,
                    selected.normal_base,
                    SurfaceTeleopState(normal_offset_m=self.args.surface_contact_height_m),
                    probe_length_m=self.args.probe_tip_offset_m,
                    frame_axis_mode=pre.frame_axis_mode,
                )
                self.surface_frame_axis_mode = pre.frame_axis_mode
                sequence = staged_surface_start_tcp_sequence(current, pre, start)
                stages = ("mid_translate", "mid_rotate", "pre", "start")
                log_path = self._surface_confirm_log_path("surface_bo_point")
                move_kwargs = {
                    "max_position_step_m": SURFACE_CONFIRM_POSITION_STEP_M,
                    "max_rotation_step_rad": SURFACE_CONFIRM_ROTATION_STEP_RAD,
                    "position_tolerance_m": 0.002,
                    "rotation_tolerance_rad": 0.03,
                    "timeout_s": 60.0,
                }
                for stage, pose in zip(stages, sequence):
                    self._post_status(f"BO point approach: {stage} ...")
                    self._move_surface_confirm_stage(
                        stage,
                        pose,
                        log_path,
                        **move_kwargs,
                    )
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    self._surface_bo_point_confirmed,
                )
            except Exception as exc:
                self._post_status(
                    f"BO point approach failed: {type(exc).__name__}: {exc}"
                )

        threading.Thread(target=worker, daemon=True).start()

    def _surface_bo_point_confirmed(self) -> None:
        self._set_stage(GuiStage.PATH_CONFIRMED)
        self._set_status(
            "BO point reached. Click Optimize local pose to record before, "
            "candidate, and verified-best measurements."
        )

    def _show_plan_result(self, result) -> None:
        if self.args.operation_mode == "bo":
            self._surface_bo_reference = None
        if self.args.operation_mode == "comparison":
            self._clear_comparison_pair_for_path_change()
            self._comparison_path_confirmed = False
        self._show_cloud("segmented_cloud", result.segmented_cloud)
        self._show_path(result.planned_path)
        self._set_stage(GuiStage.PATH_PLANNED)
        out = self.planning.output_dir
        if self.args.operation_mode == "bo":
            self._set_status(
                f"Path planned: {len(result.planned_path)} points. "
                "Optionally optimize/smooth it, then click Select path BO point. "
                f"Output: {out}"
            )
        else:
            self._set_status(f"Path planned: {len(result.planned_path)} points. Output: {out}")

    def _clear_comparison_pair_for_path_change(self) -> None:
        if self.args.operation_mode != "comparison":
            return
        if self.comparison_experiment.active_trial is not None:
            raise RuntimeError("cannot change path during an active comparison trial")
        self.comparison_experiment.clear_pair()
        self.surface_controller = None
        self._comparison_direct_control_armed = False
        self._comparison_last_nearest_index = None
        for name in (
            "comparison_segment_lines",
            "comparison_segment_endpoints",
        ):
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)

    def _replace_with_path_variant(self, path, *, status: str) -> None:
        if self.args.operation_mode == "bo":
            self._surface_bo_reference = None
        self._clear_comparison_pair_for_path_change()
        source_path = original_path_for_variant(self.planning)
        output_path = self.planning.replace_planned_path(path, backup_name="planned_path_before_geodesic.json")
        if path.metadata.get("path_variant_method") == "original":
            self._show_path(path, name_prefix="planned_path", include_normals=True)
        else:
            self._show_path(source_path, name_prefix="planned_path", include_normals=False)
            self._show_path(path, name_prefix="optimized_path", include_normals=True)
        if (
            self.args.operation_mode == "comparison"
            and self._comparison_path_confirmed
        ):
            self._set_stage(GuiStage.PATH_CONFIRMED)
            suffix = " Generate a new trial pair."
        else:
            self._set_stage(GuiStage.PATH_PLANNED)
            suffix = ""
        self._set_status(f"{status}. Output: {output_path}.{suffix}")

    def _on_use_original_path(self) -> None:
        if self.planning.planned_path is None:
            self._set_status("No planned path yet.")
            return
        try:
            original = apply_original_variant(original_path_for_variant(self.planning))
            self._replace_with_path_variant(original, status="Original path selected")
        except Exception as exc:
            self._set_status(f"Selecting original path failed: {type(exc).__name__}: {exc}")

    def _on_smooth_moving_average(self) -> None:
        if self.planning.planned_path is None:
            self._set_status("No planned path yet.")
            return
        try:
            smoothed = apply_moving_average_variant(original_path_for_variant(self.planning))
            meta = smoothed.metadata
            self._replace_with_path_variant(
                smoothed,
                status=(
                    "Moving-average path selected "
                    f"(window={meta['moving_average_window']}, passes={meta['moving_average_passes']})"
                ),
            )
        except Exception as exc:
            self._set_status(f"Moving-average smoothing failed: {type(exc).__name__}: {exc}")

    def _on_smooth_b_spline(self) -> None:
        if self.planning.planned_path is None:
            self._set_status("No planned path yet.")
            return
        try:
            smoothed = apply_b_spline_variant(original_path_for_variant(self.planning))
            self._replace_with_path_variant(
                smoothed,
                status=(
                    "B-spline path selected "
                    f"(s={float(smoothed.metadata['b_spline_smoothing_factor']):.6g})"
                ),
            )
        except Exception as exc:
            self._set_status(f"B-spline smoothing failed: {type(exc).__name__}: {exc}")

    def _on_optimize_geodesic(self) -> None:
        if self._geodesic_opt_running:
            self._set_status("Geodesic optimization is already running.")
            return
        result = self.planning.plan_result
        if result is None or self.planning.planned_path is None:
            self._set_status("No planned path yet.")
            return
        self._geodesic_opt_running = True
        self._refresh_buttons()
        self._set_status("Optimizing current planned path with geodesic simulated annealing ...")

        def worker() -> None:
            try:
                source_path = original_path_for_variant(self.planning)
                optimized = optimize_gui_planned_path_geodesic(
                    source_path,
                    result.segmented_cloud.points_base,
                )
                output_path = self.planning.replace_planned_path(optimized)
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: self._show_geodesic_result(source_path, optimized, output_path),
                )
            except Exception as exc:
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda error=exc: self._geodesic_failed(error),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _show_geodesic_result(self, source_path, path, output_path: Path) -> None:
        self._geodesic_opt_running = False
        if self.args.operation_mode == "bo":
            self._surface_bo_reference = None
        self._clear_comparison_pair_for_path_change()
        self._show_path(source_path, name_prefix="planned_path", include_normals=False)
        self._show_path(path, name_prefix="optimized_path", include_normals=True)
        if (
            self.args.operation_mode == "comparison"
            and self._comparison_path_confirmed
        ):
            self._set_stage(GuiStage.PATH_CONFIRMED)
        else:
            self._set_stage(GuiStage.PATH_PLANNED)
        meta = path.metadata
        self._set_status(
            "Geodesic optimized: "
            f"E {float(meta.get('geodesic_energy_initial', 0.0)):.4g} -> "
            f"{float(meta.get('geodesic_energy_final', 0.0)):.4g}, "
            f"accepted {int(meta.get('geodesic_sa_accepted_moves', 0))}. "
            f"Output: {output_path}. "
            + (
                "Generate a new trial pair."
                if self.args.operation_mode == "comparison"
                else ""
            )
        )

    def _geodesic_failed(self, error: Exception) -> None:
        self._geodesic_opt_running = False
        if (
            self.args.operation_mode == "comparison"
            and self._comparison_path_confirmed
        ):
            self._set_stage(GuiStage.PATH_CONFIRMED)
        else:
            self._set_stage(GuiStage.PATH_PLANNED)
        self._set_status(f"Geodesic optimization failed: {type(error).__name__}: {error}")

    def _on_confirm_path(self) -> None:
        path = self.planning.planned_path
        if path is None:
            self._set_status("No planned path yet.")
            return
        if not self.devices.connected:
            self._set_status("Devices are not connected; connect GELLO/UR/D405 before confirming the path.")
            return
        self.surface_controller = None
        self.surface_random_local_context = None
        if self.args.operation_mode == "comparison":
            self._comparison_path_confirmed = True
            self._set_stage(GuiStage.PATH_CONFIRMED)
            self._set_status(
                "Comparison path confirmed. Optionally optimize geodesic, "
                "then Generate trial pair."
            )
            return
        if self.args.control_tcp and self.args.surface_random_local_episodes:
            self._set_stage(GuiStage.PATH_CONFIRMED)
            self._set_status(
                "Path confirmed for random local episodes. Click GELLO handover, calibrate, then use Random local start."
            )
            return
        if self.args.control_tcp:
            self._set_status("Path confirmed. Moving UR5 TCP to surface path start with Cartesian servoL ...")

            def worker() -> None:
                try:
                    current = np.asarray(self.devices.get_obs()["ee_pos_rotvec"], dtype=float).reshape(6)
                    current_x_axis = rodrigues(current[3:])[:, 0]
                    pre, start = path_start_tcp_targets(
                        path,
                        approach_height_m=self.args.surface_approach_height_m,
                        contact_height_m=self.args.surface_contact_height_m,
                        probe_length_m=self.args.probe_tip_offset_m,
                        preferred_tcp_x_axis_base=current_x_axis,
                    )
                    self.surface_frame_axis_mode = pre.frame_axis_mode
                    mid_translate_pose, mid_rotate_pose, pre_pose, start_pose = staged_surface_start_tcp_sequence(current, pre, start)
                    log_path = self._surface_confirm_log_path()
                    self._write_surface_confirm_log(
                        log_path,
                        {
                            "kind": "start",
                            "frame_axis_mode": pre.frame_axis_mode,
                            "current_tcp_pose": current.tolist(),
                            "current_tcp_x_axis_base": current_x_axis.tolist(),
                            "first_path_point_base": np.asarray(path.positions_base[0], dtype=float).tolist(),
                            "first_path_normal_base": np.asarray(path.normals_base[0], dtype=float).tolist(),
                            "mid_translate_tcp_pose": mid_translate_pose.tolist(),
                            "mid_rotate_tcp_pose": mid_rotate_pose.tolist(),
                            "pre_tcp_pose": pre.tcp_pose_rotvec().tolist(),
                            "start_tcp_pose": start.tcp_pose_rotvec().tolist(),
                            "probe_tip_offset_m": float(self.args.probe_tip_offset_m),
                            "approach_height_m": float(self.args.surface_approach_height_m),
                            "contact_height_m": float(self.args.surface_contact_height_m),
                        },
                    )
                    move_kwargs = {
                        "max_position_step_m": SURFACE_CONFIRM_POSITION_STEP_M,
                        "max_rotation_step_rad": SURFACE_CONFIRM_ROTATION_STEP_RAD,
                        "position_tolerance_m": 0.002,
                        "rotation_tolerance_rad": 0.03,
                        "timeout_s": 60.0,
                    }
                    self._post_status("Moving TCP halfway toward 20 cm outward while keeping current orientation ...")
                    self._move_surface_confirm_stage("mid_translate", mid_translate_pose, log_path, **move_kwargs)
                    self._post_status("Rotating TCP to the surface-path start orientation at the halfway point ...")
                    self._move_surface_confirm_stage("mid_rotate", mid_rotate_pose, log_path, **move_kwargs)
                    self._post_status("Moving TCP the remaining half to 20 cm outward with orientation fixed ...")
                    self._move_surface_confirm_stage("pre", pre_pose, log_path, **move_kwargs)
                    self._post_status("Moving TCP from 20 cm outward to 5 cm outward from the first path point ...")
                    self._move_surface_confirm_stage("start", start_pose, log_path, **move_kwargs)
                    gui.Application.instance.post_to_main_thread(self.window, self._surface_path_confirmed)
                except Exception as exc:
                    self._post_status(f"Surface Cartesian confirm failed: {type(exc).__name__}: {exc}")

            threading.Thread(target=worker, daemon=True).start()
            return
        self._set_stage(GuiStage.PATH_CONFIRMED)
        self._set_status("Path confirmed. Next click GELLO handover; after stable control, start recording.")

    def _surface_path_confirmed(self) -> None:
        self._set_stage(GuiStage.PATH_CONFIRMED)
        self._set_status("Surface path confirmed. UR5 is at the first path point; move GELLO to a comfortable pose, then calibrate input axes.")

    def _on_surface_auto_scan_start(self) -> None:
        path = self.planning.planned_path
        if path is None:
            self._set_status("No planned path yet.")
            return
        if not self.args.control_tcp:
            self._set_status("Auto scan is only available with --control-tcp.")
            return
        if self.args.operation_mode != "auto":
            self._set_status("Start GUI with --operation-mode auto to use auto scan.")
            return
        if not self.devices.connected:
            self._set_status("Devices are not connected.")
            return
        if self._auto_scan_running:
            self._set_status("Auto scan is already running.")
            return

        self.teleop_loop.stop()
        self.recorder = EpisodeRecorder(
            data_dir=self.args.data_dir,
            agent_name=self.args.agent,
            planned_path=path,
            probe_tip_offset_m=self.args.probe_tip_offset_m,
            record_rgb_depth=not self.args.skip_rgb_depth_recording,
            episode_context={
                "operation_mode": "auto",
                "auto_session": "surface_scan",
                **path_variant_context(path),
            },
        )
        episode_dir = self.recorder.start(datetime.datetime.now().strftime("auto_scan_%m%d_%H%M%S"))
        self._auto_scan_stop_event = threading.Event()
        self._auto_scan_pause_event = threading.Event()
        self._auto_scan_paused_ack_event = threading.Event()
        self._auto_scan_running = True
        self._auto_scan_safe_position_reached = False
        self._set_stage(GuiStage.RECORDING)
        self._set_status(f"Auto scan recording started: {episode_dir}")

        def worker() -> None:
            try:
                poses = first_darboux_scan_line_tcp_poses(
                    path,
                    contact_height_m=self.args.surface_contact_height_m,
                    probe_length_m=self.args.probe_tip_offset_m,
                    frame_axis_mode=self.surface_frame_axis_mode,
                )
                assert self.recorder is not None
                result = run_surface_auto_scan(
                    devices=self.devices,
                    tcp_poses=poses,
                    normals_base=path.normals_base,
                    recorder=self.recorder,
                    stop_event=self._auto_scan_stop_event,
                    pause_event=self._auto_scan_pause_event,
                    paused_ack_event=self._auto_scan_paused_ack_event,
                    on_sample=self._on_loop_sample,
                    on_status=self._post_status,
                    max_position_step_m=SURFACE_AUTOSCAN_POSITION_STEP_M,
                    max_rotation_step_rad=SURFACE_AUTOSCAN_ROTATION_STEP_RAD,
                    force_servo=SurfaceForceServoConfig(
                        enabled=not self.args.disable_force,
                        pressure_min_n=3.0,
                        pressure_max_n=4.0,
                        max_offset_m=0.006,
                        max_lift_offset_m=0.006,
                        max_step_m=0.00025,
                        hard_lift_pressure_n=8.0,
                        hard_lift_lateral_force_n=8.0,
                        hard_lift_resume_pressure_n=4.5,
                        hard_lift_lateral_resume_n=4.5,
                        hard_lift_step_m=0.0001,
                        hard_lift_max_m=0.08,
                        lowpass_alpha=0.25,
                        mass=0.1,
                        damping=50.0,
                        stiffness=400.0,
                        pressure_gain=1.1,
                        hold_offset_retention_ratio=0.2,
                        hard_lift_release_hold_s=0.1,
                    ),
                )
                retreat_error: Exception | None = None
                if result.completed and self.args.auto_scan_safe_retreat:
                    try:
                        retreat_error = self._run_auto_scan_safe_retreat(path, result)
                    except Exception as exc:
                        retreat_error = exc

                def done() -> None:
                    if self.recorder is not None:
                        self.recorder.stop()
                    self.recorder = None
                    self._auto_scan_running = False
                    self._auto_scan_stop_event = None
                    self._auto_scan_pause_event = None
                    self._auto_scan_paused_ack_event = None
                    self._set_stage(GuiStage.PATH_CONFIRMED)
                    if result.error:
                        self._set_status(f"Auto scan stopped with error: {result.error}. Saved {result.saved_samples} samples.")
                    elif result.stopped:
                        self._set_status(f"Auto scan stopped manually. Saved {result.saved_samples} samples.")
                    elif retreat_error is not None:
                        self._set_status(
                            "Auto scan complete, but safe retreat failed: "
                            f"{type(retreat_error).__name__}: {retreat_error}. "
                            f"Saved {result.saved_samples} samples."
                        )
                    else:
                        suffix = " Safe retreat complete." if self.args.auto_scan_safe_retreat else ""
                        self._set_status(f"Auto scan complete. Saved {result.saved_samples} samples.{suffix}")

                gui.Application.instance.post_to_main_thread(self.window, done)
            except Exception as exc:
                def failed(error: Exception = exc) -> None:
                    if self.recorder is not None:
                        self.recorder.stop()
                    self.recorder = None
                    self._auto_scan_running = False
                    self._auto_scan_stop_event = None
                    self._auto_scan_pause_event = None
                    self._auto_scan_paused_ack_event = None
                    self._set_stage(GuiStage.PATH_CONFIRMED)
                    self._set_status(f"Auto scan failed: {type(error).__name__}: {error}")

                gui.Application.instance.post_to_main_thread(self.window, failed)

        self._auto_scan_thread = threading.Thread(target=worker, daemon=True)
        self._auto_scan_thread.start()

    def _run_auto_scan_safe_retreat(self, path, result) -> Exception | None:
        if result.last_pose_index is None:
            pose_index = len(path.positions_base) - 1
        else:
            pose_index = int(np.clip(result.last_pose_index, 0, len(path.positions_base) - 1))
        warning = self._run_safe_position_retreat(
            path=path,
            pose_index=pose_index,
            reason="Auto scan complete",
        )
        self._auto_scan_safe_position_reached = True
        return warning

    def _run_safe_position_retreat(
        self,
        *,
        path=None,
        pose_index: int | None = None,
        reason: str = "Safe stop",
    ) -> Exception | None:
        retreat_warning: Exception | None = None
        if path is not None and len(path.positions_base) > 0:
            if pose_index is None:
                current = np.asarray(self.devices.get_obs()["ee_pos_rotvec"], dtype=float).reshape(6)
                positions = np.asarray(path.positions_base, dtype=float)
                pose_index = int(np.argmin(np.linalg.norm(positions - current[:3], axis=1)))
            pose_index = int(np.clip(pose_index, 0, len(path.positions_base) - 1))
            try:
                self._retreat_tcp_along_path_normal(path, pose_index=pose_index, reason=reason)
            except Exception as exc:
                retreat_warning = exc
                self._post_status(
                    f"{reason}. TCP retreat reported {type(exc).__name__}: {exc}. "
                    "Continuing to safe joint position ..."
                )
        self._move_to_safe_joint_position(reason=reason)
        return retreat_warning

    def _retreat_tcp_along_path_normal(self, path, *, pose_index: int, reason: str) -> None:
        normal = np.asarray(path.normals_base[pose_index], dtype=float).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("Cannot safe-retreat because the last path normal is invalid")
        normal = normal / norm
        current = np.asarray(self.devices.get_obs()["ee_pos_rotvec"], dtype=float).reshape(6)
        retreat_pose = current.copy()
        retreat_pose[:3] += float(self.args.auto_scan_retreat_distance_m) * normal
        self._post_status(
            f"{reason}. Retreating TCP "
            f"{float(self.args.auto_scan_retreat_distance_m) * 100.0:.1f} cm along surface normal ..."
        )
        self.devices.move_tcp_pose_linear(
            retreat_pose,
            max_position_step_m=SURFACE_CONFIRM_POSITION_STEP_M,
            max_rotation_step_rad=SURFACE_CONFIRM_ROTATION_STEP_RAD,
            position_tolerance_m=0.003,
            rotation_tolerance_rad=0.05,
            timeout_s=float(self.args.auto_scan_safe_retreat_timeout_s),
        )

    def _move_to_safe_joint_position(self, *, reason: str) -> None:
        target_joints = np.radians(np.asarray(self.args.auto_scan_safe_joint_degrees, dtype=float).reshape(6))
        self._post_status(
            f"{reason}. Moving to safe joints deg: "
            + ", ".join(f"{value:.1f}" for value in self.args.auto_scan_safe_joint_degrees)
        )
        self.devices.move_joint_positions_linear(
            target_joints,
            max_joint_step_rad=float(self.args.auto_scan_safe_joint_step_rad),
            timeout_s=float(self.args.auto_scan_safe_retreat_timeout_s),
        )

    def _on_surface_auto_scan_stop(self) -> None:
        if self._auto_scan_stop_event is None or not self._auto_scan_running:
            self._set_status("Auto scan is not running.")
            return
        self._auto_scan_stop_event.set()
        self._refresh_buttons()
        self._set_status("Stop auto scan requested. Recording will stop after the current control step.")

    def _surface_bo_config(
        self,
        search_strategy: str | None = None,
        objective_variant: str | None = None,
    ) -> SurfaceBOConfig:
        lambda_pressure = float(self.args.surface_bo_lambda_pressure)
        lambda_shear = float(self.args.surface_bo_lambda_shear)
        if self.args.surface_bo_lambda_force is not None:
            lambda_pressure = float(self.args.surface_bo_lambda_force)
            lambda_shear = float(self.args.surface_bo_lambda_force)
        pressure_max = (
            float(self.args.surface_bo_force_max)
            if self.args.surface_bo_force_max is not None
            else float(self.args.surface_bo_pressure_max)
        )
        torque_axial_max = (
            float(self.args.surface_bo_torque_max)
            if self.args.surface_bo_torque_max is not None
            else float(self.args.surface_bo_torque_axial_max)
        )
        return SurfaceBOConfig(
            bounds=parse_local_bounds(self.args.surface_bo_bounds),
            n_initial=int(self.args.surface_bo_n_initial),
            n_ei=int(self.args.surface_bo_n_ei),
            search_strategy=str(search_strategy or self.args.surface_bo_search_strategy),
            objective_variant=str(objective_variant or self.args.surface_bo_objective_variant),
            random_state=int(self.args.surface_bo_random_seed),
            force_enabled=not self.args.disable_force,
            lambda_pressure=lambda_pressure,
            lambda_shear=lambda_shear,
            lambda_torque=float(self.args.surface_bo_lambda_torque),
            lambda_axial_torque=float(self.args.surface_bo_lambda_axial_torque),
            pressure_min=float(self.args.surface_bo_pressure_min),
            pressure_max=pressure_max,
            shear_max=float(self.args.surface_bo_shear_max),
            torque_tangential_max=float(self.args.surface_bo_torque_tangential_max),
            torque_axial_max=torque_axial_max,
            large_penalty=float(self.args.surface_bo_large_penalty),
            settle_s=float(self.args.surface_bo_settle_s),
            ultrasound_crop=parse_ultrasound_crop(self.args.surface_bo_ultrasound_crop),
        )

    def _surface_bo_path_and_context(
        self,
        search_strategy: str | None = None,
        objective_variant: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        strategy = str(search_strategy or self.args.surface_bo_search_strategy)
        variant = str(objective_variant or self.args.surface_bo_objective_variant)
        if self.args.operation_mode == "bo":
            selected = self._surface_bo_reference
            if selected is None:
                raise RuntimeError("No standalone BO surface point selected")
            path = self.planning.planned_path
            if path is None:
                raise RuntimeError("No planned path yet")
            return path, {
                "operation_mode": "bo",
                "bo_session": "planned_path_point",
                "bo_path_index": int(selected.path_index),
                "bo_search_strategy": strategy,
                "bo_objective_variant": variant,
                **path_variant_context(path),
            }
        path = self.planning.planned_path
        if path is None:
            raise RuntimeError("No planned path yet")
        return path, {
            "operation_mode": "auto",
            "auto_session": "surface_bo",
            "bo_search_strategy": strategy,
            "bo_objective_variant": variant,
            **path_variant_context(path),
        }

    def _surface_bo_reference_from_obs(
        self,
        obs: dict[str, Any],
        search_strategy: str | None = None,
        objective_variant: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        path, context = self._surface_bo_path_and_context(search_strategy, objective_variant)
        temp_recorder = EpisodeRecorder(
            data_dir=self.args.data_dir,
            agent_name=self.args.agent,
            planned_path=path,
            probe_tip_offset_m=self.args.probe_tip_offset_m,
            record_rgb_depth=not self.args.skip_rgb_depth_recording,
            episode_context=context,
        )
        enriched = temp_recorder.enrich_observation(obs)
        reference_pose, normal, _nearest = select_current_tcp_bo_reference(obs, enriched)
        return reference_pose, normal, enriched

    def _make_surface_bo_recorder(
        self,
        search_strategy: str | None = None,
        objective_variant: str | None = None,
    ) -> tuple[EpisodeRecorder, bool]:
        path, context = self._surface_bo_path_and_context(search_strategy, objective_variant)
        if self.recorder is not None and self.recorder.episode_dir is not None:
            return self.recorder, False
        recorder = EpisodeRecorder(
            data_dir=self.args.data_dir,
            agent_name=self.args.agent,
            planned_path=path,
            probe_tip_offset_m=self.args.probe_tip_offset_m,
            record_rgb_depth=not self.args.skip_rgb_depth_recording,
            episode_context=context,
        )
        prefix = "bo" if self.args.operation_mode == "bo" else "auto_bo"
        recorder.start(datetime.datetime.now().strftime(f"{prefix}_%m%d_%H%M%S"))
        return recorder, True

    def _on_surface_bo_optimize(
        self,
        search_strategy: str | None = None,
        objective_variant: str | None = None,
    ) -> None:
        if self._surface_bo_running:
            self._set_status("Surface BO is already running.")
            return
        if self.args.operation_mode not in {"auto", "bo"}:
            self._set_status("Start GUI with --operation-mode auto or --operation-mode bo to use online Bayesian optimization.")
            return
        if self.args.disable_ultrasound:
            self._set_status("Online BO requires ultrasound input; do not start with --disable-ultrasound.")
            return
        if self.args.operation_mode == "auto" and self.planning.planned_path is None:
            self._set_status("No planned path yet.")
            return
        if self.args.operation_mode == "bo" and self._surface_bo_reference is None:
            self._set_status("Select and move to a standalone BO surface point first.")
            return
        if not self.devices.connected:
            self._set_status("Devices are not connected.")
            return
        actual_strategy = str(search_strategy or self.args.surface_bo_search_strategy)
        actual_variant = str(objective_variant or self.args.surface_bo_objective_variant)

        self.teleop_loop.stop()
        self._surface_bo_stop_signal = SurfaceBOStopSignal()
        self._surface_bo_running = True
        self._refresh_buttons()
        self._set_status(
            "Surface BO started "
            f"({actual_strategy}/{actual_variant}). "
            "Click Stop BO to end after the current safe point."
        )

        def worker() -> None:
            temp_recorder_created = False
            bo_recorder: EpisodeRecorder | None = None
            paused_auto_scan = False
            try:
                if (
                    self.args.operation_mode == "auto"
                    and self._auto_scan_running
                    and self._auto_scan_pause_event is not None
                ):
                    self._auto_scan_pause_event.set()
                    paused_auto_scan = True
                    if self._auto_scan_paused_ack_event is not None:
                        self._auto_scan_paused_ack_event.wait(timeout=5.0)
                obs = self.devices.get_obs()
                reference_pose, normal, enriched = self._surface_bo_reference_from_obs(
                    obs,
                    actual_strategy,
                    actual_variant,
                )
                nearest = int(enriched["path_nearest_index"])
                bo_recorder, temp_recorder_created = self._make_surface_bo_recorder(
                    actual_strategy,
                    actual_variant,
                )
                config = self._surface_bo_config(actual_strategy, actual_variant)
                result = run_surface_bayes_optimization(
                    devices=self.devices,
                    reference_tcp_pose=reference_pose,
                    normal_base=normal,
                    recorder=bo_recorder,
                    config=config,
                    stop_signal=self._surface_bo_stop_signal,
                    on_status=self._post_status,
                    on_sample=self._on_loop_sample,
                )
                if bo_recorder.episode_dir is not None:
                    save_surface_bo_run_artifacts(
                        bo_recorder.episode_dir,
                        result,
                        config=config,
                        reference_tcp_pose=reference_pose,
                        normal_base=normal,
                    )
                if not result.cancelled and result.error is None:
                    wait_s = max(0.0, float(self.args.surface_bo_post_run_wait_s))
                    retreat_m = max(0.0, float(self.args.surface_bo_reset_retreat_m))
                    if wait_s > 0.0:
                        self._post_status(f"Surface BO complete. Waiting {wait_s:.2f}s before x0 reset ...")
                        time.sleep(wait_s)
                    retreat_pose, return_pose = post_run_reset_tcp_targets(
                        reference_pose,
                        normal,
                        retreat_distance_m=retreat_m,
                    )
                    reset_kwargs = {
                        "max_position_step_m": 0.001,
                        "max_rotation_step_rad": 0.006,
                        "position_tolerance_m": 0.002,
                        "rotation_tolerance_rad": 0.03,
                        "timeout_s": 90.0,
                    }
                    if retreat_m > 0.0:
                        self._post_status(f"Surface BO x0 reset: retreat {retreat_m:.3f} m along normal ...")
                        self.devices.move_tcp_pose_linear(retreat_pose, **reset_kwargs)
                    self._post_status("Surface BO x0 reset: returning to saved x0 TCP pose ...")
                    self.devices.move_tcp_pose_linear(return_pose, **reset_kwargs)

                def done() -> None:
                    self._surface_bo_running = False
                    self._surface_bo_stop_signal = None
                    self._refresh_buttons()
                    location = (
                        f"path point {self._surface_bo_reference.path_index}"
                        if self.args.operation_mode == "bo" and self._surface_bo_reference is not None
                        else f"path index {nearest}"
                    )
                    if result.error:
                        self._set_status(
                            "Surface BO stopped with motion/objective error: "
                            f"{result.error}. Trials saved: {result.trial_count}."
                        )
                    elif result.cancelled:
                        self._set_status(
                            f"Surface BO stopped manually near {location}. "
                            f"Trials saved: {result.trial_count}."
                        )
                    else:
                        self._set_status(
                            f"Surface BO complete near {location}; returned to saved x0. "
                            f"Best F={result.best_F:.4f}, trials={result.trial_count}."
                            if result.best_F is not None
                            else f"Surface BO complete near {location}; returned to saved x0. Trials={result.trial_count}."
                        )

                gui.Application.instance.post_to_main_thread(self.window, done)
            except Exception as exc:
                def failed(error: Exception = exc) -> None:
                    self._surface_bo_running = False
                    self._surface_bo_stop_signal = None
                    self._refresh_buttons()
                    self._set_status(f"Surface BO failed: {type(error).__name__}: {error}")

                gui.Application.instance.post_to_main_thread(self.window, failed)
            finally:
                if paused_auto_scan and self._auto_scan_pause_event is not None:
                    self._auto_scan_pause_event.clear()
                if temp_recorder_created and bo_recorder is not None:
                    bo_recorder.stop()

        threading.Thread(target=worker, daemon=True).start()

    def _on_surface_bo_stop(self) -> None:
        if self._surface_bo_stop_signal is None or not self._surface_bo_running:
            self._set_status("Surface BO is not running.")
            return
        self._surface_bo_stop_signal.request_stop()
        self._refresh_buttons()
        self._set_status("Stop BO requested. Current trial will finish or abort at the next waypoint, then control returns.")

    def _on_comparison_generate_pair(self) -> None:
        path = self.planning.planned_path
        if path is None or not self._comparison_path_confirmed:
            self._set_status("Confirm the planned path before generating a trial pair.")
            return
        try:
            self._comparison_pair_counter += 1
            default_pair_id = f"pair_{self._comparison_pair_counter:03d}"
            requested_pair_id = (
                self.comparison_pair_edit.text_value
                if self.comparison_pair_edit is not None
                else ""
            )
            pair_id = self._safe_identifier(requested_pair_id, default_pair_id)
            pair = generate_trial_pair(
                path,
                pair_number=self._comparison_pair_counter,
                pair_id=pair_id,
                length_m=self.args.comparison_segment_length_m,
                rng=self._comparison_rng,
            )
            if self.comparison_pair_edit is not None:
                self.comparison_pair_edit.text_value = pair.pair_id
            self.comparison_experiment.set_pair(pair)
            self.surface_controller = None
            self._comparison_direct_control_armed = False
            self._show_comparison_segment(path, pair)
            self._refresh_buttons()
            self._set_status(
                f"Generated {pair.pair_id}: path index "
                f"{pair.start_index} -> {pair.end_index}, "
                f"length={pair.length_m * 1000.0:.1f} mm. "
                "Enter Participant ID and click Confirm participant."
            )
        except Exception as exc:
            self._set_status(
                f"Generate trial pair failed: {type(exc).__name__}: {exc}"
            )

    def _on_comparison_confirm_participant(self) -> None:
        if self.comparison_participant_edit is None:
            self._set_status("Participant input is unavailable.")
            return
        try:
            self._sync_comparison_pair_id_from_field()
            participant_id = self.comparison_participant_edit.text_value
            self.comparison_experiment.confirm_participant(participant_id)
            self.surface_controller = None
            self._comparison_direct_control_armed = False
            self._comparison_finish_pending = False
            self._comparison_last_nearest_index = None
            self._refresh_buttons()
            self._set_status(
                f"Participant {self.comparison_experiment.participant_id} confirmed. "
                "Choose Full-joint trial or Darboux trial in either order."
            )
        except Exception as exc:
            self._set_status(
                f"Confirm participant failed: {type(exc).__name__}: {exc}"
            )

    def _sync_comparison_pair_id_from_field(self) -> None:
        if (
            self.comparison_pair_edit is None
            or self.comparison_experiment.pair is None
            or self.comparison_experiment.completed_modes
        ):
            return
        pair_id = self._safe_identifier(
            self.comparison_pair_edit.text_value,
            self.comparison_experiment.pair.pair_id,
        )
        self.comparison_experiment.rename_pair(pair_id)
        self.comparison_pair_edit.text_value = pair_id

    def _on_comparison_full_joint(self) -> None:
        try:
            trial = self.comparison_experiment.begin_trial("full_joint")
        except Exception as exc:
            self._set_status(
                f"Cannot start Full-joint trial: {type(exc).__name__}: {exc}"
            )
            return
        self.teleop_loop.stop()
        self.surface_controller = None
        self._comparison_direct_control_armed = False
        self._comparison_finish_pending = False
        self._set_status(
            f"{trial.pair_id} Full-joint preparation: moving UR5 to default joints ..."
        )

        def worker() -> None:
            try:
                self._move_to_safe_joint_position(
                    reason="Full-joint comparison preparation",
                )

                def ready() -> None:
                    self._set_stage(GuiStage.TELEOP_READY)
                    self._refresh_buttons()
                    self._set_status(
                        "UR5 is at default. Move GELLO until its commanded joints "
                        "match UR5, then click Take direct control."
                    )

                gui.Application.instance.post_to_main_thread(self.window, ready)
            except Exception as exc:
                def failed(error: Exception = exc) -> None:
                    self.comparison_experiment.abort_trial()
                    self._comparison_direct_control_armed = False
                    self._set_stage(GuiStage.PATH_CONFIRMED)
                    self._set_status(
                        "Full-joint preparation failed: "
                        f"{type(error).__name__}: {error}"
                    )

                gui.Application.instance.post_to_main_thread(
                    self.window,
                    failed,
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_comparison_arm_direct(self) -> None:
        active = self.comparison_experiment.active_trial
        if active is None or active.teleop_mode != "full_joint":
            self._set_status("No Full-joint trial is being prepared.")
            return
        try:
            obs = self.devices.get_obs()
            error, _target = self.devices.joint_handover_error(obs)
            tolerance = float(
                self.args.comparison_joint_handover_tolerance_rad
            )
            if error > tolerance:
                self._set_status(
                    "Direct control remains locked: max GELLO/UR5 joint "
                    f"mismatch={error:.3f} rad, required <= {tolerance:.3f} rad."
                )
                return
            self._comparison_direct_control_armed = True
            self._comparison_approach_started_monotonic = time.monotonic()
            self.teleop_loop.start_positioning(
                on_sample=self._on_loop_sample,
                on_status=self._post_status,
            )
            self._refresh_buttons()
            self._set_status(
                "Full-joint control active. Manually move the probe tip to the "
                "red segment start, then click Start 5 cm scan."
            )
        except Exception as exc:
            self._set_status(
                f"Direct handover failed: {type(exc).__name__}: {exc}"
            )

    def _on_comparison_darboux(self) -> None:
        path = self.planning.planned_path
        pair = self.comparison_experiment.pair
        if path is None or pair is None:
            self._set_status("Generate a comparison trial pair first.")
            return
        try:
            trial = self.comparison_experiment.begin_trial("darboux")
        except Exception as exc:
            self._set_status(
                f"Cannot start Darboux trial: {type(exc).__name__}: {exc}"
            )
            return
        self.teleop_loop.stop()
        self._comparison_finish_pending = False
        self._comparison_direct_control_armed = False
        self.surface_controller = SurfaceCartesianTeleopController(
            path=path,
            probe_length_m=self.args.probe_tip_offset_m,
            translation_gains_xyz=np.full(
                3,
                float(self.args.surface_translation_gain),
                dtype=float,
            ),
            rotation_gains_xyz=np.full(
                3,
                float(self.args.surface_rotation_gain),
                dtype=float,
            ),
            frame_axis_mode=self.surface_frame_axis_mode,
            use_corner_frame_modes=True,
        )
        self._comparison_approach_started_monotonic = time.monotonic()
        self._set_status(
            f"{trial.pair_id} Darboux preparation: moving to segment start ..."
        )

        def worker() -> None:
            try:
                target = random_local_start_target(
                    path,
                    tip_height_m=self.args.surface_contact_height_m,
                    probe_length_m=self.args.probe_tip_offset_m,
                    index=pair.start_index,
                    frame_axis_mode=self.surface_frame_axis_mode,
                )
                log_path = self._surface_confirm_log_path(
                    "comparison_darboux_start"
                )
                self._write_surface_confirm_log(
                    log_path,
                    {
                        "kind": "start",
                        **pair.context(),
                        **target.meta,
                    },
                )
                self._move_surface_confirm_stage(
                    "comparison_darboux_start",
                    target.tcp_pose_base,
                    log_path,
                    max_position_step_m=SURFACE_CONFIRM_POSITION_STEP_M,
                    max_rotation_step_rad=SURFACE_CONFIRM_ROTATION_STEP_RAD,
                    position_tolerance_m=0.002,
                    rotation_tolerance_rad=0.03,
                    timeout_s=60.0,
                )

                def ready() -> None:
                    self._set_stage(GuiStage.TELEOP_READY)
                    self._refresh_buttons()
                    self._set_status(
                        "Darboux start reached. For this participant perform "
                        "Set neutral, Calibrate +X, and Calibrate +Z; return "
                        "GELLO to neutral, then click Start 5 cm scan."
                    )

                gui.Application.instance.post_to_main_thread(self.window, ready)
            except Exception as exc:
                def failed(error: Exception = exc) -> None:
                    self.comparison_experiment.abort_trial()
                    self.surface_controller = None
                    self._set_stage(GuiStage.PATH_CONFIRMED)
                    self._set_status(
                        "Darboux preparation failed: "
                        f"{type(error).__name__}: {error}"
                    )

                gui.Application.instance.post_to_main_thread(
                    self.window,
                    failed,
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_comparison_start_scan(self) -> None:
        active = self.comparison_experiment.active_trial
        path = self.planning.planned_path
        pair = self.comparison_experiment.pair
        if active is None or path is None or pair is None:
            self._set_status("No prepared comparison trial.")
            return
        if active.teleop_mode == "full_joint":
            if not self._comparison_direct_control_armed:
                self._set_status("Take direct control before starting the scan.")
                return
            action_mode = "joint_position"
        else:
            if (
                self.surface_controller is None
                or not self.surface_controller.input_axes_ready
            ):
                self._set_status(
                    "Complete Set neutral, Calibrate +X, and Calibrate +Z first."
                )
                return
            action_mode = "tcp_pose"
        try:
            if active.teleop_mode == "darboux":
                assert self.surface_controller is not None
                gello_tcp_pose, ur_tcp_pose = (
                    self._current_surface_calibration_inputs()
                )
                self.surface_controller.recenter(
                    gello_tcp_pose=gello_tcp_pose,
                    ur_tcp_pose=ur_tcp_pose,
                )
                self.surface_controller.set_clutch(False)
            obs = self.devices.get_obs()
            duration, position_error, orientation_error = (
                self._comparison_start_metrics(obs)
            )
            self.comparison_experiment.set_approach_metrics(
                duration_s=duration,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
            )
            self.comparison_experiment.start_scan()
            context = {
                **self.comparison_experiment.trial_context(
                    action_mode=action_mode
                ),
                **path_variant_context(path),
            }
            self.recorder = EpisodeRecorder(
                data_dir=self.args.data_dir,
                agent_name=self.args.agent,
                planned_path=path,
                probe_tip_offset_m=self.args.probe_tip_offset_m,
                record_rgb_depth=not self.args.skip_rgb_depth_recording,
                episode_context=context,
            )
            episode_id = self._comparison_episode_id()
            self._comparison_episode_dir = self.recorder.start(episode_id)
            self._comparison_scan_started_monotonic = time.monotonic()
            self._comparison_finish_pending = False
            self._comparison_last_nearest_index = None
            self._set_stage(GuiStage.RECORDING)
            if active.teleop_mode == "full_joint":
                self.teleop_loop.start_recording(
                    recorder=self.recorder,
                    on_sample=self._on_loop_sample,
                    on_status=self._post_status,
                )
            else:
                assert self.surface_controller is not None
                self.teleop_loop.start_surface_recording(
                    controller=self.surface_controller,
                    recorder=self.recorder,
                    on_sample=self._on_loop_sample,
                    on_status=self._post_status,
                )
            self._set_status(
                f"Recording {active.teleop_mode} scan for "
                f"{pair.length_m * 1000.0:.1f} mm. "
                "It will stop near the blue endpoint, at timeout, or by "
                "Finish manually."
            )
        except Exception as exc:
            self.teleop_loop.stop()
            if self.recorder is not None:
                self.recorder.stop()
            self.recorder = None
            self._comparison_episode_dir = None
            self._comparison_scan_started_monotonic = None
            self._comparison_finish_pending = False
            self._comparison_direct_control_armed = False
            self.comparison_experiment.abort_trial()
            self.surface_controller = None
            self._set_stage(GuiStage.PATH_CONFIRMED)
            self._set_status(
                f"Start comparison scan failed: {type(exc).__name__}: {exc}"
            )

    def _on_comparison_finish_trial(self) -> None:
        self._request_comparison_finish("manual")

    def _on_comparison_finish_participant(self) -> None:
        if self.comparison_experiment.active_trial is not None:
            self._set_status("Finish the active trial before finishing participant.")
            return
        participant = self.comparison_experiment.participant_id
        if participant is None:
            self._set_status("No participant is active.")
            return
        completed = sorted(self.comparison_experiment.completed_modes)
        self.comparison_experiment.finish_participant()
        self.surface_controller = None
        self._comparison_direct_control_armed = False
        self._refresh_buttons()
        self._set_status(
            f"Participant {participant} finished; completed={completed}. "
            "The current pair is unchanged. Enter the next Participant ID."
        )

    def _comparison_start_metrics(
        self,
        obs: dict[str, Any],
    ) -> tuple[float, float, float]:
        pair = self.comparison_experiment.pair
        if pair is None:
            raise RuntimeError("No comparison pair")
        enriched = add_probe_tip_observation(
            obs,
            self.args.probe_tip_offset_m,
        )
        target_tip = (
            pair.start_position_base
            + float(self.args.surface_contact_height_m)
            * pair.start_normal_base
        )
        position_error = float(
            np.linalg.norm(
                np.asarray(
                    enriched["probe_tip_position_base"],
                    dtype=float,
                )
                - target_tip
            )
        )
        tcp_z = np.asarray(
            enriched["probe_z_axis_base"],
            dtype=float,
        ).reshape(3)
        tcp_z /= max(float(np.linalg.norm(tcp_z)), 1e-12)
        desired_z = -np.asarray(pair.start_normal_base, dtype=float).reshape(3)
        desired_z /= max(float(np.linalg.norm(desired_z)), 1e-12)
        orientation_error = float(
            np.arccos(np.clip(np.dot(tcp_z, desired_z), -1.0, 1.0))
        )
        started = self._comparison_approach_started_monotonic
        duration = 0.0 if started is None else time.monotonic() - started
        return duration, position_error, orientation_error

    def _comparison_episode_id(self) -> str:
        active = self.comparison_experiment.active_trial
        if active is None:
            raise RuntimeError("No active comparison trial")
        safe_participant = self._safe_identifier(active.participant_id, "participant")
        safe_pair = self._safe_identifier(active.pair_id, "pair")
        stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        return (
            f"comparison_{safe_participant}_{safe_pair}_"
            f"{active.teleop_mode}_{stamp}"
        )

    @staticmethod
    def _safe_identifier(value: str, fallback: str) -> str:
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(value).strip()
        ).strip("_")
        return cleaned or fallback

    def _request_comparison_finish(self, reason: str) -> None:
        if self._comparison_finish_pending:
            return
        active = self.comparison_experiment.active_trial
        if (
            active is None
            or self.comparison_experiment.phase != "scan"
        ):
            self._set_status("No comparison scan is recording.")
            return
        self._comparison_finish_pending = True
        self._refresh_buttons()
        gui.Application.instance.post_to_main_thread(
            self.window,
            lambda: self._finish_comparison_trial(reason),
        )

    def _finish_comparison_trial(self, reason: str) -> None:
        active = self.comparison_experiment.active_trial
        pair = self.comparison_experiment.pair
        path = self.planning.planned_path
        if active is None or pair is None or path is None:
            self._comparison_finish_pending = False
            return
        self.teleop_loop.stop()
        episode_dir = self._comparison_episode_dir
        if self.recorder is not None:
            self.recorder.stop()
        self.recorder = None
        scan_duration = (
            0.0
            if self._comparison_scan_started_monotonic is None
            else time.monotonic() - self._comparison_scan_started_monotonic
        )
        trial = self.comparison_experiment.finish_trial(reason)
        summary_error: Exception | None = None
        if episode_dir is not None:
            summary = {
                **pair.context(),
                "operation_mode": "comparison",
                "participant_id": trial.participant_id,
                "teleop_mode": trial.teleop_mode,
                "trial_sequence_index": int(trial.sequence_index),
                "trial_end_reason": str(reason),
                "scan_duration_s": float(scan_duration),
                **self.comparison_experiment.approach_metrics,
            }
            try:
                (episode_dir / "comparison_trial_summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                summary_error = exc
        pose_index = (
            pair.end_index
            if self._comparison_last_nearest_index is None
            else self._comparison_last_nearest_index
        )
        self._set_status(
            f"{trial.teleop_mode} trial ended ({reason}). "
            "Retreating along the nearest path normal, then moving to default ..."
        )

        def worker() -> None:
            retreat_error: Exception | None = None
            try:
                retreat_error = self._run_safe_position_retreat(
                    path=path,
                    pose_index=pose_index,
                    reason=f"Comparison trial {reason}",
                )
            except Exception as exc:
                retreat_error = exc

            def done() -> None:
                self.surface_controller = None
                self._comparison_direct_control_armed = False
                self._comparison_finish_pending = False
                self._comparison_scan_started_monotonic = None
                self._comparison_approach_started_monotonic = None
                self._comparison_episode_dir = None
                self._set_stage(GuiStage.PATH_CONFIRMED)
                completed = sorted(
                    self.comparison_experiment.completed_modes
                )
                if retreat_error is None:
                    if summary_error is None:
                        self._set_status(
                            f"Trial complete ({reason}); completed modes={completed}. "
                            "Choose the remaining mode or Finish participant."
                        )
                    else:
                        self._set_status(
                            "Trial recording stopped and safe retreat completed, "
                            "but writing comparison summary failed: "
                            f"{type(summary_error).__name__}: {summary_error}"
                        )
                else:
                    self._set_status(
                        "Trial data saved, but retreat reported "
                        f"{type(retreat_error).__name__}: {retreat_error}"
                    )

            gui.Application.instance.post_to_main_thread(self.window, done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_surface_random_local_start(self) -> None:
        path = self.planning.planned_path
        if path is None:
            self._set_status("No planned path yet.")
            return
        if not self.args.control_tcp:
            self._set_status("Random local start is only available in surface Cartesian mode.")
            return
        if not self.args.surface_random_local_episodes:
            self._set_status("Start GUI with --surface-random-local-episodes to use Random local start.")
            return
        if self.surface_controller is None or not self.surface_controller.input_axes_ready:
            self._set_status("Calibrate surface input axes first: Set neutral, Calibrate +X, Calibrate +Z.")
            return
        if not self.devices.connected:
            self._set_status("Devices are not connected.")
            return

        self._set_status("Moving to a random local path start above the surface ...")

        def worker() -> None:
            try:
                self.teleop_loop.stop()
                target = random_local_start_target(
                    path,
                    tip_height_m=self.args.surface_random_start_height_m,
                    probe_length_m=self.args.probe_tip_offset_m,
                    frame_axis_mode=self.surface_frame_axis_mode,
                )
                self.surface_random_local_context = dict(target.meta)
                log_path = self._surface_confirm_log_path("surface_random_local_start")
                self._write_surface_confirm_log(log_path, {"kind": "start", **target.meta})
                move_kwargs = {
                    "max_position_step_m": SURFACE_CONFIRM_POSITION_STEP_M,
                    "max_rotation_step_rad": SURFACE_CONFIRM_ROTATION_STEP_RAD,
                    "position_tolerance_m": 0.002,
                    "rotation_tolerance_rad": 0.03,
                    "timeout_s": 60.0,
                }
                self._move_surface_confirm_stage("random_local_start", target.tcp_pose_base, log_path, **move_kwargs)
                obs = self.devices.get_obs()
                gello_tcp_pose = self.devices.read_gello_tcp_pose(obs)
                ur_tcp_pose = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
                assert self.surface_controller is not None
                self.surface_controller.recenter(gello_tcp_pose=gello_tcp_pose, ur_tcp_pose=ur_tcp_pose)
                self.surface_controller.set_clutch(True)
                self.teleop_loop.start_surface_positioning(
                    controller=self.surface_controller,
                    on_sample=self._on_loop_sample,
                    on_status=self._post_status,
                )

                def done() -> None:
                    self._show_path_points(path, nearest_index=target.index, point_name=path_point_geometry_name(path))
                    self._refresh_buttons()
                    self._set_status(
                        "Random local start reached "
                        f"index {target.index}; GELLO clutch is active. "
                        "Move GELLO to neutral, release clutch, then move to the recording pose."
                    )

                gui.Application.instance.post_to_main_thread(self.window, done)
            except Exception as exc:
                self._post_status(f"Random local start failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _surface_confirm_log_path(self, prefix: str = "surface_confirm_path") -> Path:
        log_dir = Path("Log")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir / f"{prefix}_{stamp}.jsonl"

    def _write_surface_confirm_log(self, log_path: Path, record: dict[str, Any]) -> None:
        payload = {
            "time": datetime.datetime.now().isoformat(),
            **record,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _move_surface_confirm_stage(self, stage: str, target_pose: np.ndarray, log_path: Path, **move_kwargs) -> dict[str, Any]:
        target = np.asarray(target_pose, dtype=float).reshape(6)
        before = np.asarray(self.devices.get_obs()["ee_pos_rotvec"], dtype=float).reshape(6)
        self._write_surface_confirm_log(
            log_path,
            {
                "kind": "stage_start",
                "stage": stage,
                "target_tcp_pose": target.tolist(),
                "actual_before_tcp_pose": before.tolist(),
            },
        )

        def callback(record: dict[str, Any]) -> None:
            self._write_surface_confirm_log(log_path, {"stage": stage, **record})
            pose = record.get("actual_after_tcp_pose")
            if pose is not None:
                self._post_tcp_pose_preview(np.asarray(pose, dtype=float))

        try:
            obs = self.devices.move_tcp_pose_linear(target, waypoint_callback=callback, **move_kwargs)
        except Exception as exc:
            actual = np.asarray(self.devices.get_obs()["ee_pos_rotvec"], dtype=float).reshape(6)
            self._write_surface_confirm_log(
                log_path,
                {
                    "kind": "stage_error",
                    "stage": stage,
                    "target_tcp_pose": target.tolist(),
                    "actual_tcp_pose": actual.tolist(),
                    "position_error_m": float(np.linalg.norm(actual[:3] - target[:3])),
                    "rotation_error_rad": float(np.linalg.norm(actual[3:] - target[3:])),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        actual = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
        self._write_surface_confirm_log(
            log_path,
            {
                "kind": "stage_done",
                "stage": stage,
                "target_tcp_pose": target.tolist(),
                "actual_tcp_pose": actual.tolist(),
                "position_error_m": float(np.linalg.norm(actual[:3] - target[:3])),
                "rotation_error_rad": float(np.linalg.norm(actual[3:] - target[3:])),
            },
        )
        return obs

    def _post_tcp_pose_preview(self, tcp_pose: np.ndarray) -> None:
        obs = obs_from_tcp_pose_rotvec(tcp_pose)
        path = self.planning.planned_path

        def update() -> None:
            self._update_telemetry(obs)
            try:
                _lines, nearest = probe_path_telemetry_lines(
                    obs,
                    path,
                    probe_tip_offset_m=self.args.probe_tip_offset_m,
                )
                if path is not None and nearest is not None:
                    self._show_path_points(path, nearest_index=nearest, point_name=path_point_geometry_name(path))
            except Exception:
                pass
            self._show_probe_pose(obs, force=True)

        gui.Application.instance.post_to_main_thread(self.window, update)

    def _current_surface_calibration_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        if self.surface_controller is None:
            raise RuntimeError("Click GELLO handover before surface calibration")
        obs = self.devices.get_obs()
        gello_tcp_pose = self.devices.read_gello_tcp_pose(obs)
        ur_tcp_pose = np.asarray(obs["ee_pos_rotvec"], dtype=float).reshape(6)
        return gello_tcp_pose, ur_tcp_pose

    def _on_surface_set_neutral(self) -> None:
        try:
            gello_tcp_pose, ur_tcp_pose = self._current_surface_calibration_inputs()
            assert self.surface_controller is not None
            self.surface_controller.set_clutch(True)
            self.surface_controller.set_neutral(gello_tcp_pose=gello_tcp_pose, ur_tcp_pose=ur_tcp_pose)
            self._refresh_buttons()
            self._set_status("Neutral set. Push GELLO in the desired +X / path-t direction, then click Calibrate +X.")
        except Exception as exc:
            self._set_status(f"Set neutral failed: {type(exc).__name__}: {exc}")

    def _on_surface_calibrate_x(self) -> None:
        try:
            gello_tcp_pose, _ur_tcp_pose = self._current_surface_calibration_inputs()
            assert self.surface_controller is not None
            self.surface_controller.calibrate_x(gello_tcp_pose)
            self._refresh_buttons()
            self._set_status("Calibrated +X. Push GELLO in the desired +Z / surface-normal direction, then click Calibrate +Z.")
        except Exception as exc:
            self._set_status(f"Calibrate +X failed: {type(exc).__name__}: {exc}")

    def _on_surface_calibrate_z(self) -> None:
        try:
            gello_tcp_pose, ur_tcp_pose = self._current_surface_calibration_inputs()
            assert self.surface_controller is not None
            self.surface_controller.calibrate_z(gello_tcp_pose)
            if self.args.operation_mode == "comparison":
                self.surface_controller.set_clutch(True)
                self._refresh_buttons()
                self._set_status(
                    "Calibrated +Z. Return GELLO to the Set-neutral pose while "
                    "UR5 remains still, then click Start 5 cm scan."
                )
                return
            self.surface_controller.recenter(gello_tcp_pose=gello_tcp_pose, ur_tcp_pose=ur_tcp_pose)
            self.surface_controller.set_clutch(True)
            self.teleop_loop.stop()
            self.teleop_loop.start_surface_positioning(
                controller=self.surface_controller,
                on_sample=self._on_loop_sample,
                on_status=self._post_status,
            )
            self._refresh_buttons()
            self._set_status("Calibrated +Z. GELLO clutch is active; move GELLO back to neutral, then click Clutch GELLO to start control.")
        except Exception as exc:
            self._set_status(f"Calibrate +Z failed: {type(exc).__name__}: {exc}")

    def _on_surface_recenter(self) -> None:
        try:
            gello_tcp_pose, ur_tcp_pose = self._current_surface_calibration_inputs()
            assert self.surface_controller is not None
            self.surface_controller.recenter(gello_tcp_pose=gello_tcp_pose, ur_tcp_pose=ur_tcp_pose)
            self.surface_controller.set_clutch(not self.surface_controller.clutch_enabled)
            self._refresh_buttons()
            if self.surface_controller.clutch_enabled:
                self._set_status("GELLO clutch active. Move GELLO back freely; click Clutch GELLO again to resume control.")
            else:
                self._set_status("GELLO clutch released. Continue from the new GELLO pose.")
        except Exception as exc:
            self._set_status(f"Recenter GELLO failed: {type(exc).__name__}: {exc}")

    def _on_start_recording(self) -> None:
        path = self.planning.planned_path
        if path is None:
            self._set_status("Path is not confirmed yet.")
            return
        if not self.devices.connected or self.devices.agent is None:
            self._set_status("GELLO/devices are not connected. Click Connect devices first.")
            return
        if self.stage != GuiStage.TELEOP_READY:
            self._set_status("Cannot record directly. Click GELLO handover first and confirm stable control.")
            return
        if (
            self.args.control_tcp
            and (self.surface_controller is None or not self.surface_controller.input_axes_ready)
        ):
            self._set_status("Calibrate surface input axes first: Set neutral, Calibrate +X, Calibrate +Z.")
            return
        self.teleop_loop.stop()
        if self.args.control_tcp and self.surface_controller is None:
            self.surface_controller = SurfaceCartesianTeleopController(
                path=path,
                probe_length_m=self.args.probe_tip_offset_m,
                translation_gains_xyz=np.full(3, float(self.args.surface_translation_gain), dtype=float),
                rotation_gains_xyz=np.full(3, float(self.args.surface_rotation_gain), dtype=float),
                frame_axis_mode=self.surface_frame_axis_mode,
                use_corner_frame_modes=True,
            )
        self.recorder = EpisodeRecorder(
            data_dir=self.args.data_dir,
            agent_name=self.args.agent,
            planned_path=path,
            probe_tip_offset_m=self.args.probe_tip_offset_m,
            record_rgb_depth=not self.args.skip_rgb_depth_recording,
            episode_context={
                **path_variant_context(path),
                **(self.surface_random_local_context if self.args.control_tcp and self.surface_random_local_context else {}),
            },
        )
        episode_dir = self.recorder.start()
        self._set_stage(GuiStage.RECORDING)
        if self.args.control_tcp:
            assert self.surface_controller is not None
            self.teleop_loop.start_surface_recording(
                controller=self.surface_controller,
                recorder=self.recorder,
                on_sample=self._on_loop_sample,
                on_status=self._post_status,
            )
        else:
            self.teleop_loop.start_recording(
                recorder=self.recorder,
                on_sample=self._on_loop_sample,
                on_status=self._post_status,
            )
        self._set_status(f"Recording episode: {episode_dir}")

    def _on_toggle_fine_scan(self) -> None:
        if self.recorder is None:
            self._set_status("Not recording yet; the flag is saved only during recording.")
            return
        flag = self.recorder.toggle_fine_scan_flag()
        self._set_status(f"Fine-scan flag = {flag}")

    def _on_stop_recording(self) -> None:
        self.teleop_loop.stop()
        if self.recorder is not None:
            self.recorder.stop()
        self.recorder = None
        if self.args.control_tcp and self.surface_controller is not None:
            self._set_stage(GuiStage.TELEOP_READY)
            self.teleop_loop.start_surface_positioning(
                controller=self.surface_controller,
                on_sample=self._on_loop_sample,
                on_status=self._post_status,
            )
            self._set_status("Episode stopped. Surface GELLO control resumed; lift safely or recalibrate before the next random local start.")
            return
        self.surface_controller = None
        self._set_stage(GuiStage.STOPPED)
        self._set_status("Episode stopped. You can start another photo positioning step or safe stop.")

    def _on_safe_stop(self) -> None:
        if self._auto_scan_stop_event is not None:
            self._auto_scan_stop_event.set()
        if self._surface_bo_stop_signal is not None:
            self._surface_bo_stop_signal.request_stop()
        self.teleop_loop.stop()
        self._set_status("Safe stop requested. Moving to safe position before releasing devices ...")

        def worker() -> None:
            error: Exception | None = None
            retreat_warning: Exception | None = None
            try:
                self._wait_for_auto_scan_worker_before_safe_stop()
                self._stop_force_monitor()
                if self.recorder is not None:
                    self.recorder.stop()
                path = self.planning.planned_path
                if self.devices.connected and not self._auto_scan_safe_position_reached:
                    retreat_warning = self._run_safe_position_retreat(path=path, reason="Safe stop")
            except Exception as exc:
                error = exc
            finally:
                self.surface_controller = None
                self.recorder = None
                self._auto_scan_running = False
                self._auto_scan_stop_event = None
                self._auto_scan_pause_event = None
                self._auto_scan_paused_ack_event = None
                self._auto_scan_thread = None
                self._auto_scan_safe_position_reached = False
                self.devices.close()

            def done() -> None:
                self._set_stage(GuiStage.DISCONNECTED)
                if error is None:
                    if retreat_warning is None:
                        self._set_status("Safe stopped at safe joint position. Devices released.")
                    else:
                        self._set_status(
                            "Safe stopped at safe joint position despite TCP retreat warning: "
                            f"{type(retreat_warning).__name__}: {retreat_warning}. Devices released."
                        )
                else:
                    self._set_status(
                        "Safe stop released devices, but safe-position motion failed: "
                        f"{type(error).__name__}: {error}"
                    )

            gui.Application.instance.post_to_main_thread(self.window, done)

        threading.Thread(target=worker, daemon=True).start()

    def _wait_for_auto_scan_worker_before_safe_stop(self) -> None:
        thread = self._auto_scan_thread
        if thread is None or not thread.is_alive() or thread is threading.current_thread():
            return
        self._post_status("Safe stop is waiting for the active auto-scan worker to release robot control ...")
        thread.join(timeout=SAFE_STOP_AUTO_WORKER_JOIN_TIMEOUT_S)
        if thread.is_alive():
            raise TimeoutError(
                "auto-scan worker is still running after "
                f"{SAFE_STOP_AUTO_WORKER_JOIN_TIMEOUT_S:.1f} s; refusing concurrent safe motion"
            )

    def _on_close(self) -> bool:
        if self._auto_scan_stop_event is not None:
            self._auto_scan_stop_event.set()
        if self._surface_bo_stop_signal is not None:
            self._surface_bo_stop_signal.request_stop()
        self.teleop_loop.stop()
        self._stop_force_monitor()
        self.devices.close()
        return True

    def _on_loop_sample(self, sample: dict[str, Any]) -> None:
        obs = sample.get("obs", sample)
        action = sample.get("action")
        meta = sample.get("meta")
        with self._lock:
            self.latest_obs = dict(obs)
            bo_lines = format_surface_bo_status_lines(meta)
            if bo_lines:
                self._last_surface_bo_status_lines = bo_lines
        self._check_comparison_scan_completion(sample)
        now = time.monotonic()
        if self._gui_update_period_s > 0.0 and now - self._last_gui_update_time < self._gui_update_period_s:
            return
        self._last_gui_update_time = now

        def update() -> None:
            enriched_obs = sample.get("enriched_obs")
            self._update_previews_from_obs(obs)
            self._update_telemetry(obs, action, enriched_obs=enriched_obs)
            self._update_path_progress_points(enriched_obs)
            self._show_probe_pose(obs)

        gui.Application.instance.post_to_main_thread(self.window, update)

    def _check_comparison_scan_completion(
        self,
        sample: dict[str, Any],
    ) -> None:
        if (
            self.args.operation_mode != "comparison"
            or self._comparison_finish_pending
            or self.comparison_experiment.active_trial is None
            or self.comparison_experiment.phase != "scan"
        ):
            return
        enriched = sample.get("enriched_obs")
        path = self.planning.planned_path
        pair = self.comparison_experiment.pair
        if enriched is None or path is None or pair is None:
            return
        nearest = int(enriched["path_nearest_index"])
        nearest = int(np.clip(nearest, 0, len(path.positions_base) - 1))
        self._comparison_last_nearest_index = nearest
        arclengths = path_arclengths(path.positions_base)
        if self.comparison_experiment.endpoint_reached(
            enriched["probe_tip_position_base"],
            nearest_arclength_m=float(arclengths[nearest]),
            reference_height_m=float(self.args.surface_contact_height_m),
        ):
            self._request_comparison_finish("reached")
            return
        if (
            self._comparison_scan_started_monotonic is not None
            and time.monotonic() - self._comparison_scan_started_monotonic
            >= float(self.args.comparison_timeout_s)
        ):
            self._request_comparison_finish("timeout")

    def _on_scene_mouse(self, event) -> gui.Widget.EventCallbackResult:
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if not event.is_button_down(gui.MouseButton.LEFT):
            return gui.Widget.EventCallbackResult.IGNORED
        if not event.is_modifier_down(gui.KeyModifier.SHIFT):
            return gui.Widget.EventCallbackResult.IGNORED
        if self.planning.frozen_frame is None:
            self._set_status("No frozen cloud yet.")
            return gui.Widget.EventCallbackResult.HANDLED

        frame = self.scene.frame
        click_xy = (float(event.x - frame.x), float(event.y - frame.y))
        camera = self.scene.scene.camera
        index = pick_nearest_projected_point(
            self.planning.frozen_frame.raw_cloud.points_base,
            click_xy=click_xy,
            view_matrix=np.asarray(camera.get_view_matrix()),
            projection_matrix=np.asarray(camera.get_projection_matrix()),
            width=int(frame.width),
            height=int(frame.height),
            max_pixel_distance=self.args.pick_radius_px,
        )
        if index is None:
            self._set_status("No cloud point selected. Zoom in or click a denser point region.")
            return gui.Widget.EventCallbackResult.HANDLED
        self.seed_index = int(index)
        self._show_seed_marker(self.planning.frozen_frame.raw_cloud.points_base[self.seed_index])
        self._set_stage(GuiStage.SEGMENTED)
        self._set_status(f"Selected seed_index={self.seed_index}. Click Plan path to segment and plan.")
        return gui.Widget.EventCallbackResult.HANDLED

    def _update_previews_from_obs(self, obs: dict[str, Any]) -> None:
        prefix = self.args.wrist_camera
        rgb = obs.get(f"{prefix}_rgb")
        if rgb is None:
            rgb = obs.get("D405_rgb") if prefix != "D405" else obs.get("Orbbec_rgb")
        if rgb is not None:
            self._update_rgb(self.d405_rgb_widget, rgb, max_width=640)
        depth = obs.get(f"{prefix}_depth")
        if depth is None:
            depth = obs.get("D405_depth") if prefix != "D405" else obs.get("Orbbec_depth")
        if depth is not None:
            self._update_depth(depth)
        ultrasound = obs.get("Ultrasound_rgb")
        if ultrasound is None:
            ultrasound = obs.get("Ultrasound_gray")
        if ultrasound is not None:
            self._update_rgb(self.ultrasound_widget, ultrasound, max_width=720)

    def _update_rgb(self, widget: gui.ImageWidget, rgb: np.ndarray, *, max_width: int | None = 640) -> None:
        image = np.asarray(rgb)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 3:
            if max_width is not None and image.shape[1] > max_width:
                scale = float(max_width) / float(image.shape[1])
                size = (int(max_width), max(1, int(round(image.shape[0] * scale))))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            widget.update_image(o3d.geometry.Image(image.astype(np.uint8, copy=False)))

    def _update_depth(self, depth: np.ndarray) -> None:
        self._update_rgb(self.d405_depth_widget, depth_to_display_rgb(depth), max_width=640)

    def _update_telemetry(
        self,
        obs: dict[str, Any],
        action: np.ndarray | None = None,
        enriched_obs: dict[str, Any] | None = None,
    ) -> None:
        lines = []
        self._update_rgb(self.force_widget, force_display_image(obs))
        if "tcp_position_base" in obs and "tcp_z_axis_base" in obs:
            try:
                probe_lines, _nearest = probe_path_telemetry_lines(
                    obs,
                    self.planning.planned_path,
                    probe_tip_offset_m=self.args.probe_tip_offset_m,
                )
                lines.extend(probe_lines)
            except Exception:
                pass
        if action is not None:
            values = np.asarray(action, dtype=float).reshape(-1)
            lines.append("Control q0-q2: " + ", ".join(f"{x:.3f}" for x in values[:3]))
            lines.append("Control q3-q5: " + ", ".join(f"{x:.3f}" for x in values[3:6]))
        if self.recorder is not None:
            display_obs = enriched_obs
            if display_obs is None and self.recorder.episode_dir is not None:
                try:
                    display_obs = self.recorder.enrich_observation(obs)
                except Exception:
                    display_obs = None
            lines.append(f"Episode sample: {self.recorder.sample_index}, fine flag: {self.recorder.fine_scan_flag}")
            if display_obs is not None and "path_nearest_index" in display_obs:
                nearest = int(display_obs["path_nearest_index"])
                total = len(self.recorder.planned_path)
                progress = float(display_obs["path_progress"]) * 100.0
                distance_mm = float(display_obs["path_distance_to_nearest_m"]) * 1000.0
                lines.append(f"Path index: {nearest}/{max(total - 1, 0)}, progress: {progress:.1f}%")
                lines.append(f"Path distance: {distance_mm:.1f} mm")
        if self._last_surface_bo_status_lines:
            lines.extend(self._last_surface_bo_status_lines)
        if not lines:
            lines.append("Telemetry: waiting for data")
        self.telemetry_label.text = "\n".join(lines)

    def _show_cloud(self, name: str, cloud: PointCloud) -> None:
        self.scene.scene.clear_geometry()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points_base)
        if cloud.colors_rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors_rgb, dtype=float) / 255.0)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0
        self.scene.scene.add_geometry(name, pcd, material)
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

    def _show_seed_marker(self, point: np.ndarray) -> None:
        self.scene.scene.remove_geometry("seed_marker")
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.paint_uniform_color([1.0, 0.75, 0.0])
        sphere.compute_vertex_normals()
        sphere.translate(np.asarray(point, dtype=float))
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self.scene.scene.add_geometry("seed_marker", sphere, material)

    def _show_comparison_segment(self, path, pair) -> None:
        for name in (
            "comparison_segment_lines",
            "comparison_segment_endpoints",
        ):
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
        source = np.asarray(path.positions_base, dtype=float)
        points = source[pair.start_index : pair.end_index + 1].copy()
        if not np.allclose(points[-1], pair.end_position_base):
            points = np.vstack([points, pair.end_position_base])
        else:
            points[-1] = pair.end_position_base
        lines = [[index, index + 1] for index in range(len(points) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile([[0.95, 0.55, 0.05]], (len(lines), 1))
        )
        line_material = rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 5.0
        self.scene.scene.add_geometry(
            "comparison_segment_lines",
            line_set,
            line_material,
        )

        endpoints = o3d.geometry.PointCloud()
        endpoints.points = o3d.utility.Vector3dVector(
            np.vstack(
                [
                    pair.start_position_base,
                    pair.end_position_base,
                ]
            )
        )
        endpoints.colors = o3d.utility.Vector3dVector(
            np.asarray(
                [
                    [1.0, 0.05, 0.05],
                    [0.05, 0.35, 1.0],
                ],
                dtype=float,
            )
        )
        point_material = rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = 12.0
        self.scene.scene.add_geometry(
            "comparison_segment_endpoints",
            endpoints,
            point_material,
        )

    def _show_path(self, path, *, name_prefix: str = "planned_path", include_normals: bool = True) -> None:
        positions = np.asarray(path.positions_base, dtype=float)
        line_name = f"{name_prefix}_lines"
        normal_name = f"{name_prefix}_normals"
        point_name = f"{name_prefix}_points"
        for name in (line_name, normal_name, point_name):
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(np.tile(path_display_color(path), (len(lines), 1)))
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 3.0
        self.scene.scene.add_geometry(line_name, line_set, mat)

        if include_normals:
            normal_length = float(self.args.normal_length_m)
            starts = positions
            ends = positions + np.asarray(path.normals_base) * normal_length
            normal_points = np.vstack([starts, ends])
            normal_lines = [[i, i + len(starts)] for i in range(len(starts))]
            normals = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(normal_points),
                lines=o3d.utility.Vector2iVector(normal_lines),
            )
            normals.colors = o3d.utility.Vector3dVector(np.tile([[0.1, 0.25, 1.0]], (len(normal_lines), 1)))
            self.scene.scene.add_geometry(normal_name, normals, mat)
        self._show_path_points(path, point_name=point_name)

    def _show_path_points(
        self,
        path,
        nearest_index: int | None = None,
        *,
        point_name: str = "planned_path_points",
    ) -> None:
        positions = np.asarray(path.positions_base, dtype=float)
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(positions)
        if nearest_index is None:
            colors = path_preview_point_colors(path)
        else:
            colors = path_point_colors(len(positions), nearest_index=nearest_index)
        points.colors = o3d.utility.Vector3dVector(colors)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 9.0
        for stale_point_name in all_path_point_geometry_names():
            if self.scene.scene.has_geometry(stale_point_name):
                self.scene.scene.remove_geometry(stale_point_name)
        self.scene.scene.add_geometry(point_name, points, material)

    def _update_path_progress_points(self, enriched_obs: dict[str, Any] | None) -> None:
        path = self.planning.planned_path
        if path is None or enriched_obs is None or "path_nearest_index" not in enriched_obs:
            return
        self._show_path_points(
            path,
            nearest_index=int(enriched_obs["path_nearest_index"]),
            point_name=path_point_geometry_name(path),
        )

    def _show_probe_pose(self, obs: dict[str, Any], *, force: bool = False) -> None:
        if "tcp_position_base" not in obs:
            return
        now = time.monotonic()
        if not force and now - self._last_scene_pose_update_time < self._scene_pose_update_period_s:
            return
        self._last_scene_pose_update_time = now
        try:
            probe_obs = add_probe_tip_observation(obs, self.args.probe_tip_offset_m)
        except Exception:
            return
        point = np.asarray(probe_obs["probe_tip_position_base"], dtype=float)
        axes = np.asarray(
            [
                probe_obs["probe_x_axis_base"],
                probe_obs["probe_y_axis_base"],
                probe_obs["probe_z_axis_base"],
            ],
            dtype=float,
        )
        length = float(self.args.probe_axis_length_m)
        points = np.vstack([point, point + axes[0] * length, point + axes[1] * length, point + axes[2] * length])
        lines = [[0, 1], [0, 2], [0, 3]]
        geom = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        geom.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 0.7, 0], [0, 0, 1]])
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 4.0
        if self.scene.scene.has_geometry("probe_axes"):
            self.scene.scene.remove_geometry("probe_axes")
        self.scene.scene.add_geometry("probe_axes", geom, mat)

    def run(self) -> None:
        gui.Application.instance.run()
