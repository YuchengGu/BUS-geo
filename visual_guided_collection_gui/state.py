from __future__ import annotations

from enum import Enum


class GuiStage(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    POSITIONING = "positioning"
    FRAME_CAPTURED = "frame_captured"
    SEGMENTED = "segmented"
    PATH_PLANNED = "path_planned"
    PATH_CONFIRMED = "path_confirmed"
    TELEOP_READY = "teleop_ready"
    RECORDING = "recording"
    STOPPED = "stopped"


_ENABLED_ACTIONS: dict[GuiStage, set[str]] = {
    GuiStage.DISCONNECTED: {"connect"},
    GuiStage.CONNECTED: {"start_photo_positioning", "safe_stop"},
    GuiStage.POSITIONING: {"capture_frame", "safe_stop"},
    GuiStage.FRAME_CAPTURED: {"recapture", "safe_stop"},
    GuiStage.SEGMENTED: {
        "resegment",
        "plan_path",
        "safe_stop",
    },
    GuiStage.PATH_PLANNED: {
        "use_original_path",
        "smooth_moving_average",
        "smooth_b_spline",
        "optimize_geodesic",
        "surface_bo_select_point",
        "surface_bo_confirm_point",
        "confirm_path",
        "resegment",
        "safe_stop",
    },
    GuiStage.PATH_CONFIRMED: {
        "use_original_path",
        "smooth_moving_average",
        "smooth_b_spline",
        "optimize_geodesic",
        "start_gello_handover",
        "surface_auto_scan_start",
        "surface_auto_scan_stop",
        "surface_bo_select_point",
        "surface_bo_confirm_point",
        "surface_bo_optimize",
        "surface_bo_run_full",
        "surface_bo_run_no_penalty",
        "surface_bo_run_force_only",
        "surface_bo_run_torque_only",
        "surface_bo_run_random",
        "surface_bo_run_uniform",
        "surface_bo_stop",
        "comparison_generate_pair",
        "comparison_confirm_participant",
        "comparison_full_joint",
        "comparison_darboux",
        "comparison_finish_participant",
        "safe_stop",
    },
    GuiStage.TELEOP_READY: {
        "start_recording",
        "toggle_fine_scan",
        "surface_random_local_start",
        "surface_set_neutral",
        "surface_calibrate_x",
        "surface_calibrate_z",
        "surface_recenter",
        "comparison_arm_direct",
        "comparison_start_scan",
        "safe_stop",
    },
    GuiStage.RECORDING: {
        "stop_recording",
        "toggle_fine_scan",
        "surface_recenter",
        "surface_auto_scan_stop",
        "surface_bo_optimize",
        "surface_bo_stop",
        "comparison_finish_trial",
        "safe_stop",
    },
    GuiStage.STOPPED: {"start_photo_positioning", "safe_stop"},
}


def enabled_actions_for_stage(stage: GuiStage) -> set[str]:
    return set(_ENABLED_ACTIONS[stage])
