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
    GuiStage.SEGMENTED: {"resegment", "plan_path", "safe_stop"},
    GuiStage.PATH_PLANNED: {"confirm_path", "resegment", "safe_stop"},
    GuiStage.PATH_CONFIRMED: {"start_gello_handover", "surface_preview_darboux_line", "safe_stop"},
    GuiStage.TELEOP_READY: {
        "start_recording",
        "toggle_fine_scan",
        "surface_random_local_start",
        "surface_set_neutral",
        "surface_calibrate_x",
        "surface_calibrate_z",
        "surface_recenter",
        "safe_stop",
    },
    GuiStage.RECORDING: {"stop_recording", "toggle_fine_scan", "surface_recenter", "safe_stop"},
    GuiStage.STOPPED: {"start_photo_positioning", "safe_stop"},
}


def enabled_actions_for_stage(stage: GuiStage) -> set[str]:
    return set(_ENABLED_ACTIONS[stage])
