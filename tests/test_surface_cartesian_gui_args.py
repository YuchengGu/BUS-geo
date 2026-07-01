from visual_guided_collection_gui.main import build_parser
from visual_guided_collection_gui.state import GuiStage, enabled_actions_for_stage


def test_surface_cartesian_teleop_is_opt_in_from_gui_args():
    legacy = build_parser().parse_args([])
    surface = build_parser().parse_args(["--control-tcp"])
    old_alias = build_parser().parse_args(["--surface-cartesian-teleop"])

    assert legacy.control_tcp is False
    assert surface.control_tcp is True
    assert old_alias.control_tcp is True


def test_surface_cartesian_gains_have_slow_defaults_and_can_be_overridden():
    defaults = build_parser().parse_args([])
    custom = build_parser().parse_args(
        [
            "--surface-translation-gain",
            "0.2",
            "--surface-rotation-gain",
            "0.15",
        ]
    )

    assert defaults.surface_translation_gain == 0.25
    assert defaults.surface_rotation_gain == 1.0
    assert custom.surface_translation_gain == 0.2
    assert custom.surface_rotation_gain == 0.15


def test_surface_calibration_buttons_are_available_after_handover_stage():
    actions = enabled_actions_for_stage(GuiStage.TELEOP_READY)

    assert "surface_set_neutral" in actions
    assert "surface_calibrate_x" in actions
    assert "surface_calibrate_z" in actions
    assert "surface_recenter" in actions


def test_surface_auto_scan_is_available_after_path_confirmed():
    actions = enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)

    assert "surface_auto_scan_start" in actions
    assert "surface_auto_scan_stop" in actions


def test_geodesic_optimization_is_available_after_path_planned_before_confirm():
    planned_actions = enabled_actions_for_stage(GuiStage.PATH_PLANNED)
    confirmed_actions = enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)

    assert "use_original_path" in planned_actions
    assert "smooth_moving_average" in planned_actions
    assert "smooth_b_spline" in planned_actions
    assert "optimize_geodesic" in planned_actions
    assert "use_original_path" in confirmed_actions
    assert "smooth_moving_average" in confirmed_actions
    assert "smooth_b_spline" in confirmed_actions
    assert "optimize_geodesic" in confirmed_actions


def test_operation_mode_and_surface_bo_are_selected_from_cli():
    defaults = build_parser().parse_args([])
    auto = build_parser().parse_args(
        [
            "--operation-mode",
            "auto",
            "--surface-bo-bounds",
            "dn=-0.05,0.05;rx=-0.0873,0.0873;ry=-0.0873,0.0873;rz=-0.0873,0.0873",
            "--surface-bo-n-initial",
            "3",
            "--surface-bo-n-ei",
            "12",
        ]
    )

    assert defaults.operation_mode == "demo"
    assert auto.operation_mode == "auto"
    assert auto.surface_bo_n_initial == 3
    assert auto.surface_bo_n_ei == 12


def test_auto_scan_safe_retreat_is_opt_in_from_cli():
    defaults = build_parser().parse_args([])
    enabled = build_parser().parse_args(["--auto-scan-safe-retreat"])

    assert defaults.auto_scan_safe_retreat is False
    assert enabled.auto_scan_safe_retreat is True
    assert defaults.auto_scan_retreat_distance_m == 0.15
    assert defaults.auto_scan_safe_joint_degrees == [-90.0, -90.0, -90.0, -90.0, 90.0, 60.0]


def test_surface_bo_buttons_are_available_after_path_confirmation_and_while_recording():
    assert "surface_bo_optimize" in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "surface_bo_optimize" in enabled_actions_for_stage(GuiStage.RECORDING)
    assert "surface_bo_stop" in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "surface_bo_stop" in enabled_actions_for_stage(GuiStage.RECORDING)
