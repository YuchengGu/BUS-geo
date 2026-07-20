from visual_guided_collection_gui.main import build_parser, resolve_args
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
            "--surface-bo-search-strategy",
            "random",
            "--surface-bo-objective-variant",
            "no_penalty",
        ]
    )
    lhs = build_parser().parse_args(["--surface-bo-search-strategy", "lhs"])

    assert defaults.operation_mode == "demo"
    assert auto.operation_mode == "auto"
    assert auto.surface_bo_n_initial == 3
    assert auto.surface_bo_n_ei == 12
    assert auto.surface_bo_search_strategy == "random"
    assert auto.surface_bo_objective_variant == "no_penalty"
    assert lhs.surface_bo_search_strategy == "lhs"
    assert auto.surface_bo_settle_s == 0.2
    assert auto.surface_bo_post_run_wait_s == 1.0
    assert auto.surface_bo_reset_retreat_m == 0.15
    assert auto.surface_bo_pressure_min == 2.0
    assert auto.surface_bo_pressure_max == 8.0
    assert auto.surface_bo_shear_max == 6.0
    assert auto.surface_bo_torque_tangential_max == 0.8
    assert auto.surface_bo_torque_axial_max == 0.5
    assert auto.surface_bo_lambda_pressure == 0.11
    assert auto.surface_bo_lambda_shear == 0.04
    assert auto.surface_bo_lambda_torque == 0.08
    assert auto.surface_bo_lambda_axial_torque == 0.02
    assert auto.surface_bo_ultrasound_crop == "99,769,542,1524"


def test_bo_only_mode_enables_tcp_control_without_auto_scan():
    args = resolve_args(build_parser().parse_args(["--operation-mode", "bo"]))

    assert args.operation_mode == "bo"
    assert args.control_tcp is True


def test_bo_only_path_point_actions_are_available_after_path_planning():
    segmented_actions = enabled_actions_for_stage(GuiStage.SEGMENTED)
    planned_actions = enabled_actions_for_stage(GuiStage.PATH_PLANNED)

    assert "surface_bo_select_point" not in segmented_actions
    assert "surface_bo_confirm_point" not in segmented_actions
    assert "surface_bo_select_point" in planned_actions
    assert "surface_bo_confirm_point" in planned_actions


def test_comparison_mode_enables_shared_surface_infrastructure():
    from visual_guided_collection_gui.main import resolve_args

    args = resolve_args(
        build_parser().parse_args(
            [
                "--operation-mode",
                "comparison",
                "--comparison-segment-length-m",
                "0.05",
            ]
        )
    )

    assert args.operation_mode == "comparison"
    assert args.control_tcp is True
    assert args.comparison_segment_length_m == 0.05
    assert args.comparison_endpoint_radius_m == 0.011
    assert args.comparison_progress_tolerance_m == 0.003
    assert args.comparison_timeout_s == 60.0
    assert args.comparison_joint_handover_tolerance_rad == 0.1


def test_comparison_actions_are_available_in_relevant_stages():
    confirmed = enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    ready = enabled_actions_for_stage(GuiStage.TELEOP_READY)
    recording = enabled_actions_for_stage(GuiStage.RECORDING)

    assert {
        "comparison_generate_pair",
        "comparison_confirm_participant",
        "comparison_full_joint",
        "comparison_darboux",
        "comparison_finish_participant",
    } <= confirmed
    assert {
        "comparison_arm_direct",
        "comparison_start_scan",
    } <= ready
    assert "comparison_finish_trial" in recording


def test_auto_scan_safe_retreat_is_opt_in_from_cli():
    defaults = build_parser().parse_args([])
    enabled = build_parser().parse_args(["--auto-scan-safe-retreat"])

    assert defaults.auto_scan_safe_retreat is False
    assert enabled.auto_scan_safe_retreat is True
    assert defaults.auto_scan_retreat_distance_m == 0.15
    assert defaults.auto_scan_safe_joint_degrees == [-90.0, -90.0, -90.0, -90.0, 90.0, 30.0]


def test_rgb_depth_recording_can_be_disabled_without_disabling_camera():
    defaults = build_parser().parse_args([])
    compact = build_parser().parse_args(["--skip-rgb-depth-recording"])

    assert defaults.skip_rgb_depth_recording is False
    assert compact.skip_rgb_depth_recording is True
    assert compact.wrist_camera == "Orbbec"


def test_surface_bo_buttons_are_available_after_path_confirmation_and_while_recording():
    assert "surface_bo_optimize" in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "surface_bo_optimize" in enabled_actions_for_stage(GuiStage.RECORDING)
    assert "surface_bo_stop" in enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)
    assert "surface_bo_stop" in enabled_actions_for_stage(GuiStage.RECORDING)


def test_surface_bo_experiment_condition_buttons_are_available_after_path_confirmation():
    confirmed = enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)

    assert {
        "surface_bo_run_full",
        "surface_bo_run_no_penalty",
        "surface_bo_run_force_only",
        "surface_bo_run_torque_only",
        "surface_bo_run_random",
        "surface_bo_run_uniform",
    } <= confirmed
