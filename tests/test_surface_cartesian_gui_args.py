from visual_guided_collection_gui.main import build_parser
from visual_guided_collection_gui.state import GuiStage, enabled_actions_for_stage


def test_surface_cartesian_teleop_is_opt_in_from_gui_args():
    legacy = build_parser().parse_args([])
    surface = build_parser().parse_args(["--surface-cartesian-teleop"])

    assert legacy.surface_cartesian_teleop is False
    assert surface.surface_cartesian_teleop is True


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

    assert defaults.surface_translation_gain == 0.5
    assert defaults.surface_rotation_gain == 0.8
    assert custom.surface_translation_gain == 0.2
    assert custom.surface_rotation_gain == 0.15


def test_surface_calibration_buttons_are_available_after_handover_stage():
    actions = enabled_actions_for_stage(GuiStage.TELEOP_READY)

    assert "surface_set_neutral" in actions
    assert "surface_calibrate_x" in actions
    assert "surface_calibrate_z" in actions
    assert "surface_recenter" in actions


def test_surface_darboux_preview_is_available_after_path_confirmed():
    actions = enabled_actions_for_stage(GuiStage.PATH_CONFIRMED)

    assert "surface_preview_darboux_line" in actions
