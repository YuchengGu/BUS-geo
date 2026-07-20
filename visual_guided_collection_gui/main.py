from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_T_TCP_CAMERA_PATHS = {
    "D405": (
        Path("hand_eye_calibration")
        / "results_0512_222937_calib_11x8_stride10"
        / "T_tcp_camera.npy"
    ),
    "Orbbec": (
        Path("hand_eye_calibration")
        / "Results_Orbbec"
        / "T_tcp_camera.npy"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open3D GUI for visual guided GELLO data collection.")
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--gui-update-hz", type=float, default=30.0)
    parser.add_argument("--agent", default="gello", choices=["gello", "dummy", "none"])
    parser.add_argument(
        "--operation-mode",
        default="demo",
        choices=["demo", "auto", "comparison", "bo"],
    )
    parser.add_argument("--wrist-camera", default="Orbbec", choices=["D405", "Orbbec"])
    parser.add_argument("--gello-port", default=None)
    parser.add_argument("--force-ip", default="192.168.1.100")
    parser.add_argument("--force-gravity-calib", default=None)
    parser.add_argument("--disable-force", action="store_true")
    parser.add_argument("--disable-ultrasound", action="store_true")
    parser.add_argument(
        "--skip-rgb-depth-recording",
        action="store_true",
        help="Keep cameras active for GUI/planning but omit *_rgb and *_depth arrays from recorded PKL files.",
    )
    parser.add_argument("--ultrasound-index", type=int, default=4)
    parser.add_argument("--data-dir", default="~/bc_data")
    parser.add_argument("--t-tcp-camera", "--T-tcp-camera", dest="t_tcp_camera", default=None)
    parser.add_argument("--planning-output-root", default="breast_path_planning/results")
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--capture-settle-s", type=float, default=0.5)
    parser.add_argument("--pick-radius-px", type=float, default=14.0)
    parser.add_argument("--normal-length-m", type=float, default=0.02)
    parser.add_argument("--probe-tip-offset-m", type=float, default=0.204)
    parser.add_argument("--probe-axis-length-m", type=float, default=0.04)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.0)
    parser.add_argument("--control-tcp", action="store_true")
    parser.add_argument("--surface-cartesian-teleop", dest="control_tcp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--surface-random-local-episodes", action="store_true")
    parser.add_argument("--surface-approach-height-m", type=float, default=0.07)
    parser.add_argument("--surface-contact-height-m", type=float, default=0.02)
    parser.add_argument("--surface-random-start-height-m", type=float, default=0.02)
    parser.add_argument("--surface-translation-gain", type=float, default=0.25)
    parser.add_argument("--surface-rotation-gain", type=float, default=1.0)
    parser.add_argument("--comparison-segment-length-m", type=float, default=0.05)
    parser.add_argument("--comparison-endpoint-radius-m", type=float, default=0.011)
    parser.add_argument("--comparison-progress-tolerance-m", type=float, default=0.003)
    parser.add_argument("--comparison-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--comparison-joint-handover-tolerance-rad",
        type=float,
        default=0.1,
    )
    parser.add_argument("--comparison-random-seed", type=int, default=0)
    parser.add_argument(
        "--surface-bo-bounds",
        default="dn=-0.011,0.011;rx=-0.034,0.034;ry=-0.034,0.034;rz=-0.034,0.034",
    )
    parser.add_argument("--surface-bo-n-initial", type=int, default=3)
    parser.add_argument("--surface-bo-n-ei", type=int, default=15)
    parser.add_argument("--surface-bo-search-strategy", choices=["bo", "random", "lhs", "uniform"], default="bo")
    parser.add_argument(
        "--surface-bo-objective-variant",
        choices=["full", "no_penalty", "force_only", "torque_only"],
        default="full",
    )
    parser.add_argument("--surface-bo-settle-s", type=float, default=0.2)
    parser.add_argument("--surface-bo-pressure-min", type=float, default=2.0)
    parser.add_argument("--surface-bo-pressure-max", type=float, default=8.0)
    parser.add_argument("--surface-bo-shear-max", type=float, default=6.0)
    parser.add_argument("--surface-bo-torque-tangential-max", type=float, default=0.8)
    parser.add_argument("--surface-bo-torque-axial-max", type=float, default=0.5)
    parser.add_argument("--surface-bo-random-seed", type=int, default=0)
    parser.add_argument("--surface-bo-lambda-pressure", type=float, default=0.11)
    parser.add_argument("--surface-bo-lambda-shear", type=float, default=0.04)
    parser.add_argument("--surface-bo-lambda-torque", type=float, default=0.08)
    parser.add_argument("--surface-bo-lambda-axial-torque", type=float, default=0.02)
    parser.add_argument("--surface-bo-force-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--surface-bo-torque-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--surface-bo-lambda-force", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--surface-bo-large-penalty", type=float, default=1000.0)
    parser.add_argument("--surface-bo-post-run-wait-s", type=float, default=1.0)
    parser.add_argument("--surface-bo-reset-retreat-m", type=float, default=0.15)
    parser.add_argument(
        "--surface-bo-ultrasound-crop",
        default="99,769,542,1524",
        help="Crop live ultrasound before BO quality scoring as row0,row1,col0,col1; use 'none' for full frame.",
    )
    parser.add_argument("--auto-scan-safe-retreat", action="store_true")
    parser.add_argument("--auto-scan-retreat-distance-m", type=float, default=0.15)
    parser.add_argument(
        "--auto-scan-safe-joint-degrees",
        type=float,
        nargs=6,
        default=[-90.0, -90.0, -90.0, -90.0, 90.0, 30.0],
    )
    parser.add_argument("--auto-scan-safe-joint-step-rad", type=float, default=0.01)
    parser.add_argument("--auto-scan-safe-retreat-timeout-s", type=float, default=90.0)
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.operation_mode in {"comparison", "bo"}:
        args.control_tcp = True
    args.surface_cartesian_teleop = bool(args.control_tcp)
    if args.t_tcp_camera is None:
        args.t_tcp_camera = str(DEFAULT_T_TCP_CAMERA_PATHS[args.wrist_camera])
    return args


def main() -> None:
    from visual_guided_collection_gui.app import VisualGuidedCollectionApp

    args = resolve_args(build_parser().parse_args())
    VisualGuidedCollectionApp(args).run()


if __name__ == "__main__":
    main()
