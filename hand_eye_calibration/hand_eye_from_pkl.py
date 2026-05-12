#!/usr/bin/env python3
"""Compute D405 eye-in-hand calibration from GELLO pkl episodes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np


def load_pkl(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def rotvec_pose_to_transform(ee_pos_rotvec: np.ndarray) -> np.ndarray:
    pose = np.asarray(ee_pos_rotvec, dtype=float).reshape(-1)
    if pose.shape[0] < 6:
        raise ValueError(f"ee_pos_rotvec must have 6 values, got {pose.shape}")
    rotation, _ = cv2.Rodrigues(pose[3:6].reshape(3, 1))
    return make_transform(rotation, pose[:3])


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def chessboard_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size_m)
    return points


def is_usable_frame(data: dict[str, Any], require_new_frame: bool) -> bool:
    if "D405_rgb" not in data or "ee_pos_rotvec" not in data:
        return False
    meta = data.get("meta", {})
    d405_meta = meta.get("modalities", {}).get("D405", {})
    if d405_meta.get("valid") is False:
        return False
    if require_new_frame and d405_meta.get("frame_new") is not True:
        return False
    return True


def find_chessboard(rgb: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if not ok:
        return False, None
    return True, corners.astype(np.float32)


def reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    delta = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def save_detection_sheet(
    detections: list[dict[str, Any]],
    output_path: Path,
    pattern_size: tuple[int, int],
    max_images: int = 24,
) -> None:
    thumbs = []
    for detection in detections[:max_images]:
        rgb = detection["rgb"]
        corners = detection["corners"]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(bgr, pattern_size, corners, True)
        thumb = cv2.resize(bgr, (212, 120))
        label = f"i={detection['sample_index']} f={detection['frame_id']}"
        cv2.putText(
            thumb,
            label,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    if not thumbs:
        return

    rows = []
    for i in range(0, len(thumbs), 4):
        row = thumbs[i : i + 4]
        while len(row) < 4:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_path), np.vstack(rows))


def collect_detections(args: argparse.Namespace) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    episode_dir = Path(args.episode_dir)
    paths = sorted(episode_dir.glob("*.pkl"))
    pattern_size = (args.board_cols, args.board_rows)
    detections: list[dict[str, Any]] = []

    for path in paths[:: args.stride]:
        data = load_pkl(path)
        if not is_usable_frame(data, args.require_new_frame):
            continue

        ok, corners = find_chessboard(data["D405_rgb"], pattern_size)
        if not ok:
            continue

        meta = data.get("meta", {})
        d405_meta = meta.get("modalities", {}).get("D405", {})
        detections.append(
            {
                "path": str(path),
                "rgb": data["D405_rgb"],
                "corners": corners,
                "ee_pos_rotvec": np.asarray(data["ee_pos_rotvec"], dtype=float),
                "sample_index": meta.get("sample_index"),
                "frame_id": d405_meta.get("frame_id"),
            }
        )
        if args.max_frames is not None and len(detections) >= args.max_frames:
            break

    return detections, pattern_size


def calibrate_from_detections(
    detections: list[dict[str, Any]],
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> dict[str, Any]:
    object_points_single = chessboard_object_points(pattern_size, square_size_m)
    object_points = [
        object_points_single.reshape(-1, 1, 3).astype(np.float32)
        for _ in detections
    ]
    image_points = [
        np.asarray(detection["corners"], dtype=np.float32).reshape(-1, 1, 2)
        for detection in detections
    ]
    image_size = detections[0]["rgb"].shape[1], detections[0]["rgb"].shape[0]

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(  # type: ignore[reportCallIssue]
        cast(Any, object_points),
        cast(Any, image_points),
        image_size,
        None,
        None,
    )

    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []
    reprojection_errors = []
    t_base_tcp_list = []
    t_camera_board_list = []

    for detection in detections:
        image_corners = np.asarray(detection["corners"], dtype=np.float32).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(  # type: ignore[reportCallIssue]
            object_points_single,
            image_corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        r_board, _ = cv2.Rodrigues(rvec)
        t_board = tvec.reshape(3, 1)
        t_base_tcp = rotvec_pose_to_transform(detection["ee_pos_rotvec"])

        r_gripper2base.append(t_base_tcp[:3, :3])
        t_gripper2base.append(t_base_tcp[:3, 3].reshape(3, 1))
        r_target2cam.append(r_board)
        t_target2cam.append(t_board)
        t_base_tcp_list.append(t_base_tcp)
        t_camera_board_list.append(make_transform(r_board, t_board))
        reprojection_errors.append(
            reprojection_error_px(
                object_points_single,
                image_corners,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
        )

    if len(r_gripper2base) < 3:
        raise RuntimeError(f"Need at least 3 valid PnP frames, got {len(r_gripper2base)}")

    r_tcp_camera, t_tcp_camera = cv2.calibrateHandEye(  # type: ignore[reportCallIssue]
        cast(Any, r_gripper2base),
        cast(Any, t_gripper2base),
        cast(Any, r_target2cam),
        cast(Any, t_target2cam),
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    t_tcp_camera = make_transform(r_tcp_camera, t_tcp_camera)

    t_base_board_list = [
        t_base_tcp @ t_tcp_camera @ t_camera_board
        for t_base_tcp, t_camera_board in zip(t_base_tcp_list, t_camera_board_list)
    ]
    board_translations = np.array([t[:3, 3] for t in t_base_board_list])
    translation_center = np.mean(board_translations, axis=0)
    translation_errors_mm = np.linalg.norm(board_translations - translation_center, axis=1) * 1000.0
    r_ref = t_base_board_list[0][:3, :3]
    rotation_errors_deg = np.array(
        [rotation_angle_deg(r_ref.T @ t[:3, :3]) for t in t_base_board_list],
        dtype=float,
    )

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "camera_calibration_rms_px": float(rms),
        "T_tcp_camera": t_tcp_camera,
        "reprojection_error_px": reprojection_errors,
        "validation": {
            "num_pnp_frames": len(r_gripper2base),
            "base_board_translation_mean_m": translation_center.tolist(),
            "base_board_translation_error_mm": {
                "mean": float(np.mean(translation_errors_mm)),
                "p50": float(np.percentile(translation_errors_mm, 50)),
                "p95": float(np.percentile(translation_errors_mm, 95)),
                "max": float(np.max(translation_errors_mm)),
            },
            "base_board_rotation_error_deg_vs_first": {
                "mean": float(np.mean(rotation_errors_deg)),
                "p50": float(np.percentile(rotation_errors_deg, 50)),
                "p95": float(np.percentile(rotation_errors_deg, 95)),
                "max": float(np.max(rotation_errors_deg)),
            },
            "reprojection_error_px": {
                "mean": float(np.mean(reprojection_errors)),
                "p50": float(np.percentile(reprojection_errors, 50)),
                "p95": float(np.percentile(reprojection_errors, 95)),
                "max": float(np.max(reprojection_errors)),
            },
        },
    }


def write_outputs(
    output_dir: Path,
    detections: list[dict[str, Any]],
    result: dict[str, Any] | None,
    args: argparse.Namespace,
    pattern_size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_detection_sheet(detections, output_dir / "detected_chessboards.jpg", pattern_size)

    selected_frames = [
        {
            "path": detection["path"],
            "sample_index": detection["sample_index"],
            "frame_id": detection["frame_id"],
        }
        for detection in detections
    ]
    with open(output_dir / "selected_frames.json", "w", encoding="utf-8") as f:
        json.dump(selected_frames, f, indent=2)

    config = {
        "episode_dir": str(args.episode_dir),
        "board_inner_corners": [args.board_cols, args.board_rows],
        "square_size_m": args.square_size_m,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "require_new_frame": args.require_new_frame,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if result is None:
        return

    np.save(output_dir / "T_tcp_camera.npy", result["T_tcp_camera"])
    np.savez(
        output_dir / "camera_intrinsics.npz",
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"],
    )

    report = {
        "T_tcp_camera": result["T_tcp_camera"].tolist(),
        "camera_matrix": result["camera_matrix"].tolist(),
        "dist_coeffs": result["dist_coeffs"].reshape(-1).tolist(),
        "camera_calibration_rms_px": result["camera_calibration_rms_px"],
        "validation": result["validation"],
    }
    with open(output_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "calibration_report.md", "w", encoding="utf-8") as f:
        f.write("# D405 Hand-Eye Calibration Report\n\n")
        f.write(f"- episode: `{args.episode_dir}`\n")
        f.write(f"- selected frames: {len(detections)}\n")
        f.write(f"- board inner corners: {args.board_cols} x {args.board_rows}\n")
        f.write(f"- square size: {args.square_size_m} m\n")
        f.write(f"- camera calibration RMS: {result['camera_calibration_rms_px']:.4f} px\n\n")
        f.write("## T_tcp_camera\n\n")
        f.write("```text\n")
        f.write(np.array2string(result["T_tcp_camera"], precision=8, suppress_small=True))
        f.write("\n```\n\n")
        f.write("## Validation\n\n")
        f.write("```json\n")
        f.write(json.dumps(result["validation"], indent=2))
        f.write("\n```\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a chessboard in GELLO pkl frames and compute D405 T_tcp_camera."
    )
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", default="hand_eye_calibration/results")
    parser.add_argument("--board-cols", type=int, default=8, help="Number of inner corners along board columns.")
    parser.add_argument("--board-rows", type=int, default=8, help="Number of inner corners along board rows.")
    parser.add_argument("--square-size-m", type=float, default=None, help="Chessboard square size in meters.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--require-new-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detections, pattern_size = collect_detections(args)
    output_dir = Path(args.output_dir)

    print(f"Detected chessboard in {len(detections)} frames.")
    if len(detections) < 3:
        write_outputs(output_dir, detections, None, args, pattern_size)
        raise SystemExit("Need at least 3 detected frames.")

    if args.detect_only:
        write_outputs(output_dir, detections, None, args, pattern_size)
        print(f"Detection outputs written to {output_dir}")
        return

    if args.square_size_m is None:
        write_outputs(output_dir, detections, None, args, pattern_size)
        raise SystemExit("--square-size-m is required unless --detect-only is used.")

    result = calibrate_from_detections(detections, pattern_size, args.square_size_m)
    write_outputs(output_dir, detections, result, args, pattern_size)
    print(f"T_tcp_camera written to {output_dir / 'T_tcp_camera.npy'}")
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
