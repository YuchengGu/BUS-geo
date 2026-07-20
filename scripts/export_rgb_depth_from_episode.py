#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_guided_collection_gui.images import depth_to_display_rgb
from visual_guided_collection_gui.planning_session import _write_depth_png, _write_rgb_png


def load_frame(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        frame = pickle.load(f)
    if not isinstance(frame, dict):
        raise TypeError(f"Expected a dict frame in {path}, got {type(frame).__name__}")
    return frame


def choose_pkl(episode_dir: Path, frame_index: int) -> Path:
    pkls = sorted(episode_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No *.pkl files found under {episode_dir}")
    if frame_index < 0 or frame_index >= len(pkls):
        raise IndexError(f"frame-index {frame_index} is outside [0, {len(pkls) - 1}]")
    return pkls[frame_index]


def find_latest_frame_with_rgb_depth(data_root: Path, camera: str | None) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    episode_dirs = sorted(
        [path for path in data_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for episode_dir in episode_dirs:
        pkls = sorted(episode_dir.glob("*.pkl"))
        for frame_path in pkls:
            try:
                frame = load_frame(frame_path)
                find_rgb_depth_keys(frame, camera)
            except Exception:
                continue
            return frame_path
    raise FileNotFoundError(
        f"No frame with matching RGB/depth keys found under {data_root}. "
        "Episodes recorded with --skip-rgb-depth-recording cannot provide RGB/depth."
    )


def find_rgb_depth_keys(frame: dict[str, Any], camera: str | None) -> tuple[str, str]:
    if camera:
        candidates = [(f"{camera}_rgb", f"{camera}_depth")]
    else:
        prefixes = sorted(
            {
                key[: -len("_rgb")]
                for key in frame
                if key.endswith("_rgb") and f"{key[: -len('_rgb')]}_depth" in frame
            }
        )
        candidates = [(f"{prefix}_rgb", f"{prefix}_depth") for prefix in prefixes]
    for rgb_key, depth_key in candidates:
        if rgb_key in frame and depth_key in frame:
            return rgb_key, depth_key
    image_keys = [key for key in sorted(frame) if any(s in key.lower() for s in ("rgb", "depth", "image"))]
    raise KeyError(
        "No matching *_rgb/*_depth pair found. "
        f"Available image-like keys: {image_keys}. "
        "If this episode was recorded with --skip-rgb-depth-recording, RGB/depth cannot be recovered from it."
    )


def export_rgb_depth(
    *,
    frame_path: Path,
    output_dir: Path,
    camera: str | None,
) -> None:
    frame = load_frame(frame_path)
    rgb_key, depth_key = find_rgb_depth_keys(frame, camera)
    rgb = np.asarray(frame[rgb_key])
    depth = np.asarray(frame[depth_key])
    depth_vis = depth_to_display_rgb(depth)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb_png(output_dir / f"{rgb_key}.png", rgb)
    _write_rgb_png(output_dir / f"{depth_key}_vis.png", depth_vis)
    _write_depth_png(output_dir / f"{depth_key}_raw.png", depth)
    np.save(output_dir / f"{depth_key}_raw.npy", depth)
    print(f"frame: {frame_path}")
    print(f"rgb: {rgb_key}, shape={rgb.shape}, dtype={rgb.dtype}")
    print(f"depth: {depth_key}, shape={depth.shape}, dtype={depth.dtype}")
    print(f"saved under: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RGB and depth renderings from a recorded episode PKL frame.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--episode-dir", type=Path, help="Episode directory containing timestamped *.pkl frames.")
    source.add_argument("--pkl", type=Path, help="A specific frame PKL file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ubuntu22/bc_data/gello"),
        help="Used only when neither --episode-dir nor --pkl is provided. The latest frame with RGB/depth is exported.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index in --episode-dir, sorted by filename.")
    parser.add_argument("--camera", default=None, help="Camera prefix, e.g. Orbbec, D405, Ultrasound. Default: auto.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.pkl is not None:
        frame_path = args.pkl
    elif args.episode_dir is not None:
        frame_path = choose_pkl(args.episode_dir, args.frame_index)
    else:
        frame_path = find_latest_frame_with_rgb_depth(args.data_root, args.camera)
    output_dir = args.output_dir
    if output_dir is None:
        base = frame_path.parent
        output_dir = base / "export_rgb_depth"
    export_rgb_depth(frame_path=frame_path, output_dir=output_dir, camera=args.camera)


if __name__ == "__main__":
    main()
