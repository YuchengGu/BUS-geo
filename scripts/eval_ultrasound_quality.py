#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visual_guided_collection_gui.ultrasound_quality import (
    QualityNormalization,
    UltrasoundQualityScorer,
)


def _read_mha_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required to read .mha files. Install it in this environment, "
            "or use --npy-frame for a saved single-frame numpy array."
        ) from exc

    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if volume.ndim == 2:
        if frame_index not in (0, -1):
            raise SystemExit(f"2D MHA only has frame 0, requested {frame_index}")
        return np.asarray(volume)
    if volume.ndim < 3:
        raise SystemExit(f"unsupported MHA array shape: {volume.shape}")
    if not 0 <= frame_index < volume.shape[0]:
        raise SystemExit(f"frame index {frame_index} is outside [0, {volume.shape[0] - 1}]")
    return np.asarray(volume[frame_index])


def _read_frame(args: argparse.Namespace) -> np.ndarray:
    if args.npy_frame is not None:
        return np.load(args.npy_frame)
    if args.mha is None:
        raise SystemExit("provide either --mha or --npy-frame")
    return _read_mha_frame(Path(args.mha).expanduser(), int(args.frame_index))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one ultrasound frame with D/E/C/S + TOPSIS Q.")
    parser.add_argument("--mha", type=str, default=None, help="Path to an MHA ultrasound volume.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index in the MHA volume.")
    parser.add_argument("--npy-frame", type=str, default=None, help="Optional .npy single frame for environments without SimpleITK.")
    parser.add_argument(
        "--max-size",
        type=int,
        default=160,
        help="Downsample the frame so the longest image side is at most this many pixels before scoring. Use 0 for full resolution.",
    )
    parser.add_argument(
        "--speckle-max-size",
        type=int,
        default=None,
        help="Optional separate max size for speckle index. Use 0 for full resolution. Defaults to --max-size.",
    )
    parser.add_argument(
        "--confidence-max-size",
        type=int,
        default=None,
        help="Optional separate max size for confidence response D. Use 0 for full resolution. Defaults to --max-size.",
    )
    parser.add_argument(
        "--confidence-method",
        choices=["fast", "random_walker"],
        default="fast",
        help="Use fast confidence approximation for online control, or random_walker for slower offline comparison.",
    )
    parser.add_argument("--d-min", type=float, default=0.0)
    parser.add_argument("--d-max", type=float, default=1.0)
    parser.add_argument("--e-min", type=float, default=0.0)
    parser.add_argument("--e-max", type=float, default=8.0)
    parser.add_argument("--c-min", type=float, default=0.0)
    parser.add_argument("--c-max", type=float, default=128.0)
    parser.add_argument("--c-target", type=float, default=64.0)
    parser.add_argument("--s-min", type=float, default=0.0)
    parser.add_argument("--s-max", type=float, default=2.0)
    args = parser.parse_args()

    frame = _read_frame(args)
    speckle_max_size = args.max_size if args.speckle_max_size is None else args.speckle_max_size
    confidence_max_size = args.max_size if args.confidence_max_size is None else args.confidence_max_size
    scorer = UltrasoundQualityScorer(
        normalization=QualityNormalization(
            d_min=args.d_min,
            d_max=args.d_max,
            e_min=args.e_min,
            e_max=args.e_max,
            c_min=args.c_min,
            c_max=args.c_max,
            c_target=args.c_target,
            s_min=args.s_min,
            s_max=args.s_max,
        ),
        max_size=None if args.max_size <= 0 else args.max_size,
        speckle_max_size=None if speckle_max_size <= 0 else speckle_max_size,
        confidence_max_size=None if confidence_max_size <= 0 else confidence_max_size,
        confidence_method=args.confidence_method,
    )
    score = scorer.score_frame(frame)

    payload = {
        "frame_index": int(args.frame_index),
        "features": {
            "D": score.features.D,
            "E": score.features.E,
            "C": score.features.C,
            "S": score.features.S,
        },
        "normalized": {
            "d": float(score.normalized[0]),
            "e": float(score.normalized[1]),
            "c": float(score.normalized[2]),
            "s": float(score.normalized[3]),
        },
        "weighted_vector": score.weighted_vector.tolist(),
        "Q": score.Q,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
