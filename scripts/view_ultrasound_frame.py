#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from visual_guided_collection_gui.ultrasound_quality import UltrasoundQualityScorer


def read_mha_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required to read .mha files.") from exc

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


def read_frame(args: argparse.Namespace) -> np.ndarray:
    if args.npy_frame is not None:
        return np.load(args.npy_frame)
    if args.mha is None:
        raise SystemExit("provide either --mha or --npy-frame")
    return read_mha_frame(Path(args.mha).expanduser(), int(args.frame_index))


def render_frame(
    frame: np.ndarray,
    *,
    title: str,
    save_path: Path | None,
    show: bool,
    cmap: str = "gray",
) -> None:
    if save_path is not None and "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    if save_path is not None and not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.imshow(np.asarray(frame), cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one ultrasound frame and optionally show its quality score.")
    parser.add_argument("--mha", type=str, default=None, help="Path to an MHA ultrasound volume.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index in the MHA volume.")
    parser.add_argument("--npy-frame", type=str, default=None, help="Optional .npy single ultrasound frame.")
    parser.add_argument("--save", type=str, default=None, help="Save visualization PNG to this path.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    parser.add_argument("--no-score", action="store_true", help="Only visualize the image, without D/E/C/S/Q.")
    parser.add_argument("--max-size", type=int, default=160)
    parser.add_argument("--speckle-max-size", type=int, default=None)
    parser.add_argument("--confidence-max-size", type=int, default=None)
    parser.add_argument("--confidence-method", choices=["fast", "random_walker"], default="fast")
    args = parser.parse_args()

    frame = read_frame(args)
    title = f"frame {args.frame_index} | shape={tuple(frame.shape)}"
    if not args.no_score:
        speckle_max_size = args.max_size if args.speckle_max_size is None else args.speckle_max_size
        confidence_max_size = args.max_size if args.confidence_max_size is None else args.confidence_max_size
        scorer = UltrasoundQualityScorer(
            max_size=None if args.max_size <= 0 else args.max_size,
            speckle_max_size=None if speckle_max_size <= 0 else speckle_max_size,
            confidence_max_size=None if confidence_max_size <= 0 else confidence_max_size,
            confidence_method=args.confidence_method,
        )
        score = scorer.score_frame(frame)
        title = (
            f"{title}\n"
            f"Q={score.Q:.3f}  D={score.features.D:.3f}  E={score.features.E:.3f}  "
            f"C={score.features.C:.3f}  S={score.features.S:.3f}"
        )
        print(
            f"Q={score.Q:.6f} "
            f"D={score.features.D:.6f} E={score.features.E:.6f} "
            f"C={score.features.C:.6f} S={score.features.S:.6f}"
        )
    save_path = None if args.save is None else Path(args.save).expanduser()
    render_frame(frame, title=title, save_path=save_path, show=bool(args.show), cmap="gray")


if __name__ == "__main__":
    main()
