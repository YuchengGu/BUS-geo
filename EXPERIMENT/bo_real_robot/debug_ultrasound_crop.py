from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_guided_collection_gui.surface_bayes import (  # noqa: E402
    crop_ultrasound_frame,
    parse_ultrasound_crop,
)


DEFAULT_SOURCE = Path("/home/ubuntu22/bc_data/gello/bo_0706_155812")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "EXPERIMENT" / "bo_real_robot" / "results" / "ultrasound_crop_debug"
DEFAULT_FRAME_INDEX = 0
DEFAULT_CROP = "99,769,542,1524"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and tune ultrasound crop used by online BO quality scoring."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="A directory containing recorded .pkl frames, or one .pkl frame.",
    )
    parser.add_argument("--frame-index", type=int, default=DEFAULT_FRAME_INDEX, help="Sorted frame index. Negative values count from end.")
    parser.add_argument(
        "--crop",
        default=DEFAULT_CROP,
        help="Crop as row0,row1,col0,col1; use 'none' for full frame.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save", action="store_true", help="Save png/pdf files instead of only showing the window.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a matplotlib window.")
    parser.add_argument("--list-only", action="store_true", help="Print recorded frames and exit.")
    parser.add_argument("--max-list", type=int, default=80)
    args = parser.parse_args()

    files = find_pkl_files(args.source)
    if args.list_only:
        print_frame_list(files, max_list=args.max_list)
        return

    frame_index = normalize_index(args.frame_index, len(files))
    frame_path = files[frame_index]
    sample = load_pkl(frame_path)
    image = ultrasound_image_from_sample(sample)
    crop = parse_ultrasound_crop(args.crop)
    cropped = crop_ultrasound_frame(image, crop)

    print(f"source: {frame_path}")
    print(f"raw shape: {tuple(image.shape)}")
    print(f"crop: {crop}")
    print(f"cropped shape: {tuple(cropped.shape)}")

    fig = make_side_by_side_figure(image, cropped, crop)
    fig.suptitle(
        f"{frame_path.name}\nraw={tuple(image.shape)}, crop={crop}, cropped={tuple(cropped.shape)}",
        fontsize=10,
    )
    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{frame_index:04d}_{frame_path.stem}"
        save_png_pdf(fig, args.output_dir / f"{stem}_side_by_side")
        print(f"saved under: {args.output_dir}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def find_pkl_files(source: Path) -> list[Path]:
    source = source.expanduser()
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(source)
    files = sorted(source.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {source}")
    return files


def normalize_index(index: int, count: int) -> int:
    if index < 0:
        index = count + index
    if not 0 <= index < count:
        raise IndexError(f"frame index {index} out of range for {count} files")
    return index


def load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict):
        raise TypeError(f"{path} did not contain a dict sample")
    return sample


def ultrasound_image_from_sample(sample: dict[str, Any]) -> np.ndarray:
    frame = sample.get("Ultrasound_gray")
    if frame is None:
        frame = sample.get("Ultrasound_rgb")
    if frame is None:
        raise KeyError("sample has neither Ultrasound_gray nor Ultrasound_rgb")
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def print_frame_list(files: list[Path], *, max_list: int) -> None:
    for i, path in enumerate(files[: max(0, int(max_list))]):
        meta_text = ""
        try:
            sample = load_pkl(path)
            meta = sample.get("meta", {})
            if isinstance(meta, dict):
                fields = []
                for key in ("bo_role", "bo_trial_index", "bo_phase", "Q", "F", "P_f", "P_tau"):
                    if key in meta:
                        fields.append(f"{key}={meta[key]}")
                meta_text = "  " + ", ".join(fields) if fields else ""
        except Exception as exc:
            meta_text = f"  unreadable_meta={type(exc).__name__}"
        print(f"{i:04d}  {path.name}{meta_text}")
    if len(files) > max_list:
        print(f"... {len(files) - max_list} more files")


def gray_for_plot(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


def save_full_with_crop(image: np.ndarray, crop: tuple[int, int, int, int] | None, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.imshow(gray_for_plot(image), cmap="gray", vmin=0, vmax=255)
    if crop is not None:
        row0, row1, col0, col1 = crop
        ax.add_patch(
            Rectangle(
                (col0, row0),
                col1 - col0,
                row1 - row0,
                fill=False,
                edgecolor="#00d084",
                linewidth=2.5,
            )
        )
    ax.set_axis_off()
    save_png_pdf(fig, output_stem)


def save_cropped(image: np.ndarray, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.imshow(gray_for_plot(image), cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()
    save_png_pdf(fig, output_stem)


def make_side_by_side_figure(
    raw: np.ndarray,
    cropped: np.ndarray,
    crop: tuple[int, int, int, int] | None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    axes[0].imshow(gray_for_plot(raw), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Full frame")
    if crop is not None:
        row0, row1, col0, col1 = crop
        axes[0].add_patch(
            Rectangle((col0, row0), col1 - col0, row1 - row0, fill=False, edgecolor="#00d084", linewidth=2.0)
        )
    axes[1].imshow(gray_for_plot(cropped), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Crop")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    return fig


def save_png_pdf(fig, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
