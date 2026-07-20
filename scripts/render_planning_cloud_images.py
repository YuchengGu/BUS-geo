#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.pointcloud_from_d405 import PointCloud, load_point_cloud_ply


def default_case_dir() -> Path:
    root = Path("breast_path_planning/results")
    candidates = sorted(root.glob("live_gui_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "raw_cloud_base.ply").exists() and (candidate / "segmented_breast.ply").exists():
            return candidate
    raise FileNotFoundError("No live_gui_* directory with raw_cloud_base.ply and segmented_breast.ply was found.")


def pca_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:3]
    projected = centered @ axes.T
    return projected, center, axes


def set_equal_limits(ax, xy: np.ndarray, margin: float = 0.04) -> None:
    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    if span <= 0:
        span = 1.0
    pad = span * float(margin)
    center = (mins + maxs) * 0.5
    ax.set_xlim(center[0] - span * 0.5 - pad, center[0] + span * 0.5 + pad)
    ax.set_ylim(center[1] - span * 0.5 - pad, center[1] + span * 0.5 + pad)


def render_cloud_pair(
    *,
    raw_cloud: PointCloud,
    segmented_cloud: PointCloud,
    output_dir: Path,
    point_size: float,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_points = np.asarray(raw_cloud.points_base, dtype=float)
    seg_points = np.asarray(segmented_cloud.points_base, dtype=float)
    raw_proj, center, axes = pca_projection(raw_points)
    seg_proj = (seg_points - center) @ axes.T

    raw_colors = (
        np.asarray(raw_cloud.colors_rgb, dtype=float) / 255.0
        if raw_cloud.colors_rgb is not None
        else np.tile(np.array([[0.55, 0.55, 0.55]]), (len(raw_cloud), 1))
    )
    seg_colors = (
        np.asarray(segmented_cloud.colors_rgb, dtype=float) / 255.0
        if segmented_cloud.colors_rgb is not None
        else np.tile(np.array([[0.0, 0.8, 0.1]]), (len(segmented_cloud), 1))
    )

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=dpi)
    ax.scatter(raw_proj[:, 0], raw_proj[:, 1], s=point_size, c=raw_colors, linewidths=0, alpha=0.18)
    ax.scatter(seg_proj[:, 0], seg_proj[:, 1], s=point_size * 1.8, c=seg_colors, linewidths=0, alpha=0.95)
    set_equal_limits(ax, raw_proj[:, :2])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output_dir / "segmentation_rgb_cloud_render.png", transparent=False)
    fig.savefig(output_dir / "segmentation_rgb_cloud_render.pdf", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=dpi)
    ax.scatter(raw_proj[:, 0], raw_proj[:, 1], s=point_size, c="#B8B8B8", linewidths=0, alpha=0.10)
    values = seg_proj[:, 2]
    sc = ax.scatter(
        seg_proj[:, 0],
        seg_proj[:, 1],
        s=point_size * 1.8,
        c=values,
        cmap="turbo",
        linewidths=0,
        alpha=0.98,
    )
    set_equal_limits(ax, raw_proj[:, :2])
    ax.set_axis_off()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("relative depth / surface height (m)")
    fig.tight_layout(pad=0)
    fig.savefig(output_dir / "segmentation_depth_cloud_render.png", transparent=False)
    fig.savefig(output_dir / "segmentation_depth_cloud_render.pdf", transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render RGB-colored and depth-colored images from the point cloud used for surface segmentation."
    )
    parser.add_argument("--case-dir", type=Path, default=None, help="live_gui_* planning directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    case_dir = args.case_dir or default_case_dir()
    output_dir = args.output_dir or (case_dir / "segmentation_cloud_renders")
    raw_cloud_path = case_dir / "raw_cloud_base.ply"
    segmented_cloud_path = case_dir / "segmented_breast.ply"
    if not raw_cloud_path.exists() or not segmented_cloud_path.exists():
        raise FileNotFoundError(f"Expected raw_cloud_base.ply and segmented_breast.ply under {case_dir}")

    render_cloud_pair(
        raw_cloud=load_point_cloud_ply(raw_cloud_path),
        segmented_cloud=load_point_cloud_ply(segmented_cloud_path),
        output_dir=output_dir,
        point_size=float(args.point_size),
        dpi=int(args.dpi),
    )
    print(f"case_dir: {case_dir}")
    print(f"saved under: {output_dir}")


if __name__ == "__main__":
    main()
