#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_geodesic_coverage"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from breast_path_planning.path_io import load_planned_path
from breast_path_planning.path_smoothing import (
    b_spline_smooth_path,
    moving_average_smooth_path,
)
from breast_path_planning.pointcloud_from_d405 import load_point_cloud_ply
from breast_path_planning.surface_processing import estimate_normals
from EXPERIMENT.geodesic_real_robot.force_analysis import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    GROUPS,
    METHOD_ORDER,
)


PROBE_WIDTH_M = 0.04
PATH_SAMPLE_SPACING_M = 0.0008
POISSON_DEPTH = 10
POISSON_DENSITY_QUANTILE = 0.03
NORMAL_K_NEIGHBORS = 20
MAX_MESH_POINT_DISTANCE_M = 0.003
MOVING_AVERAGE_WINDOW = 5
MOVING_AVERAGE_PASSES = 2
B_SPLINE_SMOOTHING_FACTOR = 0.0007830520230250682

BOX_FIGSIZE = (3.25, 2.55)
PLOT_METHOD_ORDER = ("moving_average", "b_spline", "geodesic")
METHOD_LABELS = {
    "original": "Original\n(Clinical)",
    "moving_average": "Moving\naverage",
    "b_spline": "B-spline",
    "geodesic": "Geodesic\n(Ours)",
}
METHOD_COLORS = {
    "original": "#D95F5F",
    "moving_average": "#8C8C8C",
    "b_spline": "#4C78A8",
    "geodesic": "#4DAF7C",
}
AXIS_COLOR = "#222222"

PLANNING_DIRS = {
    2: "breast_path_planning/results/live_gui_0630_180109",
    3: "breast_path_planning/results/live_gui_0630_182431",
    4: "breast_path_planning/results/live_gui_0701_164308",
    5: "breast_path_planning/results/live_gui_0701_170313",
    6: "breast_path_planning/results/live_gui_0701_171518",
    7: "breast_path_planning/results/live_gui_0701_172804",
    8: "breast_path_planning/results/live_gui_0701_174815",
}


def densify_polyline(points: np.ndarray, *, spacing_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    spacing = float(spacing_m)
    if spacing <= 0.0:
        raise ValueError("spacing_m must be positive")
    dense = [values[0]]
    for start, end in zip(values[:-1], values[1:]):
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(length / spacing)))
        for step in range(1, steps + 1):
            dense.append(start + (end - start) * (step / steps))
    return np.asarray(dense, dtype=float)


def build_mesh_adjacency(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=int)
    neighbors: list[dict[int, float]] = [dict() for _ in range(len(points))]
    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            weight = float(np.linalg.norm(points[first] - points[second]))
            previous = neighbors[first].get(second)
            if previous is None or weight < previous:
                neighbors[first][second] = weight
                neighbors[second][first] = weight
    return tuple(tuple(values.items()) for values in neighbors)


def multi_source_mesh_distances(
    adjacency: tuple[tuple[tuple[int, float], ...], ...],
    *,
    source_indices: np.ndarray,
) -> np.ndarray:
    distances = np.full(len(adjacency), np.inf, dtype=float)
    queue: list[tuple[float, int]] = []
    for source in np.unique(np.asarray(source_indices, dtype=int)):
        if source < 0 or source >= len(adjacency):
            raise IndexError(f"source index {source} is outside the mesh")
        distances[source] = 0.0
        heapq.heappush(queue, (0.0, int(source)))
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance > distances[vertex]:
            continue
        for neighbor, weight in adjacency[vertex]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def covered_triangle_area(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_distances: np.ndarray,
    *,
    radius_m: float,
) -> tuple[float, int]:
    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=int)
    distances = np.asarray(vertex_distances, dtype=float)
    triangle_distances = np.mean(distances[faces], axis=1)
    covered_mask = np.isfinite(triangle_distances) & (
        triangle_distances <= float(radius_m)
    )
    vectors_a = points[faces[:, 1]] - points[faces[:, 0]]
    vectors_b = points[faces[:, 2]] - points[faces[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1)
    return float(np.sum(areas[covered_mask])), int(np.count_nonzero(covered_mask))


def path_coverage_area(
    path_positions: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    adjacency: tuple[tuple[tuple[int, float], ...], ...],
    *,
    probe_width_m: float,
    sample_spacing_m: float,
) -> tuple[float, int, int, float]:
    dense = densify_polyline(path_positions, spacing_m=sample_spacing_m)
    nearest_distances, nearest_indices = cKDTree(vertices).query(dense, k=1)
    sources = np.unique(np.asarray(nearest_indices, dtype=int))
    surface_distances = multi_source_mesh_distances(
        adjacency,
        source_indices=sources,
    )
    area, covered = covered_triangle_area(
        vertices,
        triangles,
        surface_distances,
        radius_m=0.5 * float(probe_width_m),
    )
    return area, covered, int(len(sources)), float(np.max(nearest_distances))


def analyze_groups(
    groups: Iterable[int],
    *,
    repo_root: Path,
    data_root: Path,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for group in groups:
        planning_dir = repo_root / PLANNING_DIRS[int(group)]
        print(f"\nGroup {group}: {planning_dir}")
        vertices, triangles = reconstruct_clean_mesh(
            planning_dir / "segmented_breast.ply"
        )
        print(f"  mesh vertices={len(vertices)}, triangles={len(triangles)}")
        adjacency = build_mesh_adjacency(vertices, triangles)
        paths = load_group_paths(
            int(group),
            planning_dir=planning_dir,
            data_root=data_root,
        )
        group_rows = []
        for method in METHOD_ORDER:
            area_m2, covered, source_count, snap_max = path_coverage_area(
                paths[method],
                vertices,
                triangles,
                adjacency,
                probe_width_m=PROBE_WIDTH_M,
                sample_spacing_m=PATH_SAMPLE_SPACING_M,
            )
            row: dict[str, float | int | str] = {
                "group": int(group),
                "method": method,
                "coverage_area_m2": area_m2,
                "coverage_area_cm2": area_m2 * 10000.0,
                "covered_triangle_count": covered,
                "mesh_vertex_count": int(len(vertices)),
                "mesh_triangle_count": int(len(triangles)),
                "path_source_vertex_count": source_count,
                "path_snap_max_mm": snap_max * 1000.0,
                "probe_width_mm": PROBE_WIDTH_M * 1000.0,
            }
            group_rows.append(row)
            print(
                f"  {method:15s} area={area_m2 * 10000.0:.3f} cm^2, "
                f"sources={source_count}, max_snap={snap_max * 1000.0:.3f} mm"
            )
        original_area = float(group_rows[0]["coverage_area_m2"])
        for row in group_rows:
            row["coverage_retention_percent"] = (
                100.0 * float(row["coverage_area_m2"]) / original_area
            )
        rows.extend(group_rows)
    return rows


def reconstruct_clean_mesh(
    cloud_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    cloud = load_point_cloud_ply(cloud_path)
    normals = estimate_normals(
        cloud.points_base,
        k_neighbors=NORMAL_K_NEIGHBORS,
    )
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(cloud.points_base)
    source.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        source,
        depth=int(POISSON_DEPTH),
    )
    density_values = np.asarray(densities)
    if density_values.size:
        threshold = float(np.quantile(density_values, POISSON_DENSITY_QUANTILE))
        mesh.remove_vertices_by_mask(density_values < threshold)
    mesh = mesh.crop(source.get_axis_aligned_bounding_box())

    mesh_vertices = o3d.geometry.PointCloud()
    mesh_vertices.points = mesh.vertices
    source_distances = np.asarray(
        mesh_vertices.compute_point_cloud_distance(source),
        dtype=float,
    )
    mesh.remove_vertices_by_mask(
        source_distances > float(MAX_MESH_POINT_DISTANCE_M)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    triangles = np.asarray(mesh.triangles, dtype=int)
    if len(triangles) == 0:
        raise RuntimeError(f"Poisson reconstruction produced no triangles: {cloud_path}")
    clusters, counts, _areas = mesh.cluster_connected_triangles()
    cluster_values = np.asarray(clusters, dtype=int)
    count_values = np.asarray(counts, dtype=int)
    keep_cluster = int(np.argmax(count_values))
    mesh.remove_triangles_by_mask(cluster_values != keep_cluster)
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError(f"Cleaned mesh is empty: {cloud_path}")
    return vertices, triangles


def load_group_paths(
    group: int,
    *,
    planning_dir: Path,
    data_root: Path,
) -> dict[str, np.ndarray]:
    original = load_planned_path(
        planning_dir / "planned_path_before_geodesic.json"
    ).positions_base
    return {
        "original": np.asarray(original, dtype=float),
        "moving_average": moving_average_smooth_path(
            original,
            window=MOVING_AVERAGE_WINDOW,
            passes=MOVING_AVERAGE_PASSES,
        ),
        "b_spline": b_spline_smooth_path(
            original,
            smoothing_factor=B_SPLINE_SMOOTHING_FACTOR,
        ),
        "geodesic": load_recorded_geodesic_path(
            group,
            data_root=data_root,
            endpoint_fallback=np.asarray(original, dtype=float),
        ),
    }


def load_recorded_geodesic_path(
    group: int,
    *,
    data_root: Path,
    endpoint_fallback: np.ndarray,
) -> np.ndarray:
    episode = next(
        episode
        for episode in GROUPS[group]["episodes"]
        if _episode_method(data_root / episode) == "geodesic"
    )
    frame_paths = sorted((data_root / episode).glob("*.pkl"))
    point_count = len(endpoint_fallback)
    points: dict[int, np.ndarray] = {}
    for frame_path in frame_paths[::25] + frame_paths[-1:]:
        _collect_path_targets(frame_path, points)
        if len(points) >= point_count:
            break
    points.setdefault(0, endpoint_fallback[0])
    points.setdefault(point_count - 1, endpoint_fallback[-1])
    if sorted(points) != list(range(point_count)):
        for frame_path in frame_paths:
            _collect_path_targets(frame_path, points)
            if len(points) >= point_count:
                break
    if sorted(points) != list(range(point_count)):
        raise RuntimeError(
            f"Could not reconstruct all {point_count} geodesic points from {episode}"
        )
    return np.stack([points[index] for index in range(point_count)])


def _episode_method(episode_dir: Path) -> str:
    first = next(iter(sorted(episode_dir.glob("*.pkl"))), None)
    if first is None:
        raise FileNotFoundError(f"No PKL frames in {episode_dir}")
    with first.open("rb") as handle:
        frame = pickle.load(handle)
    return str(frame.get("meta", {}).get("path_variant_method", ""))


def _collect_path_targets(
    frame_path: Path,
    points: dict[int, np.ndarray],
) -> None:
    with frame_path.open("rb") as handle:
        frame = pickle.load(handle)
    indices = np.asarray(frame.get("path_indices", []), dtype=int)
    positions = np.asarray(frame.get("path_target_positions_base", []), dtype=float)
    for index, position in zip(indices, positions):
        points[int(index)] = position.copy()


def plot_coverage_boxplot(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    _configure_matplotlib()
    values = _coverage_values(rows)
    fig, axis = plt.subplots(figsize=BOX_FIGSIZE)
    _draw_method_boxplot(
        axis,
        values,
        ylabel="Coverage\nretention\n$(\\%)$",
        ylabel_x=-0.21,
    )
    axis.axhline(100.0, color="#777777", linestyle="--", linewidth=0.7)
    fig.tight_layout(pad=0.9)
    return _save_figure(fig, output_dir / "coverage_retention_boxplot")


def _coverage_values(
    rows: list[dict[str, float | int | str]],
) -> list[list[float]]:
    return [
        [
            float(row["coverage_retention_percent"])
            for row in rows
            if row["method"] == method
        ]
        for method in PLOT_METHOD_ORDER
    ]


def _draw_method_boxplot(
    axis: plt.Axes,
    values: list[list[float]],
    *,
    ylabel: str,
    ylabel_x: float,
) -> None:
    positions = np.arange(1, len(PLOT_METHOD_ORDER) + 1)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": AXIS_COLOR, "linewidth": 1.0},
        whiskerprops={"color": AXIS_COLOR, "linewidth": 0.8},
        capprops={"color": AXIS_COLOR, "linewidth": 0.8},
    )
    for patch, method in zip(
        boxes["boxes"],
        PLOT_METHOD_ORDER,
        strict=True,
    ):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.45)
        patch.set_edgecolor(AXIS_COLOR)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for index, (method, series) in enumerate(
        zip(PLOT_METHOD_ORDER, values, strict=True),
        start=1,
    ):
        jitter = rng.normal(0.0, 0.03, size=len(series))
        axis.scatter(
            np.full(len(series), index) + jitter,
            series,
            s=8,
            color=METHOD_COLORS[method],
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [METHOD_LABELS[method] for method in PLOT_METHOD_ORDER],
        rotation=25,
        ha="center",
    )
    axis.set_ylabel(
        ylabel,
        rotation=0,
        labelpad=22,
        fontsize=7,
    )
    axis.yaxis.set_label_coords(ylabel_x, 0.5)
    _style_axis(axis)
    _add_geodesic_coverage_gain(axis, values)
    for tick in axis.get_xticklabels():
        if "(Ours)" in tick.get_text():
            tick.set_fontweight("bold")


def _add_geodesic_coverage_gain(
    axis: plt.Axes,
    values: list[list[float]],
) -> None:
    if "geodesic" not in PLOT_METHOD_ORDER:
        return
    geodesic_index = PLOT_METHOD_ORDER.index("geodesic")
    geodesic_median = _finite_median(values[geodesic_index])
    if not np.isfinite(geodesic_median):
        return
    lines: list[str] = []
    short_labels = {
        "moving_average": "MA",
        "b_spline": "B-spline",
    }
    for method, series in zip(PLOT_METHOD_ORDER, values, strict=True):
        if method == "geodesic" or method not in short_labels:
            continue
        baseline_median = _finite_median(series)
        if not np.isfinite(baseline_median):
            continue
        gain_pp = geodesic_median - baseline_median
        sign = "+" if gain_pp >= 0 else "-"
        lines.append(f"vs {short_labels[method]} {sign}{abs(gain_pp):.1f} pp")
    if not lines:
        return
    axis.text(
        0.98,
        0.40,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=METHOD_COLORS["geodesic"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )


def _finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def write_rows(
    rows: list[dict[str, float | int | str]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(csv_path: Path) -> list[dict[str, float | int | str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def analysis_config() -> dict[str, float | int]:
    return {
        "probe_width_m": PROBE_WIDTH_M,
        "path_sample_spacing_m": PATH_SAMPLE_SPACING_M,
        "poisson_depth": POISSON_DEPTH,
        "poisson_density_quantile": POISSON_DENSITY_QUANTILE,
        "normal_k_neighbors": NORMAL_K_NEIGHBORS,
        "max_mesh_point_distance_m": MAX_MESH_POINT_DISTANCE_M,
    }


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            "font.size": 8,
            "axes.linewidth": 0.8,
        }
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(
        axis="both",
        labelsize=7,
        width=0.8,
        length=3.0,
        colors=AXIS_COLOR,
    )
    axis.spines["left"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_color(AXIS_COLOR)


def _save_figure(
    fig: plt.Figure,
    output_stem: Path,
) -> tuple[Path, Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = (
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".svg"),
        output_stem.with_suffix(".png"),
    )
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = _parser().parse_args()
    output_dir = Path(args.output_dir) / "coverage_area"
    csv_path = output_dir / "coverage_area_metrics.csv"
    config_path = output_dir / "coverage_area_config.json"
    config = analysis_config()
    cached_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else None
    )
    if (
        csv_path.exists()
        and not args.recompute
        and cached_config == config
    ):
        print(f"Loading existing metrics: {csv_path}")
        rows = read_rows(csv_path)
    else:
        rows = analyze_groups(
            args.groups,
            repo_root=REPO_ROOT,
            data_root=Path(args.data_root).expanduser(),
        )
        write_rows(rows, csv_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        print(f"Saved metrics: {csv_path}")
    for path in plot_coverage_boxplot(rows, output_dir):
        print(f"Saved: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute mesh-union probe coverage and plot coverage retention."
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=list(range(2, 9)),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recompute", action="store_true")
    return parser


if __name__ == "__main__":
    main()
