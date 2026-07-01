#!/usr/bin/env python3
from __future__ import annotations

import json
import glob
import select
import sys
import time
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_path_planning.geodesic_path import GeodesicPathParams, resample_path_with_surface_geodesics
from breast_path_planning.path_io import load_planned_path, save_planned_path
from breast_path_planning.pointcloud_from_d405 import PointCloud, load_point_cloud_ply
from breast_path_planning.surface_processing import estimate_normals


# 直接在这里改。目录里必须有 segmented_breast.ply 和 planned_path_before_geodesic.json。
# INPUT_DIRS 非空时按列表循环；否则按 INPUT_GLOB 自动找一批目录。
INPUT_DIRS: list[str] = []
INPUT_GLOB = "breast_path_planning/results/live_gui_0615_*"
INPUT_DIR: str | None = None
M = 50000000
INITIAL_TEMPERATURE = 1
COOLING_RATE = 0.995
PERTURBATION_RADIUS_M = 0.01
MAX_CANDIDATE_STEP_M = 0.0075
CORNER_PERTURBATION_SCALE = 0.1
MAX_ITERATIONS = 5000
ENERGY_RECORD_INTERVAL = 10
RANDOM_SEED = 0
NORMAL_K_NEIGHBORS = 20
SHOW_ENERGY_PLOT = False
SHOW_PATH_EVOLUTION_OPEN3D = True
EVOLUTION_SNAPSHOT_COUNT = 9
POISSON_DEPTH = 9
POISSON_DENSITY_QUANTILE = 0.03
SURFACE_MESH_COLOR = [0.7, 0.7, 0.55]
CONTINUOUS_PATH_TUBE_RADIUS_M = 0.0008
CONTINUOUS_PATH_SAMPLES_PER_SEGMENT = 9

# 视角选择：改这里即可。
# 可选: "iso", "front", "top", "side", "custom"
OPEN3D_VIEW_PRESET = "iso"
OPEN3D_VIEW_FRONT: list[float] | None = None
OPEN3D_VIEW_UP: list[float] | None = None
OPEN3D_VIEW_LOOKAT: list[float] | None = None
OPEN3D_VIEW_ZOOM = 0.75
OPEN3D_SAVE_VIEW_ON_ENTER = True
OPEN3D_CAPTURE_WIDTH = 2400
OPEN3D_CAPTURE_HEIGHT = 1800


def main() -> None:
    input_dirs = resolve_input_dirs(
        configured_dir=INPUT_DIR,
        configured_dirs=INPUT_DIRS,
        input_glob=INPUT_GLOB,
    )
    if not input_dirs:
        raise FileNotFoundError(
            "No valid input directories found. Set INPUT_DIRS, INPUT_DIR, or INPUT_GLOB at the top."
        )
    print(f"Found {len(input_dirs)} input directories.")
    for index, input_dir in enumerate(input_dirs, start=1):
        print(f"\n=== Geodesic demo {index}/{len(input_dirs)} ===")
        run_single_demo(input_dir)


def run_single_demo(input_dir: Path) -> None:
    cloud_path = input_dir / "segmented_breast.ply"
    path_path = input_dir / "planned_path_before_geodesic.json"
    if not cloud_path.exists() or not path_path.exists():
        raise FileNotFoundError(
            f"Expected segmented_breast.ply and planned_path_before_geodesic.json under {input_dir}"
        )

    output_dir = input_dir / "geodesic_single_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Loading point cloud and initial path ...")
    cloud = load_point_cloud_ply(cloud_path)
    initial_path = load_planned_path(path_path)
    print(f"Surface points: {len(cloud)}")
    print(f"Initial path points: {len(initial_path)}")

    print("Estimating surface normals with the same PCA routine used online ...")
    surface_normals = estimate_normals(cloud.points_base, k_neighbors=NORMAL_K_NEIGHBORS)

    params = GeodesicPathParams(
        max_iterations=MAX_ITERATIONS,
        fidelity_weight=M,
        initial_temperature=INITIAL_TEMPERATURE,
        cooling_rate=COOLING_RATE,
        perturbation_radius_m=PERTURBATION_RADIUS_M,
        max_candidate_step_m=MAX_CANDIDATE_STEP_M,
        corner_perturbation_scale=CORNER_PERTURBATION_SCALE,
        random_seed=RANDOM_SEED,
        energy_record_interval=ENERGY_RECORD_INTERVAL,
    )
    print("Parameters:")
    print(json.dumps(_params_dict(params), indent=2))
    print("Annealing progress:")

    records: list[dict[str, float | int]] = []
    snapshots: list[dict[str, object]] = []

    def on_progress(record: dict[str, float | int]) -> None:
        records.append(record)
        print(
            "  iter={iteration:5d}  T={temperature:.4g}  "
            "E={total:.6g}  Kg={curvature:.6g}  Fid={fidelity:.6g}  "
            "accepted={accepted_moves}".format(**record),
            flush=True,
        )

    def on_snapshot(record: dict[str, float | int], positions: np.ndarray) -> None:
        snapshots.append({"record": dict(record), "positions": positions.copy()})

    optimized = resample_path_with_surface_geodesics(
        initial_path,
        cloud.points_base,
        surface_normals_base=surface_normals,
        params=params,
        progress_callback=on_progress,
        path_snapshot_callback=on_snapshot,
    )

    save_planned_path(optimized, output_dir / "planned_path_geodesic.json")
    _write_json(output_dir / "energy_history.json", optimized.metadata["geodesic_energy_history"])
    _write_json(output_dir / "path_evolution_snapshots.json", _serializable_snapshots(snapshots))
    _write_json(output_dir / "params.json", _params_dict(params) | {"input_dir": str(input_dir)})
    print("Done.")
    print(f"Initial E: {optimized.metadata['geodesic_energy_initial']:.6g}")
    print(f"Final   E: {optimized.metadata['geodesic_energy_final']:.6g}")
    print(f"Initial curvature: {optimized.metadata['geodesic_curvature_initial']:.6g}")
    print(f"Final   curvature: {optimized.metadata['geodesic_curvature_final']:.6g}")
    print(f"Final fidelity: {optimized.metadata['geodesic_fidelity_final']:.6g}")
    print(f"Optimized path: {output_dir / 'planned_path_geodesic.json'}")

    if SHOW_ENERGY_PLOT:
        show_energy_plot(records, output_dir / "energy_curve.png")
    if SHOW_PATH_EVOLUTION_OPEN3D:
        show_open3d_path_evolution(cloud, surface_normals, snapshots, output_dir=output_dir)


def resolve_input_dir(configured_dir: str | None) -> Path:
    if not configured_dir or not configured_dir.strip():
        raise ValueError(
            "Set INPUT_DIR at the top of EXPERIMENT/demo_geodesic_annealing.py "
            "before running this demo."
        )
    return Path(configured_dir).expanduser()


def resolve_input_dirs(
    *,
    configured_dir: str | None,
    configured_dirs: list[str],
    input_glob: str,
) -> list[Path]:
    if configured_dirs:
        candidates = [Path(value).expanduser() for value in configured_dirs]
    elif configured_dir and configured_dir.strip():
        candidates = [resolve_input_dir(configured_dir)]
    else:
        candidates = [Path(value) for value in sorted(glob.glob(input_glob))]
    return [path for path in candidates if _is_valid_input_dir(path)]


def _is_valid_input_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "segmented_breast.ply").exists()
        and (path / "planned_path_before_geodesic.json").exists()
    )


def show_energy_plot(records: list[dict[str, float | int]], output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping energy plot.")
        return
    if not records:
        print("No energy records; skipping energy plot.")
        return
    iterations = np.asarray([record["iteration"] for record in records], dtype=float)
    total = np.asarray([record["total"] for record in records], dtype=float)
    curvature = np.asarray([record["curvature"] for record in records], dtype=float)
    fidelity = np.asarray([record["fidelity"] for record in records], dtype=float)
    temperature = np.asarray([record["temperature"] for record in records], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(iterations, total, label="total E", linewidth=2.0)
    axes[0].plot(iterations, curvature, label="curvature", linewidth=1.5)
    axes[0].plot(iterations, fidelity, label="fidelity", linewidth=1.5)
    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, temperature, color="tab:red", label="temperature")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Temperature")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    print(f"Energy plot saved: {output_png}")
    plt.show()


def show_open3d_path_evolution(
    cloud: PointCloud,
    surface_normals: np.ndarray,
    snapshots: list[dict[str, object]],
    *,
    output_dir: Path | None = None,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed; skipping path evolution rendering.")
        return
    selected = _select_even_snapshots(snapshots, EVOLUTION_SNAPSHOT_COUNT)
    if not selected:
        print("No path snapshots; skipping path evolution rendering.")
        return

    geometries = [_surface_render_geometry(o3d, cloud, surface_normals)]
    for index, snapshot in enumerate(selected):
        color = _evolution_color(index, len(selected))
        positions = np.asarray(snapshot["positions"], dtype=float)
        smooth = _catmull_rom_path(positions, samples_per_segment=CONTINUOUS_PATH_SAMPLES_PER_SEGMENT)
        geometries.extend(
            _path_tube_geometries(
                o3d,
                smooth,
                color=color,
                radius=CONTINUOUS_PATH_TUBE_RADIUS_M,
            )
        )
    print("Open3D path evolution:")
    print("  surface: very light reconstructed mesh")
    print("  path: smooth tubes, red/orange early -> green final")
    print(f"  shown snapshots: {len(selected)}")
    if OPEN3D_SAVE_VIEW_ON_ENTER and output_dir is not None:
        _run_open3d_interactive_capture(o3d, geometries, cloud, output_dir)
    else:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Continuous geodesic evolution: red to green",
            **_open3d_view_kwargs(cloud),
        )


def _cloud_geometry(o3d, cloud: PointCloud):
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(cloud.points_base)
    if cloud.colors_rgb is not None:
        geom.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors_rgb, dtype=float) / 255.0)
    else:
        geom.colors = o3d.utility.Vector3dVector(np.tile([[0.65, 0.65, 0.65]], (len(cloud), 1)))
    return geom


def _surface_render_geometry(o3d, cloud: PointCloud, surface_normals: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points_base)
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(surface_normals, dtype=float))
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=int(POISSON_DEPTH),
        )
        density_values = np.asarray(densities)
        if density_values.size > 0:
            threshold = float(np.quantile(density_values, POISSON_DENSITY_QUANTILE))
            mesh.remove_vertices_by_mask(density_values < threshold)
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(SURFACE_MESH_COLOR)
        return mesh
    except Exception as exc:
        print(f"Surface mesh reconstruction failed; falling back to point cloud: {exc}")
        return _cloud_geometry(o3d, cloud)


def _open3d_view_kwargs(cloud: PointCloud, *, preset: str | None = None) -> dict[str, object]:
    selected = OPEN3D_VIEW_PRESET if preset is None else preset
    lookat = _open3d_lookat(cloud)
    zoom = float(OPEN3D_VIEW_ZOOM)
    presets = {
        "iso": {
            "front": [-0.45, -0.45, -0.77],
            "up": [-0.35, -0.35, 0.87],
        },
        "front": {
            "front": [0.0, -1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
        "top": {
            "front": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
        },
        "side": {
            "front": [-1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
    }
    if selected == "custom":
        if OPEN3D_VIEW_FRONT is None or OPEN3D_VIEW_UP is None:
            raise ValueError("custom view requires OPEN3D_VIEW_FRONT and OPEN3D_VIEW_UP")
        front = OPEN3D_VIEW_FRONT
        up = OPEN3D_VIEW_UP
        if OPEN3D_VIEW_LOOKAT is not None:
            lookat = [float(value) for value in OPEN3D_VIEW_LOOKAT]
    elif selected in presets:
        front = presets[selected]["front"]
        up = presets[selected]["up"]
    else:
        raise ValueError(f"Unsupported OPEN3D_VIEW_PRESET: {selected!r}")
    return {
        "front": [float(value) for value in front],
        "lookat": lookat,
        "up": [float(value) for value in up],
        "zoom": zoom,
    }


def _open3d_lookat(cloud: PointCloud) -> list[float]:
    points = np.asarray(cloud.points_base, dtype=float)
    if points.shape[0] == 0:
        return [0.0, 0.0, 0.0]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return ((lower + upper) * 0.5).astype(float).tolist()


def _run_open3d_interactive_capture(o3d, geometries: list[object], cloud: PointCloud, output_dir: Path) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Drag view, then press Enter in terminal to save",
        width=int(OPEN3D_CAPTURE_WIDTH),
        height=int(OPEN3D_CAPTURE_HEIGHT),
        visible=True,
    )
    for geometry in geometries:
        vis.add_geometry(geometry)
    view = _open3d_view_kwargs(cloud)
    control = vis.get_view_control()
    control.set_front(view["front"])
    control.set_lookat(view["lookat"])
    control.set_up(view["up"])
    control.set_zoom(float(view["zoom"]))
    print("Open3D window is interactive.")
    print("  1. Drag/zoom/rotate in the Open3D window until the view is right.")
    print("  2. Return to this terminal and press Enter to save PNG/PDF.")
    print("  3. Close the Open3D window to skip saving.")
    saved = False
    try:
        while True:
            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
            if _terminal_enter_pressed():
                png_path, pdf_path = _open3d_capture_paths(output_dir)
                vis.capture_screen_image(str(png_path), do_render=True)
                _png_to_pdf(png_path, pdf_path)
                print(f"Saved current Open3D view: {png_path}")
                print(f"Saved current Open3D view: {pdf_path}")
                saved = True
                break
            time.sleep(0.03)
    finally:
        vis.destroy_window()
    if not saved:
        print("Open3D window closed without saving.")


def _terminal_enter_pressed() -> bool:
    if not sys.stdin.isatty():
        return False
    readable, _writable, _errors = select.select([sys.stdin], [], [], 0.03)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def _open3d_capture_paths(output_dir: Path) -> tuple[Path, Path]:
    base = Path(output_dir)
    return base / "path_evolution_view.png", base / "path_evolution_view.pdf"


def _png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    image = plt.imread(png_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 300.0, height / 300.0))
    axis = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(image)
    axis.axis("off")
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)


def _path_tube_geometries(
    o3d,
    points: np.ndarray,
    *,
    color: list[float],
    radius: float | None = None,
):
    positions = np.asarray(points, dtype=float)
    tube_radius = CONTINUOUS_PATH_TUBE_RADIUS_M if radius is None else float(radius)
    geometries = []
    for start, end in zip(positions[:-1], positions[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=tube_radius,
            height=length,
            resolution=16,
        )
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)
        cylinder.rotate(_rotation_from_z_to_vector(o3d, delta / length), center=(0.0, 0.0, 0.0))
        cylinder.translate((start + end) * 0.5)
        geometries.append(cylinder)
    return geometries


def _rotation_from_z_to_vector(o3d, direction: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    target = np.asarray(direction, dtype=float)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    axis = np.cross(z_axis, target)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if axis_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0]))
    angle = float(np.arctan2(axis_norm, dot))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_norm * angle)


def _catmull_rom_path(points: np.ndarray, *, samples_per_segment: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    if len(values) < 3 or samples_per_segment <= 1:
        return values.copy()

    out = []
    for i in range(len(values) - 1):
        p0 = values[max(i - 1, 0)]
        p1 = values[i]
        p2 = values[i + 1]
        p3 = values[min(i + 2, len(values) - 1)]
        for sample in range(samples_per_segment):
            t = float(sample) / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(point)
    out.append(values[-1])
    return np.asarray(out, dtype=float)


def _select_even_snapshots(snapshots: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0 or not snapshots:
        return []
    if len(snapshots) <= count:
        return list(snapshots)
    indices = np.linspace(0, len(snapshots) - 1, num=count)
    return [snapshots[int(round(index))] for index in indices]


def _evolution_color(index: int, total: int) -> list[float]:
    if total <= 1:
        t = 1.0
    else:
        t = float(index) / float(total - 1)
    return _evolution_color_from_t(t)


def _evolution_color_from_t(t: float) -> list[float]:
    value = float(np.clip(t, 0.0, 1.0))
    red = np.array([0.95, 0.10, 0.05], dtype=float)
    yellow = np.array([0.95, 0.72, 0.05], dtype=float)
    green = np.array([0.00, 0.75, 0.20], dtype=float)
    if value < 0.5:
        local = value / 0.5
        color = red * (1.0 - local) + yellow * local
    else:
        local = (value - 0.5) / 0.5
        color = yellow * (1.0 - local) + green * local
    return color.tolist()


def _serializable_snapshots(snapshots: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for snapshot in snapshots:
        out.append(
            {
                "record": snapshot["record"],
                "positions_base": np.asarray(snapshot["positions"], dtype=float).tolist(),
            }
        )
    return out


def _params_dict(params: GeodesicPathParams) -> dict[str, float | int | None]:
    return {
        "m": float(params.fidelity_weight),
        "initial_temperature": float(params.initial_temperature),
        "cooling_rate": float(params.cooling_rate),
        "perturbation_radius_m": float(params.perturbation_radius_m),
        "max_candidate_step_m": (
            None if params.max_candidate_step_m is None else float(params.max_candidate_step_m)
        ),
        "corner_perturbation_scale": float(params.corner_perturbation_scale),
        "max_iterations": int(params.max_iterations),
        "energy_record_interval": int(params.energy_record_interval),
        "random_seed": params.random_seed,
    }


def _write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
