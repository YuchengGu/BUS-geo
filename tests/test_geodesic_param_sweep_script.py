import csv
import importlib.util
from pathlib import Path

import numpy as np

from breast_path_planning.path_io import PlannedPath, save_planned_path
from breast_path_planning.pointcloud_from_d405 import PointCloud, save_point_cloud_ply


def _load_sweep_module():
    script = Path(__file__).resolve().parents[1] / "experiment" / "run_geodesic_param_sweep.py"
    spec = importlib.util.spec_from_file_location("run_geodesic_param_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_geodesic_param_sweep_writes_summary_paths_and_energy_history(tmp_path):
    module = _load_sweep_module()
    input_dir = tmp_path / "live_gui_test"
    input_dir.mkdir()
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    colors = np.tile(np.array([[120, 130, 140]], dtype=np.uint8), (len(surface_points), 1))
    save_point_cloud_ply(PointCloud(surface_points, colors), input_dir / "segmented_breast.ply")
    save_planned_path(
        PlannedPath(
            positions_base=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            normals_base=np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1)),
        ),
        input_dir / "planned_path.json",
    )

    output_dir = module.run_sweep(
        input_dir=input_dir,
        output_dir=tmp_path / "sweep",
        m_values=[0.01],
        temperature_values=[1.0],
        alpha_values=[0.995],
        radius_values=[1.5],
        max_iterations=200,
        energy_record_interval=20,
        seed_base=10,
    )

    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["m"] == "0.01"
    assert float(rows[0]["E_final"]) < float(rows[0]["E_initial"])
    run_dir = Path(rows[0]["run_dir"])
    assert (run_dir / "planned_path_geodesic.json").exists()
    assert (run_dir / "energy_history.json").exists()
    assert (run_dir / "params.json").exists()
