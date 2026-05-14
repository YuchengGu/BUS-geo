from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from breast_path_planning.geometry import normalize_rows

SCHEMA_VERSION = "planned_path_v1"


@dataclass
class PlannedPath:
    positions_base: np.ndarray
    normals_base: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    frame: str = "base"

    def __post_init__(self) -> None:
        self.positions_base = np.asarray(self.positions_base, dtype=float)
        self.normals_base = normalize_rows(np.asarray(self.normals_base, dtype=float))
        validate_planned_path(self)

    def __len__(self) -> int:
        return int(self.positions_base.shape[0])


def validate_planned_path(path: PlannedPath) -> None:
    if path.frame != "base":
        raise ValueError(f"planned path frame must be 'base', got {path.frame!r}")
    if path.positions_base.ndim != 2 or path.positions_base.shape[1] != 3:
        raise ValueError(f"positions_base must have shape (N, 3), got {path.positions_base.shape}")
    if path.normals_base.ndim != 2 or path.normals_base.shape[1] != 3:
        raise ValueError(f"normals_base must have shape (N, 3), got {path.normals_base.shape}")
    if path.positions_base.shape != path.normals_base.shape:
        raise ValueError(
            "positions_base and normals_base must have the same shape, "
            f"got {path.positions_base.shape} and {path.normals_base.shape}"
        )
    if path.positions_base.shape[0] == 0:
        raise ValueError("planned path must contain at least one point")
    if not np.isfinite(path.positions_base).all():
        raise ValueError("positions_base contains NaN or inf")
    if not np.isfinite(path.normals_base).all():
        raise ValueError("normals_base contains NaN or inf")
    norms = np.linalg.norm(path.normals_base, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("normals_base must be unit vectors")


def planned_path_to_dict(path: PlannedPath) -> dict[str, Any]:
    validate_planned_path(path)
    points = []
    for index, (position, normal) in enumerate(zip(path.positions_base, path.normals_base)):
        points.append(
            {
                "index": index,
                "position_base": [float(x) for x in position],
                "normal_base": [float(x) for x in normal],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "frame": path.frame,
        "points": points,
        "metadata": path.metadata,
    }


def planned_path_from_dict(data: dict[str, Any]) -> PlannedPath:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported planned path schema: {data.get('schema_version')!r}")
    if data.get("frame") != "base":
        raise ValueError(f"planned path frame must be 'base', got {data.get('frame')!r}")
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("planned path JSON must contain a non-empty points list")

    positions = []
    normals = []
    for expected_index, point in enumerate(points):
        if int(point.get("index", expected_index)) != expected_index:
            raise ValueError("planned path point indices must be contiguous from 0")
        positions.append(point["position_base"])
        normals.append(point["normal_base"])
    return PlannedPath(
        positions_base=np.asarray(positions, dtype=float),
        normals_base=np.asarray(normals, dtype=float),
        metadata=dict(data.get("metadata", {}) or {}),
        frame="base",
    )


def save_planned_path(path: PlannedPath, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(planned_path_to_dict(path), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_planned_path(input_path: str | Path) -> PlannedPath:
    with open(input_path, "r", encoding="utf-8") as f:
        return planned_path_from_dict(json.load(f))

