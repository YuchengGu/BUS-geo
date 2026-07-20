from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breast_path_planning.geometry import normalize_vector
from breast_path_planning.path_io import PlannedPath
from breast_path_planning.pointcloud_from_d405 import PointCloud
from visual_guided_collection_gui.surface_teleop import path_tangents


@dataclass(frozen=True)
class RandomSurfaceReference:
    cloud_index: int
    position_base: np.ndarray
    normal_base: np.ndarray
    reference_path: PlannedPath


@dataclass(frozen=True)
class PathBOReference:
    path_index: int
    position_base: np.ndarray
    normal_base: np.ndarray
    tangent_base: np.ndarray


def select_random_surface_reference(
    segmented_cloud: PointCloud,
    normals_base: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> RandomSurfaceReference:
    if len(segmented_cloud) == 0:
        raise ValueError("segmented surface is empty")
    normals = np.asarray(normals_base, dtype=float)
    if normals.shape != segmented_cloud.points_base.shape:
        raise ValueError("normals_base must match segmented surface points")
    generator = rng if rng is not None else np.random.default_rng()
    index = int(generator.integers(0, len(segmented_cloud)))
    position = segmented_cloud.points_base[index].copy()
    normal = normals[index].copy()
    path = PlannedPath(
        positions_base=position.reshape(1, 3),
        normals_base=normal.reshape(1, 3),
        metadata={
            "source": "segmented_surface",
            "path_variant_method": "bo_random_surface_point",
            "surface_cloud_index": index,
        },
    )
    return RandomSurfaceReference(
        cloud_index=index,
        position_base=position,
        normal_base=path.normals_base[0].copy(),
        reference_path=path,
    )


def select_random_path_bo_reference(
    path: PlannedPath,
    *,
    rng: np.random.Generator | None = None,
) -> PathBOReference:
    if len(path) == 0:
        raise ValueError("planned path is empty")
    generator = rng if rng is not None else np.random.default_rng()
    index = int(generator.integers(0, len(path)))
    tangents = path_tangents(path.positions_base)
    return PathBOReference(
        path_index=index,
        position_base=np.asarray(path.positions_base[index], dtype=float).copy(),
        normal_base=normalize_vector(np.asarray(path.normals_base[index], dtype=float)),
        tangent_base=normalize_vector(np.asarray(tangents[index], dtype=float)),
    )
