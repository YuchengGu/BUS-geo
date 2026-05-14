from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np

from breast_path_planning.geometry import transform_points


@dataclass
class PointCloud:
    points_base: np.ndarray
    colors_rgb: np.ndarray | None = None
    pixels_uv: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points_base = np.asarray(self.points_base, dtype=float)
        if self.points_base.ndim != 2 or self.points_base.shape[1] != 3:
            raise ValueError(f"points_base must have shape (N, 3), got {self.points_base.shape}")
        if self.colors_rgb is not None:
            self.colors_rgb = np.asarray(self.colors_rgb)
            if self.colors_rgb.shape != self.points_base.shape:
                raise ValueError("colors_rgb must have shape (N, 3)")
            self.colors_rgb = self.colors_rgb.astype(np.uint8, copy=False)
        if self.pixels_uv is not None:
            self.pixels_uv = np.asarray(self.pixels_uv, dtype=int)
            if self.pixels_uv.shape != (self.points_base.shape[0], 2):
                raise ValueError("pixels_uv must have shape (N, 2)")

    def __len__(self) -> int:
        return int(self.points_base.shape[0])

    def subset(self, mask: np.ndarray) -> "PointCloud":
        valid = np.asarray(mask, dtype=bool)
        if valid.shape[0] != len(self):
            raise ValueError("mask length must match point cloud length")
        colors = None if self.colors_rgb is None else self.colors_rgb[valid]
        pixels = None if self.pixels_uv is None else self.pixels_uv[valid]
        return PointCloud(self.points_base[valid], colors, pixels)


def realsense_frames_to_point_cloud(
    color_frame: Any,
    depth_frame: Any,
    T_base_camera: np.ndarray,
    *,
    pointcloud: Any | None = None,
    stride: int = 2,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> PointCloud:
    """Generate a base-frame point cloud using librealsense pointcloud.

    This follows the librealsense pattern:

    ``pc.map_to(color_frame); points = pc.calculate(depth_frame)``.

    The RealSense SDK already uses the camera intrinsics internally. The returned
    vertices are in the D405 camera coordinate system and are converted here to
    UR base coordinates with ``T_base_camera``.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if color_frame is None or depth_frame is None:
        raise ValueError("color_frame and depth_frame are required")

    if pointcloud is None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - depends on hardware env
            raise ImportError("pyrealsense2 is required for RealSense SDK point cloud generation") from exc
        pointcloud = rs.pointcloud()

    pointcloud.map_to(color_frame)
    rs_points = pointcloud.calculate(depth_frame)
    vertices_camera = np.asanyarray(rs_points.get_vertices()).view(np.float32).reshape(-1, 3)
    texcoords = np.asanyarray(rs_points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color_frame.get_data())
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise ValueError(f"Expected color frame data with shape (H, W, 3), got {color_image.shape}")

    valid = np.isfinite(vertices_camera).all(axis=1)
    z = vertices_camera[:, 2]
    valid &= (z >= min_depth_m) & (z <= max_depth_m)
    if stride > 1:
        valid_indices = np.flatnonzero(valid)[::stride]
    else:
        valid_indices = np.flatnonzero(valid)
    if valid_indices.shape[0] == 0:
        return PointCloud(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 2), dtype=int))

    selected_camera = vertices_camera[valid_indices].astype(float, copy=False)
    selected_tex = texcoords[valid_indices]
    points_base = transform_points(selected_camera, T_base_camera)

    height, width = color_image.shape[:2]
    u = np.clip(np.rint(selected_tex[:, 0] * (width - 1)), 0, width - 1).astype(int)
    v = np.clip(np.rint(selected_tex[:, 1] * (height - 1)), 0, height - 1).astype(int)
    colors = color_image[v, u].astype(np.uint8, copy=False)
    pixels = np.stack([u, v], axis=1)
    return PointCloud(points_base, colors, pixels)


def save_point_cloud_ply(cloud: PointCloud, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    has_color = cloud.colors_rgb is not None
    with open(output, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cloud)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, point in enumerate(cloud.points_base):
            if has_color:
                r, g, b = cloud.colors_rgb[i]
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def load_point_cloud_ply(input_path: str | Path) -> PointCloud:
    path = Path(input_path)
    payload = path.read_bytes()
    marker = b"end_header\n"
    header_end = payload.find(marker)
    marker_len = len(marker)
    if header_end < 0:
        marker = b"end_header\r\n"
        header_end = payload.find(marker)
        marker_len = len(marker)
    if header_end < 0:
        raise ValueError("PLY file is missing end_header")

    header_text = payload[:header_end].decode("ascii", errors="strict")
    body = payload[header_end + marker_len :]
    lines = header_text.splitlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not an ASCII PLY file")
    format_line = next((line.strip() for line in lines if line.startswith("format ")), None)
    if format_line not in {"format ascii 1.0", "format binary_little_endian 1.0"}:
        raise ValueError(f"Unsupported PLY format: {format_line}")

    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines[1:]:
        pieces = line.strip().split()
        if not pieces:
            continue
        if pieces[:2] == ["element", "vertex"]:
            vertex_count = int(pieces[2])
            in_vertex = True
            continue
        if pieces[0] == "element":
            in_vertex = False
            continue
        if in_vertex and pieces[0] == "property":
            if pieces[1] == "list":
                raise ValueError("PLY list vertex properties are not supported")
            properties.append((pieces[1], pieces[-1]))

    if vertex_count is None:
        raise ValueError("PLY file is missing element vertex")
    property_names = [name for _dtype, name in properties]
    required = {"x", "y", "z"}
    if not required.issubset(set(property_names)):
        raise ValueError("PLY vertex properties must include x, y, z")

    prop_index = {name: i for i, name in enumerate(property_names)}
    color_names = None
    if {"red", "green", "blue"}.issubset(prop_index):
        color_names = ("red", "green", "blue")
    elif {"r", "g", "b"}.issubset(prop_index):
        color_names = ("r", "g", "b")

    if format_line == "format ascii 1.0":
        body_text = body.decode("ascii", errors="strict")
        data_lines = body_text.splitlines()[:vertex_count]
        if len(data_lines) != vertex_count:
            raise ValueError(f"PLY expected {vertex_count} vertices, got {len(data_lines)}")

        points = []
        colors = [] if color_names is not None else None
        for line in data_lines:
            pieces = line.strip().split()
            if len(pieces) < len(properties):
                raise ValueError("PLY vertex row has fewer values than properties")
            points.append([float(pieces[prop_index["x"]]), float(pieces[prop_index["y"]]), float(pieces[prop_index["z"]])])
            if color_names is not None and colors is not None:
                colors.append([int(float(pieces[prop_index[name]])) for name in color_names])
    else:
        dtype = _ply_vertex_dtype(properties)
        records = np.frombuffer(body, dtype=dtype, count=vertex_count)
        if records.shape[0] != vertex_count:
            raise ValueError(f"PLY expected {vertex_count} vertices, got {records.shape[0]}")
        points = np.stack([records["x"], records["y"], records["z"]], axis=1).astype(float)
        colors = None
        if color_names is not None:
            colors = np.stack([records[name] for name in color_names], axis=1).astype(np.uint8)

    color_array = None if colors is None else np.asarray(colors, dtype=np.uint8)
    return PointCloud(np.asarray(points, dtype=float), color_array)


def _ply_vertex_dtype(properties: list[tuple[str, str]]) -> np.dtype:
    type_map = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    fields = []
    for dtype_name, property_name in properties:
        if dtype_name not in type_map:
            raise ValueError(f"Unsupported PLY vertex property type: {dtype_name}")
        fields.append((property_name, type_map[dtype_name]))
    return np.dtype(fields)
