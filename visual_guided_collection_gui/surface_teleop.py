from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breast_path_planning.geometry import normalize_vector, rodrigues
from breast_path_planning.path_io import PlannedPath


DEFAULT_APPROACH_HEIGHT_M = 0.2
DEFAULT_CONTACT_HEIGHT_M = 0.05
DEFAULT_PROBE_LENGTH_M = 0.2


@dataclass
class SurfaceTeleopState:
    progress_m: float = 0.0
    lateral_offset_m: float = 0.0
    normal_offset_m: float = DEFAULT_CONTACT_HEIGHT_M


@dataclass
class SurfaceTcpTarget:
    probe_tip_position_base: np.ndarray
    tcp_position_base: np.ndarray
    tcp_rotation_base: np.ndarray
    frame_axis_mode: str = "world-y"

    def tcp_pose_rotvec(self) -> np.ndarray:
        return np.concatenate([self.tcp_position_base, matrix_to_rotvec(self.tcp_rotation_base)])


class SurfaceCartesianTeleopController:
    def __init__(
        self,
        *,
        path: PlannedPath,
        probe_length_m: float = DEFAULT_PROBE_LENGTH_M,
        translation_gains_xyz: np.ndarray | None = None,
        rotation_gains_xyz: np.ndarray | None = None,
        frame_axis_mode: str = "world-y",
        use_corner_frame_modes: bool = False,
    ) -> None:
        self.path = path
        self.probe_length_m = float(probe_length_m)
        self.frame_axis_mode = frame_axis_mode
        self.use_corner_frame_modes = bool(use_corner_frame_modes)
        self._corner_indices = {
            int(index)
            for index in path.metadata.get("corner_indices", [])
            if 0 <= int(index) < len(path.positions_base)
        }
        self.translation_gains_xyz = (
            np.ones(3, dtype=float)
            if translation_gains_xyz is None
            else np.asarray(translation_gains_xyz, dtype=float).reshape(3)
        )
        self.rotation_gains_xyz = (
            np.ones(3, dtype=float)
            if rotation_gains_xyz is None
            else np.asarray(rotation_gains_xyz, dtype=float).reshape(3)
        )
        self._path_tangents = path_tangents(path.positions_base)
        self._path_arclengths = path_arclengths(path.positions_base)
        self._last_gello_pose: np.ndarray | None = None
        self._neutral_gello_pose: np.ndarray | None = None
        self._calibrated_x_axis: np.ndarray | None = None
        self._input_axes: np.ndarray | None = None
        self._target_tcp_pose: np.ndarray | None = None
        self._base_progress_m = 0.0
        self._base_lateral_offset_m = 0.0
        self._base_normal_offset_m = DEFAULT_CONTACT_HEIGHT_M
        self._tcp_position_residual_base = np.zeros(3, dtype=float)
        self._tcp_rotation_residual_local = np.eye(3, dtype=float)
        self._clutch_enabled = False

    @property
    def initialized(self) -> bool:
        return self._last_gello_pose is not None and self._target_tcp_pose is not None

    @property
    def input_axes_ready(self) -> bool:
        return self._input_axes is not None

    @property
    def clutch_enabled(self) -> bool:
        return self._clutch_enabled

    def set_clutch(self, enabled: bool) -> None:
        self._clutch_enabled = bool(enabled)

    def set_neutral(self, *, gello_tcp_pose: np.ndarray, ur_tcp_pose: np.ndarray) -> np.ndarray:
        gello_pose = np.asarray(gello_tcp_pose, dtype=float).reshape(6)
        tcp_pose = np.asarray(ur_tcp_pose, dtype=float).reshape(6)
        self._neutral_gello_pose = gello_pose.copy()
        self._calibrated_x_axis = None
        self._input_axes = None
        self._last_gello_pose = gello_pose.copy()
        self._sync_target_to_tcp_pose(tcp_pose)
        self._clutch_enabled = False
        return tcp_pose.copy()

    def reset(self, *, gello_tcp_pose: np.ndarray, ur_tcp_pose: np.ndarray) -> np.ndarray:
        return self.set_neutral(gello_tcp_pose=gello_tcp_pose, ur_tcp_pose=ur_tcp_pose)

    def calibrate_x(self, gello_tcp_pose: np.ndarray) -> np.ndarray:
        if self._neutral_gello_pose is None:
            raise RuntimeError("Set neutral before calibrating +X")
        gello_pose = np.asarray(gello_tcp_pose, dtype=float).reshape(6)
        axis = normalize_vector(gello_pose[:3] - self._neutral_gello_pose[:3])
        self._calibrated_x_axis = axis
        self._input_axes = None
        return axis.copy()

    def calibrate_z(self, gello_tcp_pose: np.ndarray) -> np.ndarray:
        if self._neutral_gello_pose is None:
            raise RuntimeError("Set neutral before calibrating +Z")
        if self._calibrated_x_axis is None:
            raise RuntimeError("Calibrate +X before calibrating +Z")
        gello_pose = np.asarray(gello_tcp_pose, dtype=float).reshape(6)
        x_axis = self._calibrated_x_axis
        z_raw = normalize_vector(gello_pose[:3] - self._neutral_gello_pose[:3])
        z_axis = normalize_vector(z_raw - np.dot(z_raw, x_axis) * x_axis)
        y_axis = normalize_vector(np.cross(z_axis, x_axis))
        self._input_axes = np.column_stack([x_axis, y_axis, z_axis])
        return z_axis.copy()

    def recenter(self, *, gello_tcp_pose: np.ndarray, ur_tcp_pose: np.ndarray) -> np.ndarray:
        gello_pose = np.asarray(gello_tcp_pose, dtype=float).reshape(6)
        tcp_pose = np.asarray(ur_tcp_pose, dtype=float).reshape(6)
        self._last_gello_pose = gello_pose.copy()
        self._neutral_gello_pose = gello_pose.copy()
        self._sync_target_to_tcp_pose(tcp_pose)
        return self._target_tcp_pose.copy()

    def update(self, *, gello_tcp_pose: np.ndarray, ur_tcp_pose: np.ndarray) -> np.ndarray:
        gello_pose = np.asarray(gello_tcp_pose, dtype=float).reshape(6)
        tcp_pose = np.asarray(ur_tcp_pose, dtype=float).reshape(6)
        if not self.initialized:
            return self.set_neutral(gello_tcp_pose=gello_pose, ur_tcp_pose=tcp_pose)
        if self._input_axes is None:
            raise RuntimeError("Calibrate +X and +Z before surface Cartesian teleop")
        if self._clutch_enabled:
            return self.recenter(gello_tcp_pose=gello_pose, ur_tcp_pose=tcp_pose)

        assert self._last_gello_pose is not None
        assert self._target_tcp_pose is not None
        assert self._neutral_gello_pose is not None

        neutral_gello_pose = self._neutral_gello_pose
        absolute_gello_translation = self._input_axes.T @ (gello_pose[:3] - neutral_gello_pose[:3])
        absolute_gello_translation = absolute_gello_translation * self.translation_gains_xyz
        progress_m = self._base_progress_m + absolute_gello_translation[0]
        lateral_offset_m = self._base_lateral_offset_m - absolute_gello_translation[1]
        normal_offset_m = self._base_normal_offset_m + absolute_gello_translation[2]

        path_position, tangent, normal = self._sample_path(progress_m)
        frame_axis_mode = self._frame_axis_mode_for_actual_tip(tangent, normal, tcp_pose)
        base_target = build_tcp_target(
            path_position,
            tangent,
            normal,
            SurfaceTeleopState(
                progress_m=progress_m,
                lateral_offset_m=lateral_offset_m,
                normal_offset_m=normal_offset_m,
            ),
            probe_length_m=self.probe_length_m,
            frame_axis_mode=frame_axis_mode,
        )

        absolute_rotation_base = rodrigues(gello_pose[3:6]) @ rodrigues(neutral_gello_pose[3:6]).T
        delta_rotation_reference = self._input_axes.T @ matrix_to_rotvec(absolute_rotation_base)
        delta_tcp_local = gello_rotation_increment_to_tcp_local(
            delta_rotation_reference,
            self.rotation_gains_xyz,
        )

        next_position = base_target.tcp_position_base + self._tcp_position_residual_base
        next_rotation = base_target.tcp_rotation_base @ self._tcp_rotation_residual_local @ rodrigues(delta_tcp_local)
        next_pose = np.concatenate([next_position, matrix_to_rotvec(next_rotation)])
        self._last_gello_pose = gello_pose.copy()
        self._target_tcp_pose = next_pose.copy()
        return next_pose

    def _nearest_path_frame(self, probe_tip_position_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(self.path.positions_base, dtype=float)
        normals = np.asarray(self.path.normals_base, dtype=float)
        probe = np.asarray(probe_tip_position_base, dtype=float).reshape(3)
        index = int(np.argmin(np.linalg.norm(positions - probe, axis=1)))
        return self._path_tangents[index], normals[index]

    def _nearest_path_state(self, probe_tip_position_base: np.ndarray) -> tuple[float, float, float]:
        positions = np.asarray(self.path.positions_base, dtype=float)
        normals = np.asarray(self.path.normals_base, dtype=float)
        probe = np.asarray(probe_tip_position_base, dtype=float).reshape(3)
        index = int(np.argmin(np.linalg.norm(positions - probe, axis=1)))
        tangent = self._path_tangents[index]
        normal = normals[index]
        b_axis = build_tnb_frame(tangent, normal)[:, 2]
        offset = probe - positions[index]
        return (
            float(self._path_arclengths[index]),
            float(np.dot(offset, b_axis)),
            float(np.dot(offset, normal)),
        )

    def _nearest_path_point_index(self, probe_tip_position_base: np.ndarray) -> int:
        positions = np.asarray(self.path.positions_base, dtype=float)
        probe = np.asarray(probe_tip_position_base, dtype=float).reshape(3)
        return int(np.argmin(np.linalg.norm(positions - probe, axis=1)))

    def _sample_path(self, progress_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(self.path.positions_base, dtype=float)
        normals = np.asarray(self.path.normals_base, dtype=float)
        progress = float(np.clip(progress_m, self._path_arclengths[0], self._path_arclengths[-1]))
        if len(positions) == 1:
            return positions[0].copy(), self._path_tangents[0].copy(), normals[0].copy()
        segment = int(np.searchsorted(self._path_arclengths, progress, side="right") - 1)
        segment = int(np.clip(segment, 0, len(positions) - 2))
        segment_start = self._path_arclengths[segment]
        segment_length = self._path_arclengths[segment + 1] - segment_start
        ratio = 0.0 if segment_length <= 1e-12 else (progress - segment_start) / segment_length
        position = positions[segment] + ratio * (positions[segment + 1] - positions[segment])
        if segment_length <= 1e-12:
            tangent = self._path_tangents[segment]
        else:
            tangent = normalize_vector(
                (1.0 - ratio) * self._path_tangents[segment] + ratio * self._path_tangents[segment + 1],
                fallback=positions[segment + 1] - positions[segment],
            )
        normal = normalize_vector(
            normals[segment] + ratio * (normals[segment + 1] - normals[segment]),
            fallback=normals[segment],
        )
        return position, tangent, normal

    def _set_base_path_state_from_tcp_pose(self, tcp_pose: np.ndarray) -> None:
        pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        rotation = rodrigues(pose[3:6])
        probe_tip = pose[:3] + self.probe_length_m * rotation[:, 2]
        progress, lateral_offset, normal_offset = self._nearest_path_state(probe_tip)
        self._base_progress_m = progress
        self._base_lateral_offset_m = lateral_offset
        self._base_normal_offset_m = normal_offset

    def _sync_target_to_tcp_pose(self, tcp_pose: np.ndarray) -> None:
        pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        self._target_tcp_pose = pose.copy()
        self._set_base_path_state_from_tcp_pose(pose)
        path_position, tangent, normal = self._sample_path(self._base_progress_m)
        frame_axis_mode = self._frame_axis_mode_for_actual_tip(tangent, normal, pose)
        reference = build_tcp_target(
            path_position,
            tangent,
            normal,
            SurfaceTeleopState(
                progress_m=self._base_progress_m,
                lateral_offset_m=self._base_lateral_offset_m,
                normal_offset_m=self._base_normal_offset_m,
            ),
            probe_length_m=self.probe_length_m,
            frame_axis_mode=frame_axis_mode,
        )
        rotation = rodrigues(pose[3:6])
        self._tcp_position_residual_base = pose[:3] - reference.tcp_position_base
        self._tcp_rotation_residual_local = reference.tcp_rotation_base.T @ rotation

    def _frame_axis_mode_for_actual_tip(self, tangent: np.ndarray, normal: np.ndarray, tcp_pose: np.ndarray) -> str:
        if not self.use_corner_frame_modes or not self._corner_indices:
            return self.frame_axis_mode
        pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        rotation = rodrigues(pose[3:6])
        return choose_tcp_frame_axis_mode_for_rotation(
            tangent,
            normal,
            rotation,
            allowed_modes=("world-y", "-world-y"),
        )


def build_tnb_frame(tangent_base: np.ndarray, normal_base: np.ndarray) -> np.ndarray:
    n = normalize_vector(normal_base, fallback=np.array([0.0, 0.0, 1.0]))
    raw_t = normalize_vector(tangent_base, fallback=np.array([1.0, 0.0, 0.0]))
    t = normalize_vector(raw_t - np.dot(raw_t, n) * n, fallback=np.array([1.0, 0.0, 0.0]))
    if abs(float(np.dot(t, n))) > 1e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        t = normalize_vector(fallback - np.dot(fallback, n) * n, fallback=np.array([1.0, 0.0, 0.0]))
    b = normalize_vector(np.cross(t, n), fallback=np.array([0.0, -1.0, 0.0]))
    t = normalize_vector(np.cross(n, b), fallback=t)
    return np.column_stack([t, n, b])


def build_tcp_frame(
    tangent_base: np.ndarray,
    normal_base: np.ndarray,
    preferred_x_axis_base: np.ndarray | None = None,
    frame_axis_mode: str | None = None,
) -> np.ndarray:
    raw_n = normalize_vector(normal_base, fallback=np.array([0.0, 0.0, 1.0]))
    tnb = build_tnb_frame(tangent_base, normal_base)
    t = tnb[:, 0]
    n = tnb[:, 1]
    b = tnb[:, 2]
    if frame_axis_mode is None:
        frame_axis_mode = "world-y" if preferred_x_axis_base is None else choose_tcp_frame_axis_mode(t, raw_n, preferred_x_axis_base, allowed_modes=("world-y", "-world-y"))
    if frame_axis_mode in {"world-y", "-world-y"}:
        return build_tcp_frame_with_x_projected_from_world_y(raw_n, reverse=frame_axis_mode.startswith("-"))
    if frame_axis_mode == "global-y":
        return build_tcp_frame_with_y_perpendicular_to_global_scan(n)

    x_axis, y_axis = _tcp_frame_xy_axes(t, b, frame_axis_mode)
    return np.column_stack([x_axis, y_axis, -n])


def build_tcp_frame_with_x_projected_from_world_y(
    normal_base: np.ndarray,
    *,
    reverse: bool = False,
    world_y_base: np.ndarray | None = None,
) -> np.ndarray:
    n = normalize_vector(normal_base, fallback=np.array([0.0, 0.0, 1.0]))
    if world_y_base is None:
        world_y_base = np.array([0.0, 1.0, 0.0], dtype=float)
    world_y = normalize_vector(world_y_base, fallback=np.array([0.0, 1.0, 0.0]))
    fallback = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(fallback, n))) > 0.95:
        fallback = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = normalize_vector(world_y - np.dot(world_y, n) * n, fallback=fallback - np.dot(fallback, n) * n)
    if reverse:
        x_axis = -x_axis
    z_axis = -n
    y_axis = normalize_vector(np.cross(z_axis, x_axis), fallback=np.array([1.0, 0.0, 0.0]))
    x_axis = normalize_vector(np.cross(y_axis, z_axis), fallback=x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def build_tcp_frame_with_y_perpendicular_to_global_scan(
    normal_base: np.ndarray,
    *,
    global_scan_direction_base: np.ndarray | None = None,
) -> np.ndarray:
    n = normalize_vector(normal_base, fallback=np.array([0.0, 0.0, 1.0]))
    if global_scan_direction_base is None:
        global_scan_direction_base = np.array([1.0, 0.0, 0.0], dtype=float)
    scan = normalize_vector(global_scan_direction_base, fallback=np.array([1.0, 0.0, 0.0]))
    base_z = np.array([0.0, 0.0, 1.0], dtype=float)
    perpendicular = normalize_vector(np.cross(base_z, scan), fallback=np.array([0.0, 1.0, 0.0]))
    y_axis = normalize_vector(perpendicular - np.dot(perpendicular, n) * n, fallback=np.array([0.0, 1.0, 0.0]))
    z_axis = -n
    x_axis = normalize_vector(np.cross(y_axis, z_axis), fallback=np.array([1.0, 0.0, 0.0]))
    y_axis = normalize_vector(np.cross(z_axis, x_axis), fallback=y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def choose_tcp_frame_axis_mode(
    tangent_base: np.ndarray,
    normal_base: np.ndarray,
    preferred_x_axis_base: np.ndarray,
    *,
    allowed_modes: tuple[str, ...] = ("t", "-t", "b", "-b"),
) -> str:
    tnb = build_tnb_frame(tangent_base, normal_base)
    t = tnb[:, 0]
    b = tnb[:, 2]
    preferred = normalize_vector(preferred_x_axis_base, fallback=-t)

    def candidate_x(mode: str) -> np.ndarray:
        if mode in {"world-y", "-world-y"}:
            return build_tcp_frame(t, normal_base, frame_axis_mode=mode)[:, 0]
        return _tcp_frame_xy_axes(t, b, mode)[0]

    return max(allowed_modes, key=lambda mode: float(np.dot(preferred, candidate_x(mode))))


def choose_tcp_frame_axis_mode_for_rotation(
    tangent_base: np.ndarray,
    normal_base: np.ndarray,
    reference_rotation_base: np.ndarray,
    *,
    allowed_modes: tuple[str, ...] = ("t", "-t", "b", "-b"),
) -> str:
    reference = np.asarray(reference_rotation_base, dtype=float).reshape(3, 3)
    if not allowed_modes:
        raise ValueError("allowed_modes must not be empty")

    def rotation_distance(mode: str) -> float:
        candidate = build_tcp_frame(tangent_base, normal_base, frame_axis_mode=mode)
        return float(np.linalg.norm(matrix_to_rotvec(candidate @ reference.T)))

    return min(allowed_modes, key=rotation_distance)


def _tcp_frame_xy_axes(tangent: np.ndarray, binormal: np.ndarray, frame_axis_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if frame_axis_mode == "t":
        return tangent, binormal
    if frame_axis_mode == "-t":
        return -tangent, -binormal
    if frame_axis_mode == "b":
        return binormal, -tangent
    if frame_axis_mode == "-b":
        return -binormal, tangent
    raise ValueError(f"Unsupported TCP frame axis mode: {frame_axis_mode!r}")


def gello_translation_to_surface_delta(
    delta_gello_xyz: np.ndarray,
    tangent_base: np.ndarray,
    normal_base: np.ndarray,
    gains_xyz: np.ndarray | None = None,
) -> np.ndarray:
    if gains_xyz is None:
        gains_xyz = np.ones(3, dtype=float)
    delta = np.asarray(delta_gello_xyz, dtype=float).reshape(3) * np.asarray(gains_xyz, dtype=float).reshape(3)
    tnb = build_tnb_frame(tangent_base, normal_base)
    t = tnb[:, 0]
    n = tnb[:, 1]
    b = tnb[:, 2]
    return delta[0] * t - delta[1] * b + delta[2] * n


def gello_rotation_increment_to_tcp_local(
    delta_theta_gello_xyz: np.ndarray,
    gains_xyz: np.ndarray | None = None,
) -> np.ndarray:
    if gains_xyz is None:
        gains_xyz = np.ones(3, dtype=float)
    delta = np.asarray(delta_theta_gello_xyz, dtype=float).reshape(3) * np.asarray(gains_xyz, dtype=float).reshape(3)
    return np.array([delta[1], delta[0], -delta[2]], dtype=float)


def build_tcp_target(
    path_position_base: np.ndarray,
    tangent_base: np.ndarray,
    normal_base: np.ndarray,
    state: SurfaceTeleopState,
    *,
    probe_length_m: float = DEFAULT_PROBE_LENGTH_M,
    tcp_local_rotation_increment: np.ndarray | None = None,
    preferred_tcp_x_axis_base: np.ndarray | None = None,
    frame_axis_mode: str | None = None,
) -> SurfaceTcpTarget:
    tnb = build_tnb_frame(tangent_base, normal_base)
    n = tnb[:, 1]
    b = tnb[:, 2]
    probe_tip = (
        np.asarray(path_position_base, dtype=float).reshape(3)
        + float(state.lateral_offset_m) * b
        + float(state.normal_offset_m) * n
    )
    if frame_axis_mode is None and preferred_tcp_x_axis_base is not None:
        frame_axis_mode = choose_tcp_frame_axis_mode(tnb[:, 0], n, preferred_tcp_x_axis_base, allowed_modes=("world-y", "-world-y"))
    if frame_axis_mode is None:
        frame_axis_mode = "world-y"
    base_rotation = build_tcp_frame(tnb[:, 0], n, frame_axis_mode=frame_axis_mode)
    if tcp_local_rotation_increment is not None:
        base_rotation = base_rotation @ rodrigues(np.asarray(tcp_local_rotation_increment, dtype=float).reshape(3))
    tcp_position = probe_tip + float(probe_length_m) * n
    return SurfaceTcpTarget(probe_tip, tcp_position, base_rotation, frame_axis_mode=frame_axis_mode)


def path_start_tcp_targets(
    path: PlannedPath,
    *,
    approach_height_m: float = DEFAULT_APPROACH_HEIGHT_M,
    contact_height_m: float = DEFAULT_CONTACT_HEIGHT_M,
    probe_length_m: float = DEFAULT_PROBE_LENGTH_M,
    preferred_tcp_x_axis_base: np.ndarray | None = None,
    frame_axis_mode: str | None = None,
) -> tuple[SurfaceTcpTarget, SurfaceTcpTarget]:
    positions = np.asarray(path.positions_base, dtype=float)
    normals = np.asarray(path.normals_base, dtype=float)
    if len(path) < 1:
        raise ValueError("path must contain at least one point")
    if len(path) >= 2:
        tangent = positions[1] - positions[0]
    else:
        tangent = np.array([1.0, 0.0, 0.0], dtype=float)
    normal = normals[0]
    if frame_axis_mode is None:
        frame_axis_mode = (
            choose_tcp_frame_axis_mode(tangent, normal, preferred_tcp_x_axis_base, allowed_modes=("world-y", "-world-y"))
            if preferred_tcp_x_axis_base is not None
            else "world-y"
        )
    pre = build_tcp_target(
        positions[0],
        tangent,
        normal,
        SurfaceTeleopState(normal_offset_m=approach_height_m),
        probe_length_m=probe_length_m,
        frame_axis_mode=frame_axis_mode,
    )
    start = build_tcp_target(
        positions[0],
        tangent,
        normal,
        SurfaceTeleopState(normal_offset_m=contact_height_m),
        probe_length_m=probe_length_m,
        frame_axis_mode=pre.frame_axis_mode,
    )
    return pre, start


def staged_surface_start_tcp_sequence(
    current_tcp_pose: np.ndarray,
    pre_target: SurfaceTcpTarget,
    start_target: SurfaceTcpTarget,
) -> list[np.ndarray]:
    current = np.asarray(current_tcp_pose, dtype=float).reshape(6)
    pre_pose = pre_target.tcp_pose_rotvec()
    start_pose = start_target.tcp_pose_rotvec()
    mid_position = 0.5 * (current[:3] + pre_pose[:3])
    mid_translate_pose = np.concatenate([mid_position, current[3:]])
    mid_rotate_pose = np.concatenate([mid_position, start_pose[3:]])
    return [
        mid_translate_pose,
        mid_rotate_pose,
        np.concatenate([pre_pose[:3], start_pose[3:]]),
        start_pose,
    ]


def first_darboux_scan_line_tcp_poses(
    path: PlannedPath,
    *,
    contact_height_m: float = DEFAULT_CONTACT_HEIGHT_M,
    probe_length_m: float = DEFAULT_PROBE_LENGTH_M,
    frame_axis_mode: str = "world-y",
) -> list[np.ndarray]:
    positions = np.asarray(path.positions_base, dtype=float)
    normals = np.asarray(path.normals_base, dtype=float)
    if len(path) < 1:
        raise ValueError("path must contain at least one point")
    line_positions = positions
    line_normals = normals
    line_tangents = path_tangents(line_positions)
    poses: list[np.ndarray] = []
    for position, tangent, normal in zip(line_positions, line_tangents, line_normals):
        target = build_tcp_target(
            position,
            tangent,
            normal,
            SurfaceTeleopState(normal_offset_m=contact_height_m),
            probe_length_m=probe_length_m,
            frame_axis_mode=frame_axis_mode,
        )
        poses.append(target.tcp_pose_rotvec())
    return poses


def path_tangents(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("path must contain at least one point")
    if len(points) == 1:
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (1, 1))
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    return np.vstack([normalize_vector(t, fallback=np.array([1.0, 0.0, 0.0])) for t in tangents])


def path_arclengths(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("path must contain at least one point")
    if len(points) == 1:
        return np.zeros(1, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def interpolate_tcp_poses(
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    max_position_step_m: float = 0.01,
    max_rotation_step_rad: float = 0.05,
) -> list[np.ndarray]:
    start = np.asarray(start_pose, dtype=float).reshape(6)
    target = np.asarray(target_pose, dtype=float).reshape(6)
    if max_position_step_m <= 0.0:
        raise ValueError("max_position_step_m must be positive")
    if max_rotation_step_rad <= 0.0:
        raise ValueError("max_rotation_step_rad must be positive")

    position_delta = target[:3] - start[:3]
    start_rotation = rodrigues(start[3:])
    target_rotation = rodrigues(target[3:])
    relative_rotvec = matrix_to_rotvec(target_rotation @ start_rotation.T)
    position_steps = int(np.ceil(np.linalg.norm(position_delta) / float(max_position_step_m)))
    rotation_steps = int(np.ceil(np.linalg.norm(relative_rotvec) / float(max_rotation_step_rad)))
    steps = max(1, position_steps, rotation_steps)
    poses = []
    for i in range(1, steps + 1):
        alpha = i / steps
        position = start[:3] + alpha * position_delta
        rotation = rodrigues(alpha * relative_rotvec) @ start_rotation
        poses.append(np.concatenate([position, matrix_to_rotvec(rotation)]))
    poses[-1] = target.copy()
    return poses


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_angle = (float(np.trace(value)) - 1.0) / 2.0
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3, dtype=float)
    if np.pi - angle < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(value) + 1.0, 0.0) / 2.0)
        if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > 1e-12:
            axis[1] = value[0, 1] / (2.0 * axis[0])
            axis[2] = value[0, 2] / (2.0 * axis[0])
        elif axis[1] >= axis[2] and axis[1] > 1e-12:
            axis[0] = value[0, 1] / (2.0 * axis[1])
            axis[2] = value[1, 2] / (2.0 * axis[1])
        elif axis[2] > 1e-12:
            axis[0] = value[0, 2] / (2.0 * axis[2])
            axis[1] = value[1, 2] / (2.0 * axis[2])
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        return normalize_vector(axis, fallback=np.array([1.0, 0.0, 0.0])) * angle
    axis = np.array(
        [
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ],
        dtype=float,
    ) / (2.0 * np.sin(angle))
    return axis * angle
