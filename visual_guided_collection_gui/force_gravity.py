from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from breast_path_planning.geometry import rodrigues


@dataclass(frozen=True)
class ForceGravityCalibration:
    force_bias_sensor: np.ndarray
    gravity_base: np.ndarray
    torque_bias_sensor: np.ndarray
    sample_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_bias_sensor", _as_vector(self.force_bias_sensor, 3))
        object.__setattr__(self, "gravity_base", _as_vector(self.gravity_base, 3))
        object.__setattr__(self, "torque_bias_sensor", _as_vector(self.torque_bias_sensor, 3))

    @property
    def bias_wrench(self) -> np.ndarray:
        return np.concatenate([self.force_bias_sensor, self.torque_bias_sensor])

    def gravity_force_sensor(self, tcp_rotvec: np.ndarray) -> np.ndarray:
        rotation_base_tcp = rodrigues(tcp_rotvec)
        return rotation_base_tcp.T @ self.gravity_base

    def compensate(self, raw_wrench: np.ndarray, tcp_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = _as_vector(raw_wrench, 6)
        pose = _as_vector(tcp_pose, 6)
        gravity_wrench = np.zeros(6, dtype=float)
        gravity_wrench[:3] = self.gravity_force_sensor(pose[3:6])
        compensated = raw - self.bias_wrench - gravity_wrench
        return compensated, self.bias_wrench.copy(), gravity_wrench


@dataclass(frozen=True)
class DemoForceGravityCompensation:
    mass_kg: float
    com_sensor_m: np.ndarray
    force_bias_sensor: np.ndarray
    torque_bias_sensor: np.ndarray
    rotation_tcp_sensor: np.ndarray
    sample_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "com_sensor_m", _as_vector(self.com_sensor_m, 3))
        object.__setattr__(self, "force_bias_sensor", _as_vector(self.force_bias_sensor, 3))
        object.__setattr__(self, "torque_bias_sensor", _as_vector(self.torque_bias_sensor, 3))
        rotation = np.asarray(self.rotation_tcp_sensor, dtype=float).reshape(3, 3)
        object.__setattr__(self, "rotation_tcp_sensor", rotation)

    @property
    def bias_wrench(self) -> np.ndarray:
        return np.concatenate([self.force_bias_sensor, self.torque_bias_sensor])

    @property
    def gravity_base(self) -> np.ndarray:
        return np.array([0.0, 0.0, -9.80665 * float(self.mass_kg)], dtype=float)

    def gravity_wrench_sensor(self, tcp_pose: np.ndarray) -> np.ndarray:
        pose = _as_vector(tcp_pose, 6)
        rotation_base_tcp = rodrigues(pose[3:6])
        rotation_base_sensor = rotation_base_tcp @ self.rotation_tcp_sensor
        gravity_force_sensor = rotation_base_sensor.T @ self.gravity_base
        gravity_torque_sensor = np.cross(self.com_sensor_m, gravity_force_sensor)
        gravity_wrench = np.zeros(6, dtype=float)
        gravity_wrench[:3] = gravity_force_sensor
        gravity_wrench[3:6] = gravity_torque_sensor
        return gravity_wrench

    def with_bias(self, bias_wrench: np.ndarray) -> "DemoForceGravityCompensation":
        bias = _as_vector(bias_wrench, 6)
        return DemoForceGravityCompensation(
            mass_kg=float(self.mass_kg),
            com_sensor_m=self.com_sensor_m.copy(),
            force_bias_sensor=bias[:3],
            torque_bias_sensor=bias[3:6],
            rotation_tcp_sensor=self.rotation_tcp_sensor.copy(),
            sample_count=int(self.sample_count),
        )

    def compensate(self, raw_wrench: np.ndarray, tcp_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = _as_vector(raw_wrench, 6)
        gravity_wrench = self.gravity_wrench_sensor(tcp_pose)
        compensated = raw - self.bias_wrench - gravity_wrench
        return compensated, self.bias_wrench.copy(), gravity_wrench


def demo1_force_gravity_compensation() -> DemoForceGravityCompensation:
    """Return the hard-coded model used by demo1's force-control loop."""
    rotation_tcp_sensor = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return DemoForceGravityCompensation(
        mass_kg=0.69177,
        com_sensor_m=np.array([-0.00324762, -0.0281622, 0.0304605], dtype=float),
        force_bias_sensor=np.zeros(3, dtype=float),
        torque_bias_sensor=np.zeros(3, dtype=float),
        rotation_tcp_sensor=rotation_tcp_sensor,
    )


def fit_force_gravity_calibration(tcp_poses: np.ndarray, raw_wrenches: np.ndarray) -> ForceGravityCalibration:
    poses = np.asarray(tcp_poses, dtype=float)
    wrenches = np.asarray(raw_wrenches, dtype=float)
    if poses.ndim != 2 or poses.shape[1] < 6:
        raise ValueError(f"Expected tcp_poses shape (N, 6), got {poses.shape}")
    if wrenches.ndim != 2 or wrenches.shape[1] < 6:
        raise ValueError(f"Expected raw_wrenches shape (N, 6), got {wrenches.shape}")
    if poses.shape[0] != wrenches.shape[0]:
        raise ValueError("tcp_poses and raw_wrenches must have the same number of samples")
    if poses.shape[0] < 3:
        raise ValueError("At least three no-contact poses are required")

    matrix_rows = []
    force_values = []
    for pose, wrench in zip(poses, wrenches, strict=True):
        rotation_base_tcp = rodrigues(pose[3:6])
        matrix_rows.append(np.concatenate([np.eye(3, dtype=float), rotation_base_tcp.T], axis=1))
        force_values.append(wrench[:3])
    design = np.vstack(matrix_rows)
    targets = np.concatenate(force_values)
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)

    torque_bias = np.mean(wrenches[:, 3:6], axis=0)
    return ForceGravityCalibration(
        force_bias_sensor=solution[:3],
        gravity_base=solution[3:6],
        torque_bias_sensor=torque_bias,
        sample_count=int(poses.shape[0]),
    )


def save_force_gravity_calibration(path: str | Path, calibration: ForceGravityCalibration) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        force_bias_sensor=calibration.force_bias_sensor,
        gravity_base=calibration.gravity_base,
        torque_bias_sensor=calibration.torque_bias_sensor,
        sample_count=np.array([calibration.sample_count], dtype=int),
        model=np.array(["raw_force = force_bias_sensor + R_base_tcp.T @ gravity_base + contact_force"]),
        frame_assumption=np.array(["force_sensor_frame_equals_tcp_frame"]),
    )


def load_force_gravity_calibration(path: str | Path) -> ForceGravityCalibration:
    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as data:
        sample_count = int(np.asarray(data.get("sample_count", [0]), dtype=int).reshape(-1)[0])
        return ForceGravityCalibration(
            force_bias_sensor=np.asarray(data["force_bias_sensor"], dtype=float),
            gravity_base=np.asarray(data["gravity_base"], dtype=float),
            torque_bias_sensor=np.asarray(data["torque_bias_sensor"], dtype=float),
            sample_count=sample_count,
        )


def _as_vector(value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < size:
        raise ValueError(f"Expected at least {size} values, got {array.size}")
    return array[:size].copy()
