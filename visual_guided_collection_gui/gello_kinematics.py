from __future__ import annotations

import numpy as np

from visual_guided_collection_gui.surface_teleop import matrix_to_rotvec


def gello_ur5_joint_action_to_tcp_pose(action: np.ndarray) -> np.ndarray:
    joints = np.asarray(action, dtype=float).reshape(-1)
    if joints.shape[0] < 6:
        raise ValueError(f"GELLO action must contain at least 6 joints, got {joints.shape[0]}")
    transform = _ur5_forward_kinematics(joints[:6])
    return np.concatenate([transform[:3, 3], matrix_to_rotvec(transform[:3, :3])])


def _ur5_forward_kinematics(joints: np.ndarray) -> np.ndarray:
    q = np.asarray(joints, dtype=float).reshape(6)
    a = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0], dtype=float)
    d = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823], dtype=float)
    alpha = np.array([np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0], dtype=float)
    transform = np.eye(4, dtype=float)
    for theta, ai, di, alphai in zip(q, a, d, alpha):
        transform = transform @ _dh_transform(theta, ai, di, alphai)
    return transform


def _dh_transform(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    ca = float(np.cos(alpha))
    sa = float(np.sin(alpha))
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
