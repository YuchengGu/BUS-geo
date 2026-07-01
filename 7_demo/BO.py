import numpy as np
from skopt import gp_minimize
import SimpleITK as sitk

# ---------------- ① 配置 ----------------
IS_SIMULATION   = True
SIMULATION_MHA  = r"C:\Users\31588\Desktop\SJTU\CTP\250427_YangJiang_43\Capture_R.mha"
L_probe         = 0.04          # 40 mm
f_max           = 4.0           # N
τz_max          = 0.05          # N·m
λ_f, λ_τ, λ_hodge, λ_κ = 0.00, 0.001, 0.001, 0.001   # λ_κ 新加

# ---------------- ② 模拟数据 ----------------
qualities   = np.array([0.1, 0.35, 0.65, 0.45, 0.6, 0.75,
                        0.45, 0.45, 0.4, 0.35, 0.65, 0.75, 0.69, 0.80, 0.78])
current_call = 0

# ---------------- ③ 术前已知（当前目标点） ----------------
n_skin  = np.array([0, 0, 1.0])     # 替换为真实网格法向
kappa_G = 0.18                            # 替换为真实高斯曲率 mm⁻²

# ---------------- ④ 探头法向计算 ----------------
def euler_to_probe_normal(yaw, pitch, roll):
    y, p, r = yaw, pitch, roll
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,         0,        1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0,        1, 0],
                   [-np.sin(p),0, np.cos(p)]])
    Rx = np.array([[1, 0,        0],
                   [0, np.cos(r),-np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    return (Rz @ Ry @ Rx) @ np.array([0., 0., 1.])

# ---------------- ⑤ 模拟辅助 ----------------
def get_simulated_quality(yaw, pitch, roll, depth):
    global current_call
    if current_call < len(qualities):
        q = qualities[current_call]
        current_call += 1
        return q
    return 0.0

def get_simulated_forces(yaw, pitch, roll, depth):
    # 让力也随参数变化
    fz = 2.0 + depth + 0.1 * abs(pitch)
    tx = 0.05 * roll + 0.01 * yaw
    ty = 0.05 * pitch + 0.01 * roll
    tz = 0.05 * yaw + 0.01 * pitch
    return np.array([0., 0., fz, tx, ty, tz])

# ---------------- ⑥ Hodge 分解 ----------------
def hodge_decompose_force(F_vec, τ_vec, n_skin):
    F_n   = np.dot(F_vec, n_skin)
    F_tang = F_vec - F_n * n_skin
    τ_n   = np.dot(τ_vec, n_skin)
    τ_tang = τ_vec - τ_n * n_skin
    return F_n, F_tang, τ_n, τ_tang

# ---------------- ⑦ 目标函数 ----------------
def objective(params):
    yaw, pitch, roll, depth = params

    # 7.1 信号
    quality = get_simulated_quality(yaw, pitch, roll, depth)
    forces  = get_simulated_forces(yaw, pitch, roll, depth)
    fx, fy, fz, tx, ty, tz = forces

    # 7.2 几何分解
    F_n, F_tang, τ_n, τ_tang = hodge_decompose_force(
        np.array([fx, fy, fz]), np.array([tx, ty, tz]), n_skin)

    # 7.3 力/矩惩罚
    Fn = max(0.0001, np.linalg.norm(F_n))
    P_force  = (np.abs(fx)+np.abs(fy))/fz + max(0.001, fz)**2/f_max**2
    P_torque = (np.abs(tx)+np.abs(ty))/fz + max(0.001, np.abs(tz))**2/τz_max**2
    P_hodge  = (np.linalg.norm(F_tang)/Fn + np.linalg.norm(τ_tang)/(Fn*L_probe))

    # 7.4 曲率-角度惩罚
    n_probe   = euler_to_probe_normal(yaw, pitch, roll)
    cos_theta = np.clip(np.dot(n_probe, n_skin), 0, 1)
    theta     = np.arccos(cos_theta)
    kappa_norm= np.tanh(kappa_G / 0.05)      # κ_ref = 0.05 mm⁻²
    theta_ref = 5.0 * np.pi / 180.0          # 5°
    P_kappa   = kappa_norm * (theta / theta_ref)**2

    # 7.5 总目标
    noise = 0.01 * np.random.randn()
    F = quality - P_force - P_torque - λ_hodge*P_hodge - λ_κ*P_kappa + noise
    print(f"[DEBUG] yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}, depth={depth:.2f} → F={F:.3f}")
    return -F


# ---------------- ⑧ 主流程 ----------------
if __name__ == "__main__":
    yaw = 0
    pitch = 0.234
    roll = 0.48
    depth = 1
    init = [yaw, pitch, roll, depth]
    search_space = [(-15, 15), (-10, 10), (-5, 5), (0, 14)]
    result = gp_minimize(objective, search_space, x0=init, 
                         n_calls=14, acq_func="EI", random_state=42)
    best_yaw, best_pitch, best_roll, best_depth = result.x
    best_Q = -result.fun
    print(f"\nBest: yaw={best_yaw:5.1f}° pitch={best_pitch:4.1f}° "
          f"roll={best_roll:4.1f}° depth={best_depth:4.1f} mm")
    print(f"Best F: {best_Q:.3f}")