# 设计草案：基于曲面路径约束的末端笛卡尔遥操作

## 背景

当前项目已有乳腺表面路径规划能力，能输出一条蛇形 coverage 路径。用户希望不要直接逐点硬跟踪，也不要让 GELLO 逐关节映射到 UR5，而是：

```text
GELLO 末端沿直线方向操作
UR5 TCP 自动沿乳腺曲面上的蛇形曲线运动
操作者仍能小范围微调方向、高度和姿态
```

这要求把 GELLO 解释为末端空间输入设备，把 UR5 解释为 Cartesian TCP pose follower。

## 总体架构

```text
planned_path.json / segmented_breast.ply
  -> 拐点抽取
  -> 拐点间曲面路径生成
  -> 全局曲线 Gamma(S)
  -> Darboux frame D(S) and UR5 TCP frame R_tcp_base(S)
  -> GELLO 增量输入解码
  -> TCP target generator
  -> UR5 servoL
  -> episode 保存
```

## 路径生成

从蛇形路径中保留拐点：

```text
P0, P1, P2, ..., PN
```

对每段 `Pk -> P{k+1}`，在 segmented point cloud 上建立 kNN graph，然后用 Dijkstra/A* 求最短曲面路径。

边代价初版：

```text
cost(i,j) = ||x_i - x_j||
```

增强版：

```text
cost(i,j) = ||x_i - x_j|| + lambda_normal * (1 - n_i dot n_j)
```

拼接所有段：

```text
Gamma(S), S in [0, L_total]
```

其中 `S` 是全局弧长进度。

## 曲面 frame

每个路径采样点上构造：

```text
t(S) = dGamma/dS / ||dGamma/dS||
n(S) = surface normal
b(S) = normalize(t(S) x n(S))
D(S) = [t(S), n(S), b(S)]  # Darboux TNB frame
R_tcp_base(S) = [t(S), b(S), -n(S)]  # UR5 TCP +z points into surface
```

这相当于沿曲线的 Darboux TNB frame。它不需要完整 DOF workspace field，因为任务只要求沿已知 coverage 路径运动。

## GELLO 输入解码

GELLO 不再输出 UR5 joint command。它输出末端增量：

```text
Delta g = g_t - g_{t-1}
```

运行开始时确定操作轴：

```text
e_forward = 用户沿期望前进方向短推得到
e_up = 用户沿真实世界 / UR base +z 方向短推得到
e_side = 由 e_forward 和 e_up 正交化得到
```

更新全局路径状态：

```text
S_t = clamp(S_{t-1} + k_s * dot(Delta g, e_forward), 0, L_total)
y_t = clamp(y_{t-1} - k_y * dot(Delta g, e_side), -y_max, y_max)
h_t = clamp(h_{t-1} + k_h * dot(Delta g, e_up), -h_max, h_max)
```

姿态微调由 GELLO wrist rotation 的增量得到：

```text
R_micro_t = clamp_rotation(R_micro_{t-1} * Delta R_gello)
```

## TCP target

探头尖端目标位置，以及反推得到的 UR5 TCP 目标位置：

```text
p_tip_des = Gamma(S_t) + y_t * b(S_t) + h_t * n(S_t)
p_tcp_des = p_tip_des + probe_length * n(S_t)
```

UR5 TCP 目标姿态：

```text
R_des = F(S_t) @ R_micro_t
```

最终发送给 UR5：

```text
tcp_pose_rotvec = [p_tcp_des_x, p_tcp_des_y, p_tcp_des_z, rx, ry, rz]
```

其中 `rx, ry, rz` 是 `R_des` 的 rotation vector。

## GELLO 行程和 clutch

GELLO 物理行程不需要覆盖整条路径。控制使用增量式输入：

```text
Delta S = k_s * dot(Delta g, e_forward)
```

需要 clutch/recenter：

```text
clutch pressed:
  只更新 GELLO reference
  不更新 S/y/h/R_micro

clutch released:
  继续用 GELLO 增量控制路径状态
```

这样操作者可以反复把 GELLO 拉回舒适位置，再继续推进。

## UR5 执行

推荐新增 URRobot Cartesian 接口：

```python
def command_tcp_pose(self, tcp_pose_rotvec: np.ndarray) -> None:
    self.robot.servoL(tcp_pose_rotvec, velocity, acceleration, dt, lookahead_time, gain)
```

不要复用现有 `RobotEnv.step(joints)`，因为它的语义是 joint command。

应新增独立 control loop：

```text
read GELLO
read UR obs
update S/y/h/R_micro
compute tcp target
servoL target
save obs/control/meta
```

## 安全限制

初始建议：

```text
y_max = 0.005~0.010 m
h_max = 0.002~0.005 m
orientation_micro_max = 5~15 deg
max_tcp_step_per_cycle = 根据控制频率限制
```

还需要：

- target pose 连续性检查。
- 工作空间范围检查。
- 拐点附近曲线和 frame 平滑。
- 实机前先 mock/sim 验证。

## 数据保存

每帧建议保存：

```text
surface_path_progress_S
surface_path_total_length
surface_path_segment_index
surface_path_local_s
surface_path_y_offset
surface_path_h_offset
surface_path_clutch_state
target_tcp_pose_base
actual_tcp_pose_base
target_frame_t_base
target_frame_b_base
target_frame_n_base
tracking_error_position
tracking_error_rotation
```

这些字段可以和现有路径 residual 一起保存，方便后续训练或分析。

## 与旧 GELLO 标定的关系

旧命令：

```bash
python scripts/gello_get_offset.py \
    --start-joints -1.57 -1.57 -1.57 -1.57 1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAW77E7-if00-port0
```

旧用途是让 GELLO joint 和 UR5 joint 对齐。新模式不需要这个对齐。

新模式仍需要：

- GELLO 自身 FK 合理。
- joint signs 让 GELLO TCP 运动方向正确。
- 运行开始时做 forward direction calibration。

因此 `gello_get_offset.py` 可以作为 GELLO 自身检查工具，但不是新控制模式的必要采集步骤。

## 第一版验收标准

离线验收：

- 输入已有 `planned_path.json` 和点云，能生成全局曲面路径 `Gamma(S)`。
- 可视化显示曲线贴合表面，拐点处平滑。
- 可视化显示 Darboux frame 连续、无翻转。

控制验收：

- mock GELLO 增量输入时，`S` 能连续走完整路径。
- clutch 时移动 GELLO 不改变 `S/y/h`。
- 生成的 TCP target 连续、有限幅。

实机前验收：

- 不连接 UR5 时记录一段 target pose，检查速度和姿态变化。
- 连接 UR5 后先空中低速跟踪，不接触人体/模型。
- 再进行曲面贴合采集。
