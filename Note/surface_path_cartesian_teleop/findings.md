# 发现记录：曲面路径约束的 GELLO/UR5 末端遥操作

## 仓库现状

- `breast_path_planning/` 已经能从 D405 或 PLY 点云生成乳腺表面蛇形路径。
- `planned_path.json` 当前 schema 包含：

```json
{
  "position_base": [x, y, z],
  "normal_base": [nx, ny, nz]
}
```

- `breast_path_planning/path_planner.py` 当前生成的是 adaptive slice + serpentine 路径。
- `visual_guided_collection_gui/` 已有路径规划、确认和 GELLO 采集流程说明。
- `gello/robots/ur.py` 当前只暴露 joint-space command：

```python
command_joint_state(joint_state)
```

内部使用 RTDE `servoJ`。

- `gello/env.py` 已经在 obs 中加入 UR TCP 的 base 坐标信息：

```text
tcp_position_base
tcp_x_axis_base
tcp_y_axis_base
tcp_z_axis_base
```

## 讨论结论

### 与论文 DOF 的关系

论文 DOF 方法是：

```text
点云 + keypoints
  -> 扩散方程生成整个工作空间 orientation field
  -> 任意位置 query local frame
  -> controller 在 local frame 中动作
```

本项目更需要：

```text
已知 coverage 路径
  -> 拐点之间生成曲面曲线
  -> 沿曲线构造 Darboux TNB frame
  -> GELLO 控制路径参数和微调
  -> UR5 TCP 跟踪目标位姿
```

所以本方案不是照搬论文，而是借用“在局部曲面 frame 中表达动作”的思想，改成适配本项目的路径约束 Cartesian teleoperation。

### 主曲率线 / 测地线 / 扩散方法比较

第一版推荐测地线 / kNN graph shortest path：

- 给定两个拐点时，测地线最符合“从一个点到另一个点”的需求。
- 在点云上工程实现简单，适合现有 `planned_path.json` 和 segmented point cloud。
- 结果是一条明确曲线，可直接用 GELLO forward 输入映射到弧长进度。

主曲率线：

- 可作为方向偏好或路径平滑约束。
- 不保证连接给定拐点。
- 对点云二阶曲率估计敏感，在近球面/平坦区/脐点附近不稳定。

扩散方程：

- 可生成平滑方向场，抗噪，适合后续增强。
- 对单段拐点连接来说实现较重。
- 从起点沿梯度积分不一定精确到达 sink，需要吸附/纠偏。

## 控制语义

GELLO 不再输出 UR5 joint command，而是提供末端输入：

```text
Delta gello forward -> Delta S
Delta gello side -> Delta y_offset
Delta gello up/down -> Delta h_offset
Delta gello wrist rotation -> Delta R_micro
```

UR5 接收：

```text
T_base_tcp_des = (p_tcp_des, R_des)
```

其中：

```text
p_tip_des = Gamma(S) + y_offset * b(S) + h_offset * n(S)
p_tcp_des = p_tip_des + probe_length * n(S)
R_des = R_tcp_base(S) @ R_micro, where R_tcp_base(S) = [t(S), b(S), -n(S)]
```

## GELLO 行程不足的处理

不能用绝对位移完成整条路径。应该使用增量控制：

```text
S_t = S_{t-1} + k_s * dot(g_t - g_{t-1}, e_forward)
```

配合 clutch/recenter：

```text
clutch pressed:
  更新 GELLO reference，不改变 UR5 target
clutch released:
  继续使用增量驱动路径进度
```

这样 GELLO 可以像鼠标一样反复前推，累计完成完整蛇形路径。

## GELLO x / forward 方向

不要强依赖 URDF 中的 TCP x 轴。更稳的方式是启动时用户短推标定：

```text
1. 记录 GELLO TCP 初始位置 g0
2. 用户按自己认为的 forward 推 2-3 cm
3. 记录 g1
4. e_forward = normalize(g1 - g0)
```

其他操作轴可由初始 TCP 姿态或与 `e_forward` 正交化得到。

## `gello_get_offset.py` 的角色变化

旧模式中：

```text
gello_get_offset.py 用于 GELLO joint 和 UR5 joint 对齐
```

新模式中不再需要 UR5-GELLO 关节对齐。仍可能需要：

- 确认 GELLO 自身 joint signs 正确。
- 确认 GELLO 正运动学能给出合理 TCP 位姿。
- 但不需要让 GELLO 姿态和 UR5 初始姿态一致。

## 需要后续确认

- GELLO 当前是否已有 FK / URDF 可直接得到 TCP 位姿。
- GELLO 是否有可用按钮作为 clutch；没有则先用键盘。
- UR5 当前 RTDE 版本是否稳定支持 `servoL`。
- 超声探头沿 UR5 TCP z 轴延伸的长度 `probe_length`。
- `planned_path.json` 是否需要扩展 row/turn metadata，避免从点序列中猜拐点。
