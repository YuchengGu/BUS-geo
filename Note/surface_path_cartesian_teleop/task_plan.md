# 任务计划：曲面路径约束的 GELLO/UR5 末端遥操作

## 目标

把现有“GELLO 关节到 UR5 关节”的示教模式扩展为“GELLO 末端输入到 UR5 TCP 曲面路径跟随”的采集模式。

目标使用场景：

```text
乳腺点云
  -> 生成蛇形 coverage 路径
  -> 只保留蛇形路径拐点
  -> 每两个拐点之间生成一段贴合曲面的曲线
  -> 拼接成全局曲面路径 Gamma(S)
  -> GELLO 末端增量控制路径进度和小范围微调
  -> UR5 TCP 通过 Cartesian servo 跟随曲面路径
  -> 作为一个完整 episode 保存采集数据
```

核心交互语义：

```text
GELLO 末端沿“前进方向”的增量 -> 全局路径进度 Delta S
GELLO 末端横向增量 -> 曲面内横向微调 y_offset
GELLO 末端竖向增量 -> 表面法向微调 h_offset
GELLO 手腕旋转增量 -> 工具姿态微调 R_micro
```

UR5 不再逐关节跟随 GELLO。UR5 接收的是根据曲面路径和 Darboux frame 计算出的 TCP 目标位姿。

## 当前结论

第一版推荐方案：

```text
拐点之间使用测地线 / kNN graph shortest path 生成曲面中心线
  -> 弧长重采样和平滑
  -> 用曲线切向 t、表面法向 n、横向 b = t x n 构造 Darboux TNB frame
  -> GELLO 作为增量式末端输入设备
  -> UR5 使用 RTDE servoL 跟踪 TCP pose
```

不建议第一版直接使用论文 DOF 全空间扩散场。DOF 的思想有参考价值，但当前任务是给定 coverage 路径后的曲线跟随，Darboux frame 更直接。

## 阶段

### Phase 1: 路径表示和拐点抽取

状态：pending

任务：

- 从现有 `planned_path.json` 读取蛇形路径点和法向。
- 根据路径顺序检测拐点，或从路径规划器直接输出 row/turn metadata。
- 保存曲面段端点列表：

```text
P0, P1, P2, ..., PN
```

- 明确每个拐点在点云中的最近索引，方便后续 graph search。

验收：

- 能从已有规划路径中得到 `turnpoints.json`。
- 可视化显示拐点和原始蛇形路径一致。

### Phase 2: 拐点之间生成曲面曲线

状态：pending

任务：

- 基于 segmented point cloud 建 kNN graph。
- 每条边代价使用局部距离，可加入法向变化惩罚：

```text
cost(i,j) = ||x_i - x_j|| + lambda_normal * (1 - n_i dot n_j)
```

- 对每段 `Pk -> P{k+1}` 使用 Dijkstra/A* 生成点云上的曲面路径。
- 拼接所有段为全局曲线 `Gamma(S)`。
- 对拼接曲线做弧长重采样和平滑，避免拐点处位置不连续或速度突变。

验收：

- 生成 `surface_track.json`，包含全局路径点、弧长、切向、法向、段索引。
- 可视化曲线贴合 segmented point cloud。

### Phase 3: Darboux frame 和目标 TCP 位姿

状态：pending

任务：

- 对全局路径点计算：

```text
t(S) = dGamma/dS / ||dGamma/dS||
n(S) = interpolated surface normal
b(S) = normalize(t(S) x n(S))
D(S) = [t(S), n(S), b(S)]   # Darboux TNB frame
R_tcp_base(S) = [t(S), b(S), -n(S)]  # UR5 TCP frame
```

- 定义探头尖端和 UR5 TCP 目标：

```text
p_tip_des(S) = Gamma(S) + y_offset * b(S) + h_offset * n(S)
p_tcp_des(S) = p_tip_des(S) + probe_length * n(S)
R_des(S) = R_tcp_base(S) @ R_micro
```

- 处理法向正负号和工具 z 轴对齐关系。
- 对姿态做连续性处理，避免 frame 翻转。

验收：

- 离线可视化每个路径点的 frame。
- 相邻 frame 的角度变化平滑，无 180 度翻转。

### Phase 4: GELLO 末端输入模式

状态：pending

任务：

- 从 GELLO 关节角计算 GELLO TCP 位姿。
- 不再把 GELLO joint 直接映射到 UR5 joint。
- 运行开始时建立操作方向：

```text
方式 A: 使用 GELLO 初始 TCP x 轴作为 forward
方式 B: 用户短推 2-3 cm，系统用实际位移定义 forward
```

- 使用增量控制，避免 GELLO 物理行程不够：

```text
Delta g = g_t - g_{t-1}
Delta S = k_s * dot(Delta g, e_forward)
Delta y = -k_y * dot(Delta g, e_side)
Delta h = k_h * dot(Delta g, e_up)
```

- 增加 clutch/recenter 机制：

```text
按住 clutch 时，GELLO 可以拉回中立位，不改变 UR5 目标
松开后继续增量控制
```

验收：

- GELLO 可像鼠标一样多次小幅前推，累计推进完整路径。
- clutch 时移动 GELLO 不改变路径进度。

### Phase 5: UR5 Cartesian servo 执行

状态：pending

任务：

- 在 `gello/robots/ur.py` 增加 TCP pose command 接口，例如：

```python
command_tcp_pose(tcp_pose_rotvec)
```

- 内部调用 RTDE `servoL([x,y,z,rx,ry,rz], ...)`。
- 新增独立 Cartesian control loop，不复用 `agent.act -> env.step(joints)` 语义。
- 限制相邻目标位姿变化、速度、法向压入量和姿态微调角。

验收：

- mock/sim 模式下可以打印或记录连续 TCP target。
- 实机前可离线检查 target pose 连续性和工作空间范围。

### Phase 6: GUI / 采集流程集成

状态：pending

任务：

- 和 `visual_guided_collection_gui` 的路径确认流程对接。
- 在确认路径后进入“曲面路径遥操作采集”模式。
- 保存 episode 时记录：

```text
global_path_progress_S
current_segment_index
y_offset
h_offset
R_micro
target_tcp_pose_base
actual_tcp_pose_base
path_tracking_error
clutch_state
```

验收：

- 一个完整 episode 内自动经过多个拐点和多段曲线。
- 保存数据能复现每帧目标位姿和实际 TCP 偏差。

## 关键设计约束

- 主语是末端 TCP，不是关节。
- GELLO 是输入设备，不是 UR5 关节镜像。
- UR5 最终执行 TCP pose target，推荐通过 RTDE `servoL`。
- 采集任务是一个全局路径 `Gamma(S)`，不是每段单独任务。
- 微调来自 GELLO 偏离 forward 虚拟轨道的 residual。
- GELLO 物理行程不足用增量控制和 clutch 解决。
- `gello_get_offset.py` 不再用于 UR5-GELLO 关节对齐；最多用于检查 GELLO 自身正运动学的 joint signs/零位。

## 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
 拐点处路径/姿态不连续 | UR5 TCP 抖动或工具突然转向 | 曲线圆角/样条平滑，frame SLERP 或并行运输 |
 点云图最短路锯齿 | TCP 轨迹不平滑 | 弧长重采样、路径平滑、法向滤波 |
 GELLO forward 方向不符合直觉 | 操作困难 | 启动时用用户短推标定 forward |
 GELLO 行程不够 | 不能走完整路径 | 增量式输入 + clutch/recenter |
 servoL target 跳变 | 实机风险 | 速度/加速度/姿态变化限制，先 mock/sim 验证 |
 法向正负号错误 | 工具朝向反了或压入方向错 | 实物确认探头沿 UR5 TCP z 轴延伸方向和 `probe_length` |

## 暂不做

- 第一版不实现论文完整 DOF workspace diffusion。
- 第一版不使用主曲率线作为硬路径生成器。
- 第一版不让 UR5 逐关节跟随 GELLO。
- 第一版不在没有安全限幅和 mock 验证前直接接实机。

## 错误记录

当前无。
