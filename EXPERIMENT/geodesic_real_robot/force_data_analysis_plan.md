# 测地线路径优化实机数据分析方案

## 1. 分析目的

本实验比较同一条初始蛇形路径经过四种处理后，在真实机械臂扫查中的接触稳定性、路径跟踪性能、运动平滑性和力控介入程度：

1. `original`：原始规划路径；
2. `moving_average`：移动平均平滑；
3. `b_spline`：B-spline 平滑；
4. `geodesic`：基于测地曲率能量和模拟退火的路径优化。

主要研究问题不是“哪种方法的平均力最小”，而是：

- 是否减少过大接触压力及其持续时间；
- 是否降低切向力和探头卡滞风险；
- 是否降低力矩和接触偏心；
- 是否减少 force servo 和 hard-lift 的介入；
- 是否提高经过拐点时的接触与运动平稳性；
- 是否在不明显损失路径覆盖和跟踪精度的情况下实现上述改善。

## 2. 已有数据和实验分组

2026-06-30 共记录 3 条路径，每条路径包含 4 种方法，共 12 个 episode。

| 路径组 | 耦合条件 | 方法 | Episode |
|---|---|---|---|
| Path 1 | 未涂超声耦合剂 | Original | `auto_scan_0630_174040` |
| Path 1 | 未涂超声耦合剂 | Moving average | `auto_scan_0630_174314` |
| Path 1 | 未涂超声耦合剂 | B-spline | `auto_scan_0630_174556` |
| Path 1 | 未涂超声耦合剂 | Geodesic | `auto_scan_0630_174853` |
| Path 2 | 已涂超声耦合剂 | Original | `auto_scan_0630_180145` |
| Path 2 | 已涂超声耦合剂 | Geodesic | `auto_scan_0630_180536` |
| Path 2 | 已涂超声耦合剂 | Moving average | `auto_scan_0630_180839` |
| Path 2 | 已涂超声耦合剂 | B-spline | `auto_scan_0630_181204` |
| Path 3 | 已涂超声耦合剂 | Geodesic | `auto_scan_0630_182540` |
| Path 3 | 已涂超声耦合剂 | Original | `auto_scan_0630_182849` |
| Path 3 | 已涂超声耦合剂 | Moving average | `auto_scan_0630_183142` |
| Path 3 | 已涂超声耦合剂 | B-spline | `auto_scan_0630_183431` |

每帧通过以下字段确认方法，不能仅根据运行顺序推断：

```python
sample["meta"]["path_variant_method"]
```

当前这些 PKL 保存了 RGB、深度、机器人状态、路径信息、力/力矩及力控状态，但没有超声图像。因此本方案只能分析接触力学和机器人运动，不能从这些 episode 计算超声图像质量 \(Q\)。

## 3. 坐标、符号和原始字段

补偿后的六维力/力矩为：

\[
\mathbf{w}
=
[F_x,F_y,F_z,\tau_x,\tau_y,\tau_z]^T,
\]

其中力的单位为 \(\mathrm{N}\)，力矩的单位为 \(\mathrm{N\,m}\)。本项目约定压力为负的 \(F_z\)，所以定义接触压力大小：

\[
p=-F_z.
\]

主要 PKL 字段：

| 字段 | 含义 |
|---|---|
| `force` | 重力和零偏补偿后的六维力/力矩 |
| `force_raw` | 传感器原始六维读数 |
| `force_gravity` | 当前姿态下估计的重力项 |
| `force_bias` | 悬空归零得到的零偏项 |
| `force_gravity_calibrated` | 是否加载重力补偿 |
| `force_zeroed` | 是否完成归零 |
| `force_pressure_n` | 已计算的 \(p=-F_z\) |
| `force_tangential_norm_n` | 已计算的 \(\sqrt{F_x^2+F_y^2}\) |
| `force_shear_ratio` | 已计算的切向力/法向力比值 |
| `torque_axial_z_nm` | 轴向力矩 \(\tau_z\) |
| `torque_tangential_norm_nm` | 横向力矩 \(\sqrt{\tau_x^2+\tau_y^2}\) |
| `tcp_position_base` | 实际 TCP 位置 |
| `ee_pos_rotvec` | 实际 TCP 位姿，后三维为旋转向量 |
| `probe_tip_position_base` | 根据探头长度计算的探头尖端位置 |
| `control` | 当前发送给机器人端的控制量 |
| `path_progress` | 实际探头相对参考路径的最近点进度 |
| `path_distance_to_nearest_m` | 探头尖端到最近参考路径点的距离 |
| `path_residuals_base` | 探头尖端相对参考路径点的残差 |
| `meta` | 时间、路径方法、自动扫查和力控状态 |

传感器补偿和机器人运动补偿必须分开：

\[
\mathbf{w}_{\mathrm{comp}}
=
\mathbf{w}_{\mathrm{raw}}
-\mathbf{w}_{\mathrm{gravity}}
-\mathbf{w}_{\mathrm{bias}},
\]

而 force servo 的运动补偿是沿曲面法向施加位置偏移：

\[
\mathbf{p}_{\mathrm{cmd}}
=
\mathbf{p}_{\mathrm{ref}}
+d_n\mathbf{n}.
\]

## 4. 数据有效性和预处理

### 4.1 有效帧

只将满足以下条件的帧纳入主要力学统计：

- `meta["modalities"]["force"]["valid"] == True`；
- `force_gravity_calibrated == True`；
- `force_zeroed == True`；
- 六维力/力矩均为有限数；
- 时间戳严格递增。

同时报告无效帧比例，不能直接静默删除。

### 4.2 时间轴

优先使用：

```python
sample["meta"]["sample_mono_ns"]
```

转换为相对时间：

\[
t_k=\frac{T_k-T_0}{10^9}.
\]

所有积分必须使用真实的相邻时间差：

\[
\Delta t_k=t_k-t_{k-1},
\]

不能默认控制周期严格恒定。

### 4.3 进度轴

同时保留两种进度：

1. 实际路径进度：`path_progress`；
2. 指令路径进度：由 `auto_scan_pose_index`、`auto_scan_waypoint_index` 和对应 count 计算。

实际进度适合表示探头真正移动到了哪里；指令进度适合对齐四种路径方法。二者不能混为一个量。

### 4.4 滤波原则

- 所有安全阈值、峰值和 hard-lift 事件使用未额外滤波的补偿后力；
- 趋势图可叠加低通或 Savitzky-Golay 平滑曲线；
- 计算速度、加速度和 jerk 前需要按统一时间网格重采样并平滑；
- 必须同时保存原始曲线，避免滤波掩盖危险峰值。

### 4.5 分析区间

每个 episode 分别分析：

1. 完整扫查区间；
2. 去除起始和结束各 5% 的稳定区间；
3. 直线扫查段；
4. 拐点附近窗口；
5. hard-lift 事件窗口；
6. 高切向力或高力矩事件窗口。

## 5. 法向接触压力指标

定义：

\[
p_k=-F_{z,k}.
\]

### 5.1 基本统计

- mean、median；
- standard deviation；
- interquartile range；
- RMS；
- P90、P95、P99；
- minimum、maximum；
- coefficient of variation，仅在平均压力明显大于零时使用。

### 5.2 目标压力区间

当前目标区间为 \(3\text{--}4\,\mathrm{N}\)。计算：

\[
R_{\mathrm{target}}
=
\frac{\sum_k \Delta t_k
\mathbf{1}(3\le p_k\le4)}
{\sum_k\Delta t_k}.
\]

同时计算：

- \(p<3\,\mathrm{N}\) 的欠压时间比例；
- \(p>4\,\mathrm{N}\) 的超压时间比例；
- \(p>8\,\mathrm{N}\) 的严重超压时间比例。

### 5.3 超压暴露量

\[
I_{4}
=
\int \max(p(t)-4,0)\,dt,
\]

\[
I_{8}
=
\int \max(p(t)-8,0)\,dt.
\]

单位均为 \(\mathrm{N\,s}\)。其中 \(I_4\) 推荐作为主要实机指标，因为它同时反映超压幅度和持续时间。

### 5.4 压力平稳性

\[
TV_p=\sum_k|p_k-p_{k-1}|,
\]

\[
\mathrm{RMS}(\dot p)
=
\sqrt{
\frac{1}{N-1}
\sum_k
\left(
\frac{p_k-p_{k-1}}{\Delta t_k}
\right)^2
}.
\]

还应报告：

- 相邻帧压力跳变的 P95 和最大值；
- 压力一阶差分的 robust RMS；
- 重采样后高频频带功率；
- 目标压力区间附近的振荡次数；
- 从欠压进入超压、再返回目标区间的次数。

## 6. 切向力指标

切向合力定义为：

\[
F_t=\sqrt{F_x^2+F_y^2}.
\]

必须分析：

- \(F_x\)、\(F_y\) 各自的均值、标准差、P95 和最大绝对值；
- \(F_t\) 的 mean、median、RMS、P95、P99 和 maximum；
- \(F_t>8\,\mathrm{N}\) 的持续时间和比例；
- 横向过载积分：

\[
I_t
=
\int\max(F_t(t)-8,0)\,dt;
\]

- 切向冲量：

\[
J_t=\int F_t(t)\,dt;
\]

- 切向力总变差和 \(\mathrm{RMS}(\dot F_t)\)；
- 拐点前后切向力峰值；
- hard-lift 中由 `lateral` 原因触发的事件数。

切向力是判断探头边缘卡滞、横向拖拽和路径拐角不平顺的重要指标，不能只分析 \(F_z\)。

## 7. 法向与切向耦合指标

### 7.1 剪切比

\[
\rho_F
=
\frac{F_t}{p+\varepsilon}.
\]

仅在 \(p>1\,\mathrm{N}\) 时统计，避免接近悬空时分母过小。报告 median、P95 和 maximum。

### 7.2 合力

\[
\|\mathbf{F}\|
=
\sqrt{F_x^2+F_y^2+F_z^2}.
\]

报告合力峰值和高合力持续时间。

### 7.3 力方向稳定性

在 \(p>1\,\mathrm{N}\) 时，计算接触合力与探头轴方向的夹角：

\[
\theta_F
=
\arctan2(F_t,p).
\]

较大的 \(\theta_F\) 表示受力偏离理想法向接触。报告 mean、P95 和 maximum。

## 8. 力矩指标

### 8.1 分量和合量

横向弯矩：

\[
\tau_t
=
\sqrt{\tau_x^2+\tau_y^2}.
\]

轴向扭矩：

\[
\tau_a=|\tau_z|.
\]

总力矩：

\[
\|\boldsymbol{\tau}\|
=
\sqrt{\tau_x^2+\tau_y^2+\tau_z^2}.
\]

分别报告：

- \(\tau_x,\tau_y,\tau_z\) 的均值、标准差、P95 和最大绝对值；
- \(\tau_t\)、\(\tau_a\) 和总力矩的 mean、RMS、P95、P99、maximum；
- 力矩总变差；
- \(\mathrm{RMS}(\dot{\tau}_t)\)；
- 拐点附近力矩峰值；
- hard-lift 前后的力矩变化。

### 8.2 有效接触偏心距离

在接触力主要沿探头 \(z\) 轴，且传感器坐标和探头几何关系正确时，可定义近似的接触偏心：

\[
r_{\mathrm{cop}}
\approx
\frac{\sqrt{\tau_x^2+\tau_y^2}}
{|F_z|}.
\]

单位为米。仅在 \(|F_z|>1\,\mathrm{N}\) 时计算。它不是经过严格标定的真实压力中心，但可作为探头受力偏心和翘起趋势的代理指标。

需要报告：

- \(r_{\mathrm{cop}}\) 的 median、P95 和 maximum；
- \(r_{\mathrm{cop}}\) 超过探头合理半径的帧比例；
- 拐点附近 \(r_{\mathrm{cop}}\) 是否显著增大。

### 8.3 不推荐直接使用的力矩比值

以下比值在 \(\tau_z\) 接近零时会发散：

\[
\frac{|\tau_x|+|\tau_y|}{|\tau_z|}.
\]

如果用于 BO 目标函数必须加 \(\varepsilon\) 和裁剪；实机路径比较优先使用 \(\tau_t\)、\(\tau_a\) 和 \(r_{\mathrm{cop}}\)。

## 9. 力控和 hard-lift“救火”指标

主要字段：

```text
auto_force_servo_filtered_pressure_n
auto_force_servo_offset_m
auto_force_servo_command_offset_m
auto_force_servo_delta_offset_m
auto_force_servo_velocity_m_s
auto_force_servo_acceleration_m_s2
auto_force_servo_direction
auto_force_servo_hard_lift_active
auto_force_servo_hard_lift_entry
auto_force_servo_hard_lift_reason
auto_force_servo_hard_lift_limit_reached
auto_force_servo_inward_motion_blocked
```

### 9.1 普通 force servo 介入

- 法向 offset 的 mean、RMS、P95、maximum 和 minimum；
- 最大向外抬升距离；
- 最大向内下压距离；
- offset 总变差；
- offset 变化率 RMS；
- 向上、向下和保持状态的时间占比；
- `inward_motion_blocked` 的次数和持续时间；
- offset 接近普通 \(\pm6\,\mathrm{mm}\) 限制的时间比例。

### 9.2 hard-lift 事件

事件定义为：

```text
hard_lift_active: False -> True
```

对每个事件统计：

- 触发时间和路径进度；
- 触发原因：`pressure`、`lateral` 或二者；
- 进入时 \(p\)、\(F_t\)、\(\tau_t\)、offset；
- hard-lift 持续时间；
- 最大压力、最大切向力和最大力矩；
- 最大额外抬升距离；
- 返回 \(p<4.5\,\mathrm{N}\) 且 \(F_t<4.5\,\mathrm{N}\) 的恢复时间；
- 恢复后的压力反弹量；
- 是否达到 \(80\,\mathrm{mm}\) hard-lift 上限。

### 9.3 救火负担

推荐综合报告以下可解释指标，而不是先合成为一个任意评分：

- hard-lift 事件数；
- hard-lift 总持续时间；
- hard-lift 时间比例；
- 最大 outward offset；
- outward offset 时间积分：

\[
A_d
=
\int \max(d_n(t),0)\,dt;
\]

- 每米路径对应的 hard-lift 次数；
- 每分钟对应的 hard-lift 次数；
- 单次事件的中位恢复时间。

测地线方法若具有更小的 \(I_4\)、hard-lift 时间和 \(A_d\)，说明其自身路径更容易维持安全接触，需要控制器“救火”的程度更低。

## 10. 路径跟踪指标

### 10.1 TCP 指令跟踪

将当前实际 TCP 与上一帧实际发出的 `control` 对齐，计算：

\[
e_{\mathrm{TCP},k}
=
\mathbf{p}^{\mathrm{actual}}_k
-\mathbf{p}^{\mathrm{command}}_{k-1}.
\]

使用上一帧命令是因为当前 PKL 保存的是发出本帧命令前的观测。

报告：

- 位置误差 mean、RMS、P95、maximum；
- 姿态误差 mean、P95、maximum；
- 超过指定位置或角度误差阈值的时间比例。

### 10.2 名义路径与力控偏移

探头到曲面参考路径的总距离包含主动法向 force-servo offset，因此必须分解：

\[
\mathbf e
=
e_t\mathbf t+e_b\mathbf b+e_n\mathbf n.
\]

分别统计：

- 切平面内误差：

\[
e_{\mathrm{tan}}=\sqrt{e_t^2+e_b^2};
\]

- 法向误差 \(e_n\)；
- 扣除已知 force-servo offset 后的法向残差；
- 路径最近点索引是否单调前进；
- 实际路径进度停滞或倒退次数。

其中 \(e_{\mathrm{tan}}\) 更接近纯路径跟踪性能；法向距离不能未经分解直接解释为路径规划误差。

## 11. 机器人运动平滑性

从实际 TCP 位置和姿态计算：

- 平移速度、加速度、jerk；
- 角速度、角加速度；
- 单帧最大位置增量；
- 单帧最大姿态增量；
- 扫查总时长；
- 单位路径长度对应的扫查时间；
- 在拐点附近的速度下降和姿态变化。

平移 jerk：

\[
j(t)=\frac{d^3\mathbf p(t)}{dt^3}.
\]

建议报告：

\[
\mathrm{RMS}(a),\qquad
\mathrm{RMS}(j),\qquad
\max\|a\|,\qquad
\max\|j\|.
\]

由于数值微分会放大噪声，必须先按照统一时间网格重采样并进行固定参数平滑，四种方法使用完全相同的处理参数。

还可从关节数据分析：

- 关节速度 RMS 和峰值；
- 关节加速度 RMS；
- 关节运动总量；
- 是否接近关节极限。

## 12. 拐点专项分析

根据每个 `auto_scan_pose_index` 对应的目标位置重建离散路径。对相邻路径方向：

\[
\mathbf d_i
=
\frac{\mathbf p_{i+1}-\mathbf p_i}
{\|\mathbf p_{i+1}-\mathbf p_i\|},
\]

计算转角：

\[
\theta_i
=
\arccos
\left(
\operatorname{clip}
(\mathbf d_{i-1}^T\mathbf d_i,-1,1)
\right).
\]

将较大转角位置定义为拐点，并在其前后固定弧长或固定时间窗口内统计：

- \(p\) 峰值；
- \(F_t\) 峰值；
- \(\tau_t\) 峰值；
- \(r_{\mathrm{cop}}\) 峰值；
- force-servo offset 变化量；
- hard-lift 是否发生；
- TCP 加速度和 jerk；
- 姿态角速度峰值。

比较“拐点窗口”和“直线段窗口”，可判断测地线优化收益是否主要来自减小转折区域的剧烈变化。

## 13. 事件对齐分析

### 13.1 hard-lift 对齐

将每次 hard-lift 进入时刻设为 \(t=0\)，绘制 \([-1,3]\,\mathrm{s}\)：

- \(p(t)\)；
- \(F_t(t)\)；
- \(\tau_t(t)\)；
- \(d_n(t)\)；
- force-servo velocity。

按方法计算事件曲线的 median 和 IQR，观察触发前恶化速度及触发后恢复速度。

### 13.2 拐点对齐

将每个拐点通过时刻设为 \(t=0\)，绘制：

- 压力；
- 切向力；
- 横向力矩；
- TCP 角速度；
- 法向 offset。

这比只比较整条路径平均值更容易显示路径平滑方法的局部作用。

## 14. 重力补偿和零偏质量

必须检查：

- `force_gravity_calibrated` 有效率；
- `force_zeroed` 有效率；
- `force_raw - force_gravity - force_bias` 是否与 `force` 一致；
- 扫查开始和结束附近的补偿后力是否存在明显漂移；
- 不同姿态下补偿残差是否随 TCP 姿态系统性变化；
- 原始力矩和补偿后力矩的差异。

如果补偿后力仍与姿态明显相关，应将姿态相关残差作为局限性报告，不能把所有变化归因于路径方法。

## 15. 分组比较原则

### 15.1 主要分析

Path 2 和 Path 3 均使用耦合剂，作为主要结果。对每条路径内部比较四种方法。

### 15.2 敏感性分析

Path 1 未使用耦合剂，单独展示，并作为不同接触条件下的敏感性分析。

### 15.3 归一化配对比较

为了减小不同乳腺区域和路径难度的影响，对每条路径使用 Original 作为基准：

\[
R_{m,j}
=
\frac{M_{m,j}}
{M_{\mathrm{original},j}},
\]

其中 \(m\) 为方法，\(j\) 为路径。

对于“越小越好”的指标，\(R<1\) 表示优于 Original。建议主要展示：

- \(I_4\) ratio；
- \(I_8\) ratio；
- hard-lift duration ratio；
- \(F_t\) P95 ratio；
- \(\tau_t\) P95 ratio；
- pressure variation ratio；
- jerk RMS ratio。

### 15.4 统计限制

- 只有 3 条独立路径，且有耦合剂的路径只有 2 条；
- 数万帧不是数万个独立样本，不能把帧当作独立重复进行显著性检验；
- 路径方法的运行顺序并未完全随机；
- 无耦合剂和有耦合剂是明显混杂因素。

因此当前结果应以配对原始值、效应比例、时间曲线和事件分析为主，不宜依赖帧级 \(p\)-value。

## 16. 推荐的论文图

### 图 1：典型有耦合剂路径的完整扫查过程

共享归一化路径进度横轴，四行或四个紧凑子图：

1. \(p=-F_z\)，标出 \(3\text{--}4\,\mathrm{N}\) 目标带和 \(8\,\mathrm{N}\) 阈值；
2. \(F_t\)，标出 \(8\,\mathrm{N}\) 阈值；
3. \(\tau_t\) 和 \(r_{\mathrm{cop}}\)；
4. force-servo offset，并用背景色标记 hard-lift。

四种方法使用固定颜色，Geodesic 使用绿色。

### 图 2：三条路径的配对方法比较

每条路径用细线连接四种方法，展示：

- 超压积分 \(I_4\)；
- hard-lift 时间比例；
- \(F_t\) P95；
- \(\tau_t\) P95；
- pressure variation；
- jerk RMS。

无耦合剂路径使用空心标记，有耦合剂路径使用实心标记。

### 图 3：拐点事件对齐

显示四种方法经过拐点前后的：

- 压力；
- 切向力；
- 横向力矩；
- 法向 offset。

### 图 4：hard-lift 救火过程

以 hard-lift 触发为零时刻，显示压力下降和 offset 上升过程，并报告恢复时间。

### 补充图

- 原始力、重力项、bias 和补偿后力；
- 力/力矩有效率；
- 未涂耦合剂路径的完整结果；
- TCP 跟踪误差；
- 关节运动和平滑性；
- 路径进度与扫描时间。

## 17. 推荐的结果表

每个 episode 一行，至少包含：

```text
path_group
coupling_condition
scan_order
path_variant_method
frame_count
duration_s
force_valid_ratio
pressure_mean_n
pressure_std_n
pressure_p95_n
pressure_max_n
pressure_target_band_ratio
pressure_over_4_ratio
pressure_over_8_ratio
pressure_exposure_4_ns
pressure_exposure_8_ns
tangential_force_mean_n
tangential_force_p95_n
tangential_force_max_n
tangential_exposure_8_ns
tangential_impulse_ns
torque_tangential_p95_nm
torque_tangential_max_nm
torque_axial_p95_nm
torque_total_p95_nm
cop_proxy_p95_m
hard_lift_event_count
hard_lift_duration_s
hard_lift_ratio
hard_lift_pressure_event_count
hard_lift_lateral_event_count
hard_lift_recovery_time_median_s
hard_lift_limit_reached
force_offset_rms_mm
force_offset_max_outward_mm
force_offset_max_inward_mm
force_offset_integral_mm_s
tcp_tracking_error_p95_mm
tangential_path_error_p95_mm
normal_path_residual_p95_mm
tcp_acceleration_rms_m_s2
tcp_jerk_rms_m_s3
angular_velocity_p95_rad_s
angular_acceleration_rms_rad_s2
```

## 18. 推荐的主次指标

为避免结果出来后挑选指标，建议固定以下层级。

主要指标：

\[
\boxed{I_4=\int\max(p-4,0)\,dt}
\]

关键次要指标：

1. hard-lift 时间比例；
2. \(F_t\) P95；
3. \(\tau_t\) P95；
4. 压力标准差和总变差；
5. 拐点附近压力/切向力峰值；
6. force-servo outward offset 积分。

支持性指标：

- 目标压力区间占比；
- \(r_{\mathrm{cop}}\)；
- TCP jerk；
- 路径跟踪误差；
- 扫查时间；
- 重力补偿有效性。

## 19. 结果解释策略

如果 Geodesic 的 \(I_4\)、hard-lift 时间和切向力更低，可表述为：

> Geodesic path optimization reduced excessive contact loading and required less corrective normal motion from the force controller.

如果平均压力相近但峰值和波动下降，可表述为：

> Geodesic optimization preserved the desired mean contact level while reducing transient force peaks and contact fluctuations.

如果 force servo 最终将四种方法都控制到相近压力，但 Geodesic 所需 offset 更小，可表述为：

> Although closed-loop force regulation produced comparable steady-state pressure, the geodesic path required less compensatory displacement, indicating better agreement between the planned surface path and physical contact.

如果力差异不明显，但拐点 jerk、切向力或力矩下降，可表述为：

> The principal benefit was concentrated around path reversals, where geodesic optimization reduced tangential loading, torque and motion discontinuity.

如果三条路径结论不一致，必须展示全部结果，并将其表述为探索性实机验证，而不是进行选择性报告。

## 20. 后续分析脚本结构

建议在当前目录继续添加：

```text
trial_manifest.json
extract_force_trial_cache.py
analyze_force_trials.py
plot_force_trial_results.py
tests/test_force_trial_analysis.py
cache/
results/
```

第一次运行逐帧读取 PKL，只提取标量和小型向量到每个 episode 的压缩 NPZ 缓存。后续计算指标和修改图形时只读取缓存，避免反复读取每帧的 RGB 和深度数组。

## 21. 当前五个绘图入口

在仓库根目录运行。将 `--group 2` 改成 `1` 或 `3` 即可选择另外一组。

### 21.1 六维力学量随扫查进度变化

```bash
python -m EXPERIMENT.geodesic_real_robot.plot_case_wrench_progress --group 2
```

输出法向压力、切向合力、横向合力矩和轴向力矩随归一化扫查进度的四方法对比。

### 21.2 超压与压力平稳性

```bash
python -m EXPERIMENT.geodesic_real_robot.plot_pressure_metrics --group 2
```

输出单位时间超压负担、严重超压比例、目标压力区间比例、压力标准差、压力总变差率和压力导数 RMS。

### 21.3 切向力

```bash
python -m EXPERIMENT.geodesic_real_robot.plot_tangential_force_metrics --group 2
```

输出切向力均值、P95、峰值、变化率、导数 RMS 和剪切比 P95。超过 \(8\,\mathrm{N}\) 的切向过载指标仍在共享计算模块中保留，但当前第二组数据全部为零，因此不放在主图中。

### 21.4 力矩

```bash
python -m EXPERIMENT.geodesic_real_robot.plot_torque_metrics --group 2
```

输出横向力矩均值、P95、峰值、轴向力矩 P95、总力矩 P95 和横向力矩变化率。接触偏心代理量仍在共享计算模块中保留，但它受到探头杠杆臂和残余重力矩影响，不放在当前主图中。

### 21.5 Force-servo/hard-lift 救火程度

```bash
python -m EXPERIMENT.geodesic_real_robot.plot_rescue_metrics --group 2
```

输出 hard-lift 事件数、持续时间、时间比例、最大向外偏移、平均向外偏移负担和 offset 总变差。

所有入口同时输出矢量 PDF 和 300 dpi PNG：

```text
EXPERIMENT/geodesic_real_robot/results/group_<N>/
```

第一次分析某组数据时会建立：

```text
EXPERIMENT/geodesic_real_robot/cache/*.npz
```

后续运行直接读取缓存。原始 PKL 改变后使用 `--rebuild-cache` 强制重建。

新增第 4 组及后续数据时，只需在 `force_analysis.py` 的 `GROUPS` 中增加组号、说明和四个 episode 文件夹；五个绘图入口会自动接受新增组号。
