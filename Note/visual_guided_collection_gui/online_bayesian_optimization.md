# 在线贝叶斯优化 BO 代码与数据记录说明

本文档对应当前代码中的在线局部探头位姿优化模块。核心代码路径：

- GUI 参数入口：`visual_guided_collection_gui/main.py`
- GUI 按键与线程流程：`visual_guided_collection_gui/app.py`
- 在线 BO 目标函数、候选位姿、记录：`visual_guided_collection_gui/surface_bayes.py`
- GP/EI ask-tell 优化器：`visual_guided_collection_gui/offline_bayes.py`
- 超声图像质量评分：`visual_guided_collection_gui/ultrasound_quality.py`
- BO 路径点选择：`visual_guided_collection_gui/surface_bo_point.py`

## 1. 使用模式

BO 可以在两类模式下运行：

1. `--operation-mode bo`

   先拍照、分割、规划路径；点击 `Select path BO point` 从规划路径上随机选择一个路径点；点击 `Move to BO point` 移动到该点对应的初始 TCP 位姿；再点击 BO 条件按钮运行优化。

2. `--operation-mode auto`

   在自动扫查过程中点击 `Optimize local pose`，程序会暂停自动扫查，在当前最近路径点附近做一次局部 BO，结束后继续主流程。

当前 BO 测试更推荐使用 `--operation-mode bo`，因为它不扫完整路径，只验证某一个路径点附近的局部位姿优化。

## 2. BO 起点和参考位姿

在 `bo` 模式下，BO 起点不是随机曲面点，而是规划路径上的点：

```python
select_random_path_bo_reference(path, rng)
```

它返回：

- `path_index`：被选中的路径点编号
- `position_base`：路径点位置
- `normal_base`：该点曲面法向
- `tangent_base`：该点路径切向

点击 `Move to BO point` 后，程序会把 UR5 TCP 移动到该路径点对应的接触高度位姿：

- 预备高度：`--surface-approach-height-m`
- 接触高度：`--surface-contact-height-m`
- 探头长度：`--probe-tip-offset-m`

当前默认值：

```text
surface_approach_height_m = 0.07 m
surface_contact_height_m  = 0.02 m
probe_tip_offset_m        = 0.204 m
```

真正开始 BO 时，参考 TCP 位姿 \(x_0\) 取当前实际 TCP：

\[
\mathbf{T}_0 =
\left[
\mathbf{p}_0,\ \mathbf{r}_0
\right]
\]

其中 \(\mathbf{p}_0\) 是当前 TCP 位置，\(\mathbf{r}_0\) 是当前 TCP 旋转向量。

代码：

```python
select_current_tcp_bo_reference(obs, enriched_obs)
```

## 3. 优化变量

BO 的局部变量为：

\[
\mathbf{x}
= [d_n,\ r_x,\ r_y,\ r_z]^T
\]

含义：

- \(d_n\)：沿曲面外法向的 TCP 平移，单位 m
- \(r_x,r_y,r_z\)：在当前 TCP 局部姿态上的右乘微旋转，单位 rad

当前 GUI 默认搜索范围在 `main.py`：

```text
dn = [-0.011, 0.011] m
rx = [-0.034, 0.034] rad
ry = [-0.034, 0.034] rad
rz = [-0.034, 0.034] rad
```

对应约为：

```text
dn = +/- 11 mm
rx, ry, rz = +/- 1.95 deg
```

参数：

```bash
--surface-bo-bounds "dn=-0.011,0.011;rx=-0.034,0.034;ry=-0.034,0.034;rz=-0.034,0.034"
```

## 4. 候选 TCP 位姿计算

候选变量 \(\mathbf{x}\) 转换成机器人 TCP 目标位姿的代码：

```python
compute_candidate_tcp_pose(reference_tcp_pose, normal_base, local_x)
```

数学上：

\[
\mathbf{p}(\mathbf{x})
=
\mathbf{p}_0
+
d_n \mathbf{n}
\]

\[
\mathbf{R}(\mathbf{x})
=
\mathbf{R}_0
\exp
\left(
\widehat{
\begin{bmatrix}
r_x & r_y & r_z
\end{bmatrix}^T
}
\right)
\]

其中：

- \(\mathbf{n}\) 是当前路径点曲面外法向，base/world 坐标表达
- \(\mathbf{R}_0\) 是 BO 开始时当前 TCP 姿态
- 旋转是右乘，因此 \(r_x,r_y,r_z\) 是 TCP 局部坐标下的微旋转

最后发送给 UR5 的目标仍然是：

\[
[x,\ y,\ z,\ r_x,\ r_y,\ r_z]
\]

即 RTDE/UR 常用的 TCP 位置 + 旋转向量格式。

## 5. 超声图像质量 \(Q\)

BO 使用在线超声帧计算图像质量。代码：

```python
compute_online_objective(...)
UltrasoundQualityScorer.score_frame(...)
```

### 5.1 超声裁剪

计算 \(Q\) 前会先裁剪超声图：

```text
row0,row1,col0,col1 = 99,769,542,1524
```

对应参数：

```bash
--surface-bo-ultrasound-crop "99,769,542,1524"
```

保存到 pkl 的超声图仍然是原始图像；BO 计算和后续 BO 画图使用裁剪后的图像。

### 5.2 四个图像特征

质量分数来自四个特征：

\[
D,\ E,\ C,\ S
\]

其中：

- \(D\)：confidence-map response
- \(E\)：Shannon entropy
- \(C\)：灰度标准差，表示全局对比度
- \(S\)：speckle index

当前在线配置：

```text
quality_max_size             = None
quality_speckle_max_size     = None
quality_confidence_max_size  = 110
quality_confidence_method    = fast
```

也就是说：

- \(E,C,S\) 使用裁剪后的全分辨率图像
- \(D\) 的 confidence map 使用最大边长 110 的降采样图像，加速计算

### 5.3 特征归一化

代码中的默认归一化参数：

```text
D_min=0, D_max=1
E_min=0, E_max=8
C_min=0, C_max=128, C0=64
S_min=0, S_max=2
```

归一化为：

\[
d =
\frac{D-D_{\min}}{D_{\max}-D_{\min}}
\]

\[
e =
\frac{E-E_{\min}}{E_{\max}-E_{\min}}
\]

\[
c =
1 -
\frac{|C-C_0|}{C_{\max}-C_{\min}}
\]

\[
s =
\frac{S_{\max}-S}{S_{\max}-S_{\min}}
\]

然后裁剪到 \([0,1]\)。

### 5.4 TOPSIS 融合

权重：

\[
[w_D,w_E,w_C,w_S]
=
[0.4833,\ 0.1620,\ 0.0794,\ 0.2754]
\]

代码中会再次归一化权重和。

加权向量：

\[
\mathbf{q}
=
[w_Dd,\ w_Ee,\ w_Cc,\ w_Ss]^T
\]

正理想点：

\[
\mathbf{q}^{+}
=
[w_D,w_E,w_C,w_S]^T
\]

负理想点：

\[
\mathbf{q}^{-}
=
\mathbf{0}
\]

距离：

\[
D^+
=
\|\mathbf{q}-\mathbf{q}^{+}\|_2
\]

\[
D^-
=
\|\mathbf{q}-\mathbf{q}^{-}\|_2
\]

最终质量分数：

\[
Q
=
\frac{D^-}{D^+ + D^-}
\]

\(Q\in[0,1]\)，越大表示图像质量越好。

## 6. 力和力矩惩罚项

当前目标函数使用的是软惩罚结构：

\[
F(\mathbf{x})
=
-Q(\mathbf{x})
+
P_f(\mathbf{x})
+
P_\tau(\mathbf{x})
\]

其中 \(F\) 越小越好。

代码：

```python
compute_force_torque_penalty(...)
compute_online_objective(...)
```

### 6.1 力定义

保存的补偿后力为：

\[
\mathbf{f}=[f_x,f_y,f_z]^T
\]

当前代码把压力定义为：

\[
F_n = \max(0,\ -f_z)
\]

切向力：

\[
F_t =
\sqrt{f_x^2+f_y^2}
\]

### 6.2 力惩罚

目标压力区间：

```text
pressure_min = 2 N
pressure_max = 8 N
```

压力低于 2 N 或高于 8 N 都惩罚：

\[
F_{\mathrm{low}}
=
\max(0,\ F_{\min}-F_n)
\]

\[
F_{\mathrm{high}}
=
\max(0,\ F_n-F_{\max})
\]

切向力上限尺度：

```text
shear_max = 6 N
```

力惩罚：

\[
P_f
=
\lambda_p
\left[
\left(
\frac{F_{\mathrm{low}}}{F_{\min}}
\right)^2
+
\left(
\frac{F_{\mathrm{high}}}{F_{\max}}
\right)^2
\right]
+
\lambda_s
\left(
\frac{F_t}{F_{t,\max}}
\right)^2
\]

当前默认：

```text
lambda_pressure = 0.11
lambda_shear    = 0.04
```

### 6.3 力矩惩罚

轴向力矩：

\[
\tau_n = |\tau_z|
\]

切向力矩：

\[
\tau_t =
\sqrt{\tau_x^2+\tau_y^2}
\]

当前尺度：

```text
torque_tangential_max = 0.8 N m
torque_axial_max      = 0.5 N m
```

力矩惩罚：

\[
P_\tau
=
\lambda_\tau
\left(
\frac{\tau_t}{\tau_{t,\max}}
\right)^2
+
\lambda_{\tau z}
\left(
\frac{|\tau_z|}{\tau_{z,\max}}
\right)^2
\]

当前默认：

```text
lambda_torque        = 0.08
lambda_axial_torque  = 0.02
```

## 7. 目标函数消融版本

参数：

```bash
--surface-bo-objective-variant full
```

可选：

```text
full         F = -Q + P_f + P_tau
no_penalty   F = -Q
force_only   F = -Q + P_f
torque_only  F = -Q + P_tau
```

GUI `bo` 模式中对应按钮：

- `Run BO full`
- `Run BO no penalty`
- `Run BO force only`
- `Run BO torque only`

## 8. BO / random / LHS 搜索策略

参数：

```bash
--surface-bo-search-strategy bo
```

可选：

```text
bo
random
lhs
uniform
```

当前 GUI 按钮中：

- `Run BO full` 使用 `bo/full`
- `Run random full` 使用 `random/full`
- `Run LHS full` 使用 `lhs/full`

注意：代码里的 `"uniform"` 当前和 `"lhs"` 走同一个 `_DirectSearchOptimizer._make_lhs_points()`，GUI 实际按钮叫 `Run LHS full`。

## 9. BO 预算和 EI 参数

当前 GUI 默认：

```text
n_initial = 3
n_ei      = 15
max_trials = 18
```

参数：

```bash
--surface-bo-n-initial 3
--surface-bo-n-ei 15
```

`bo` 策略使用 `LocalBayesOptimizer`：

- 初始点：分层随机初始点，不是普通独立随机点
- GP kernel：

\[
k
=
C\cdot \mathrm{Matern}_{\nu=2.5}
+
\mathrm{WhiteKernel}
\]

- `normalize_y=True`
- `n_restarts_optimizer=2`
- EI 候选采样数：`candidate_count = 4096`
- 基础探索参数：`xi = 0.01`
- 若连续 3 次 best 没有明显变化，使用更强探索：`xi_boost = 0.1`
- 收敛窗口：`convergence_window = 7`
- 最小改善阈值：`min_improvement = 1e-4`

最小化问题的 EI：

\[
I(\mathbf{x})
=
f^\star
-
\mu(\mathbf{x})
-
\xi
\]

\[
Z
=
\frac{I(\mathbf{x})}{\sigma(\mathbf{x})}
\]

\[
\mathrm{EI}(\mathbf{x})
=
I(\mathbf{x})\Phi(Z)
+
\sigma(\mathbf{x})\phi(Z)
\]

其中：

- \(f^\star\)：当前观测到的最小 \(F\)
- \(\mu,\sigma\)：GP 后验均值和标准差
- \(\Phi,\phi\)：标准正态分布 CDF 和 PDF

## 10. 机器人移动和等待

每个 BO 候选点：

1. 计算目标 TCP
2. 插值 `servoL/move` 到目标 TCP
3. 等待 `settle_s`
4. 读取当前 obs
5. 计算 \(Q,P_f,P_\tau,F\)
6. `optimizer.tell(x,F)`

当前移动参数：

```text
max_position_step_m   = 0.001 m
max_rotation_step_rad = 0.006 rad
position_tolerance_m  = 0.002 m
rotation_tolerance_rad = 0.03 rad
timeout_s = 60 s
settle_s = 0.2 s
```

参数：

```bash
--surface-bo-settle-s 0.2
```

BO 结束后：

1. 移动回历史最优 \(x^\star\) 对应的 TCP
2. 等待 `settle_s`
3. 记录一次 `verified_best`
4. 等待 `--surface-bo-post-run-wait-s`
5. 沿法向外退 `--surface-bo-reset-retreat-m`
6. 回到保存的初始 \(x_0\) TCP

当前默认：

```text
surface_bo_post_run_wait_s = 1.0 s
surface_bo_reset_retreat_m = 0.15 m
```

## 11. 记录的数据

BO 记录使用 `EpisodeRecorder.save_sample()`，因此每个 pkl 中仍然包含常规观测字段，例如：

- `joint_positions`
- `ee_pos_rotvec`
- `tcp_position_base`
- `tcp_x_axis_base`
- `tcp_y_axis_base`
- `tcp_z_axis_base`
- `probe_tip_position_base`
- `path_nearest_index`
- `path_target_positions_base`
- `path_normals_base`
- `force`
- `force_raw`
- `force_gravity`
- `force_bias`
- `force_pressure_n`
- `force_tangential_norm_n`
- `torque_axial_z_nm`
- `torque_tangential_norm_nm`
- `Ultrasound_gray` / `Ultrasound_rgb`
- RGB/depth 字段，除非启动时加 `--skip-rgb-depth-recording`

BO 额外写入 `meta` 字段：

```text
auto_phase
bo_is_measurement
bo_counts_toward_budget
bo_measurement_role
bo_trial_index
bo_phase
bo_search_strategy
bo_objective_variant
bo_x
bo_target_tcp_pose
best_F
best_x
F
Q
D
E
C
S
P_f
P_tau
force_valid
```

含义：

- `bo_measurement_role="before"`：BO 前的参考点测量，不计入预算
- `bo_measurement_role="candidate"`：候选点测量，计入预算
- `bo_measurement_role="verified_best"`：BO 结束后回到历史最佳点再测一次，不计入预算
- `bo_is_measurement=False` 且 `auto_phase="bo_move"`：移动插值过程中的 waypoint 记录
- `bo_counts_toward_budget=True`：只有候选点为 True

除了 pkl，BO episode 文件夹还会保存：

```text
surface_bo_run.json
surface_bo_posterior.npz
```

`surface_bo_run.json` 包含：

- `reference_tcp_pose`
- `normal_base`
- `bounds`
- `n_initial`
- `n_ei`
- `search_strategy`
- `objective_variant`
- 惩罚项参数
- `before`
- `verified_best`
- `best_x`
- `best_F_observed`
- 所有 `trials`

`surface_bo_posterior.npz` 包含：

- `observed_x`
- `observed_F`
- 每个维度的 posterior slice：
  - `dn_grid`, `dn_mean`, `dn_std`, `dn_ei`
  - `rx_grid`, `rx_mean`, `rx_std`, `rx_ei`
  - `ry_grid`, `ry_mean`, `ry_std`, `ry_ei`
  - `rz_grid`, `rz_mean`, `rz_std`, `rz_ei`

## 12. 常用启动命令示例

只做 BO 测试，需要超声：

```bash
python -m visual_guided_collection_gui.main \
  --operation-mode bo \
  --control-tcp \
  --wrist-camera Orbbec \
  --gello-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAW77E7-if00-port0 \
  --force-ip 192.168.1.100 \
  --skip-rgb-depth-recording
```

如果没有力传感器，只验证 \(Q\)：

```bash
python -m visual_guided_collection_gui.main \
  --operation-mode bo \
  --control-tcp \
  --wrist-camera Orbbec \
  --gello-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAW77E7-if00-port0 \
  --disable-force \
  --skip-rgb-depth-recording
```

如果没有超声，不能运行在线 BO，因为目标函数需要实时超声图像。

## 13. 修改入口速查

BO 变量范围：

```text
visual_guided_collection_gui/main.py
--surface-bo-bounds
```

BO 次数：

```text
visual_guided_collection_gui/main.py
--surface-bo-n-initial
--surface-bo-n-ei
```

力惩罚参数：

```text
visual_guided_collection_gui/main.py
--surface-bo-pressure-min
--surface-bo-pressure-max
--surface-bo-shear-max
--surface-bo-torque-tangential-max
--surface-bo-torque-axial-max
--surface-bo-lambda-pressure
--surface-bo-lambda-shear
--surface-bo-lambda-torque
--surface-bo-lambda-axial-torque
```

目标函数结构：

```text
visual_guided_collection_gui/surface_bayes.py
compute_online_objective()
compute_force_torque_penalty()
```

候选位姿映射：

```text
visual_guided_collection_gui/surface_bayes.py
compute_candidate_tcp_pose()
```

EI/GP：

```text
visual_guided_collection_gui/offline_bayes.py
LocalBayesOptimizer
```

超声质量评分：

```text
visual_guided_collection_gui/ultrasound_quality.py
UltrasoundQualityScorer
```

