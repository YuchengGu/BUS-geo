# 任务计划：可视化引导采集 GUI

## 目标

这个子任务覆盖“每次采集时打开一个可视化 GUI，先用 GELLO 遥操作 UR5 到适合拍照的位置，再拍照、分割、规划，确认规划结果后，直接在同一个界面启动 GELLO 示教采集”。

它不是单独的 `run_env.py` 参数任务。最终操作者期望的流程是：

```text
打开可视化采集 GUI
  -> 连接 UR / D405 / force / GELLO
  -> 进入拍照定位模式
  -> 用 GELLO 遥操作 UR5，把 D405 移到适合拍照/看清乳腺的位置
  -> 冻结当前 UR TCP 位姿和 D405 RGB-D
  -> 生成 base 坐标系 RGBXYZ 点云
  -> Open3D 点云中选 seed
  -> 分割并高亮乳腺区域
  -> 确认分割
  -> 规划路径
  -> 显示分割点云 + 路径点 + 法向
  -> 确认规划结果
  -> 按一个键开始正式 GELLO 示教采集和保存 episode
  -> 每帧保存路径 residual / normals / progress
  -> 按键结束 episode
```

核心训练语义仍保持：

```text
obs_t -> action_t
```

路径 residual 必须由保存前的 `current_obs` 计算，不能用 `env.step(action)` 后的新观测。

## 与路径规划任务的边界

`breast_path_planning/` 只负责：

```text
D405/PLY 点云
  -> seed 分割
  -> 规划路径
  -> planned_path.json
  -> 路径和法向可视化
```

本 GUI 子任务负责：

```text
设备连接
  -> GELLO 拍照定位模式
  -> 调用 breast_path_planning 的分割/规划能力
  -> 规划确认
  -> 启动/停止 GELLO 采集
  -> 每帧写入路径 residual 和法向
  -> 显示采集状态
```

## 已有可复用代码

- `breast_path_planning/live_plan_from_d405.py`
  - 已能连接 UR、打开 D405、生成 base 坐标系 RGBXYZ 点云。
- `breast_path_planning/interactive_pointcloud.py`
  - 已能 Open3D 选 seed、分割高亮、确认/重选。
- `breast_path_planning/visualize_path.py`
  - 已能在一个 Open3D 窗口中显示分割点云、路径点和法向。
- `breast_path_planning/path_io.py`
  - 已能读取 `planned_path.json`。
- `breast_path_planning/path_features.py`
  - 已能根据当前探头位置计算最近路径点、lookahead residual 和法向。
- `gello/env.py`
  - 已在 obs 中提供：

```text
tcp_position_base
tcp_x_axis_base
tcp_y_axis_base
tcp_z_axis_base
```

- `gello/utils/control_utils.py`
  - 当前保存点明确，保存的是 `current_obs` 和同一时刻的 `action`。

## 坐标系约束

路径残差只有在同一坐标系下才有意义。

真实 D405 在线规划时：

```text
P_camera -> T_base_camera -> P_base
```

此时 `planned_path.json` 里的：

```text
position_base
normal_base
```

与采集时 `RobotEnv.get_obs()` 保存的：

```text
tcp_position_base
tcp_x_axis_base
tcp_y_axis_base
tcp_z_axis_base
```

处在同一个 UR base 坐标系，可以直接计算残差。

外部 PLY 离线测试时，如果 PLY 本身不是 UR base 坐标系，那么生成的 `planned_path.json` 只能用于测试 GUI、分割、规划和可视化流程，不能直接用于真实机器人采集残差。

## D405 独占资源约束

D405 / RealSense pipeline 是独占资源。当前 `RealSenseD405()` 内部会：

```text
rs.pipeline()
pipeline.start(config)
```

同一台 D405 不能被两个进程或两个独立 pipeline 同时打开。禁止出现：

```text
live_plan_from_d405.py 正在打开 D405
run_env.py 同时也打开 D405
```

否则可能出现：

```text
第二个进程打不开相机
pipeline.start 报错
wait_for_frames 超时
帧读取卡住或不稳定
```

最终 GUI 必须统一管理 D405 生命周期：

```text
GUI 启动
  -> 创建一个 RealSenseD405 实例
  -> 摆位模式持续读同一个 D405
  -> 冻结一帧用于分割规划
  -> 正式采集时继续复用同一个 D405
  -> GUI 退出时 pipeline.stop()
```

临时脚本阶段可以先运行 `live_plan_from_d405.py` 生成路径，再关闭该脚本后运行 `run_env.py`。但不能两个同时运行。统一 GUI 阶段必须消除这种双进程抢相机的风险。

## 探头尖端位置

训练中更应该使用超声探头尖端，而不是 UR TCP 原点。

建议定义：

```text
probe_tip_position_base = tcp_position_base + probe_tip_offset_m * tcp_z_axis_base
probe_x_axis_base = tcp_x_axis_base
probe_y_axis_base = tcp_y_axis_base
probe_z_axis_base = tcp_z_axis_base
```

注意：

- `probe_tip_offset_m` 必须配置化。
- 正负号需要用实物确认。
- demo1 里出现过约 `0.217 m` 和 `0.223 m` 的探头长度，不能硬编码。

## 每帧应保存的路径特征

每个 pkl 顶层 obs 应加入：

```text
probe_tip_position_base
probe_x_axis_base
probe_y_axis_base
probe_z_axis_base

path_nearest_index
path_progress
path_indices
path_lookahead_mask
path_target_positions_base
path_residuals_base
path_normals_base
path_distance_to_nearest_m
```

其中：

```text
path_residuals_base[j] = planned_path[k+j].position_base - probe_tip_position_base
path_normals_base[j] = planned_path[k+j].normal_base
```

`path_nearest_index` 建议使用单调局部搜索，避免最近点在路径上前后跳动：

```text
search_window = [last_index - backtrack, last_index + forward]
```

当前默认建议：

```text
lookahead = 8
backtrack = 3
forward = 7
```

含义：

```text
最近点搜索窗口：last_index - 3 到 last_index + 7
每帧保存 residual/normal 数量：8 个，即 k 到 k+7
```

## GUI 第一版范围

第一版 GUI 不需要做复杂美化，重点是把流程打通：

```text
1. 连接设备
2. GELLO 摆位模式：遥操作 UR5 到适合 D405 拍照的位置
3. 冻结 D405 当前帧和拍照时刻 TCP 位姿
4. 分割点云
5. 规划路径
6. 可视化确认路径与法向
7. 开始记录 episode
8. 切换 fine_scan_flag
9. 停止记录 episode
10. 优雅退出
```

窗口至少显示：

```text
D405 RGB / depth
force 六维输出
分割点云
规划路径点
路径法向
当前 probe tip
当前 probe xyz 轴
path_nearest_index / path_progress
fine_scan_flag
recording 状态
```

## 实现路线

### 阶段 1：GUI 调用现有规划流程

- [ ] 新建可视化采集 GUI 入口。
- [ ] 支持连接 UR / D405 / force。
- [ ] GUI 内只创建一个 `RealSenseD405` 实例，并在摆位、规划、采集三个阶段复用它。
- [ ] 支持 GELLO 摆位模式：先不保存 episode，只让操作者把 UR5/D405 移到适合拍照的位置。
- [ ] 支持冻结拍照时刻的 D405 frame 和 `T_base_tcp`。
- [ ] 用冻结的 `T_base_tcp @ T_tcp_camera` 生成 base 坐标系点云。
- [ ] 调用 `interactive_pointcloud.py` 做 seed 选择和分割高亮确认。
- [ ] 调用 `plan_from_segmented_cloud()` 生成 `planned_path.json`。
- [ ] 调用 `visualize_path.py` 或其内部函数显示分割点云、路径点和法向。

### 阶段 2：GUI 管理 episode 采集

- [ ] GUI 内部启动 GELLO agent。
- [ ] 用按钮替代当前 pygame 灰色/绿色保存界面。
- [ ] `开始记录 episode` 创建 episode 目录。
- [ ] `精扫 flag` 能在采集中切换，并写入每帧 pkl。
- [ ] `结束 episode` 停止保存但不强制退出程序。
- [ ] `停止并退出` 能关闭相机、robot client、force、窗口。
- [ ] 退出时确保 D405 `pipeline.stop()` 被调用。

### 阶段 3：每帧写路径 residual / normal

- [ ] 加载 GUI 当前确认的 `planned_path.json`。
- [ ] 每帧用 `current_obs` 计算 `probe_tip_position_base`。
- [ ] 调用 `compute_path_features()`。
- [ ] 将路径 residual、路径法向、progress 写入同一帧 pkl。
- [ ] 新 episode 开始时重置 `last_path_index`。

### 阶段 4：测试与验收

- [ ] 单测验证无路径时旧采集字段不变。
- [ ] 单测验证有路径时 pkl 多出 residual/normal/progress。
- [ ] 单测验证 residual 来自 `current_obs` 而不是 step 后 obs。
- [ ] 验证 GUI 不会同时启动两个 D405 pipeline。
- [ ] 实机验证 GUI 能完成“GELLO 摆位 -> 拍照 -> 规划确认 -> 开始示教 -> 结束 episode”。

## 成功标准

- 操作者每次采集只需要打开一个 GUI。
- GUI 中先用 GELLO 把 UR5/D405 摆到适合拍照的位置，再完成拍照、分割、规划和确认，最后启动 GELLO 示教采集。
- 每个保存的 pkl 都包含路径 residual、路径法向和进度。
- `obs_t -> action_t` 数据语义不变。
- D405 只由 GUI 内的一个相机实例管理，不被规划和采集两个进程同时占用。
- 不再依赖灰色/绿色 pygame 保存界面。
- 不靠 `Ctrl-C` 结束 episode。
