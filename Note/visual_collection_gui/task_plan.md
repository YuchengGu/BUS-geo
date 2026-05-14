# 任务计划：可视化采集界面

## 目标

这里的 GUI 指的就是“可视化界面/采集窗口”。这个任务只负责把当前采集过程做成一个可视化界面，替代现在的：

```bash
python experiments/launch_nodes.py --robot ur
python experiments/run_env.py --agent=gello --use_save_interface
```

以及 pygame 灰色/绿色窗口、按 `s` 开始、`Ctrl-C` 结束的流程。

窗口需要让操作者能完成：

- 连接设备。
- 看 D405 RGB/depth。
- 看力传感器六维输出。
- 看乳腺点云/分割结果。
- 看原规划路径。
- 看当前超声探头尖端位置与姿态。
- 看扫查进度。
- 点一个键调用独立路径规划文件夹里的规划函数。
- 开始记录 episode。
- 切换结节精扫 flag。
- 结束 episode。
- 优雅退出，不靠 `Ctrl-C`。

## 和路径规划文件夹的关系

- `breast_path_planning/` 负责真正的路径规划函数。
- 本可视化界面负责提供按钮、显示输入输出、调用这些函数，并加载结果。
- 可视化界面本身不重新实现路径规划算法。

换句话说：

```text
路径规划算法怎么做 -> breast_path_planning/
用户什么时候触发规划 -> 可视化界面的“规划路径”按钮
规划结果如何显示和用于采集 -> 可视化界面加载 planned_path.json
```

## 当前问题

当前 `gello/data_utils/keyboard_interface.py` 只做了：

- pygame 800x800 窗口。
- 灰色表示 normal。
- 按 `s` 后变绿色并开始保存。
- 按 `q` 回普通状态。

它不显示图像，所以用户看到“灰色界面变绿”不是图像失败，而是这个界面本来就只有颜色状态。

## GUI 第一版布局

### 左侧：实时传感器

- D405 RGB 图像。
- D405 depth 伪彩或深度预览。
- Fx/Fy/Fz/Tx/Ty/Tz。
- 当前 joint positions / joint velocities。
- 当前 TCP 和 probe tip 坐标。

### 中间：3D 视图

- 乳腺点云或分割点云。
- 原规划路径。
- 当前最近路径点。
- 当前 probe tip。
- probe x/y/z 轴方向。
- 进度条：`path_nearest_index / (N - 1)`。

### 右侧：操作区

- `连接设备`
- `拍一帧 RGB-D`
- `选择乳腺 seed`
- `分割乳腺`
- `规划路径`
- `加载路径`
- `开始记录 episode`
- `结节精扫 flag: OFF/ON`
- `结束 episode`
- `停止并安全退出`

其中 `规划路径` 按钮调用 `breast_path_planning/` 里的规划函数，输入是当前冻结的 RGB-D/点云和分割结果，输出是 `planned_path.json`。

## 状态机

建议 GUI 明确维护状态：

```text
Idle
  -> Connected
  -> PathLoaded
  -> Recording
  -> EpisodeStopped
  -> Shutdown
```

分割和路径规划在界面里是操作步骤，但算法实现仍在独立文件夹：

```text
Connected
  -> FrameCaptured
  -> Segmented
  -> PathPlanned
  -> Recording
```

## episode 记录语义

点击 `开始记录 episode`：

- 创建 episode 目录。
- 写 `episode_manifest.json`。
- `recording=True`。
- `sample_index=0`。
- `fine_scan_flag=0`。

点击 `结节精扫 flag`：

- OFF -> ON：后续 pkl 写 `fine_scan_flag=1`。
- ON -> OFF：后续 pkl 写 `fine_scan_flag=0`。
- 同时写 `operator_events.jsonl`。

点击 `结束 episode`：

- `recording=False`。
- flush 状态。
- 写 episode summary。
- 不退出程序。

点击 `停止并安全退出`：

- 若正在记录，先结束 episode。
- 停止控制循环。
- 关闭相机、force、robot client、窗口。

## 技术路线

推荐 Python GUI：

- PySide6 或 PyQt5/PyQt6 做窗口。
- OpenCV/Numpy 显示 D405 RGB/depth。
- pyqtgraph/Open3D/VTK 三选一做 3D 视图。
- 控制循环和 GUI 分线程，避免渲染卡住 robot control。

不建议第一版直接把 demo1 C++ Qt 程序改成采集 GUI，因为当前 pkl、GELLO agent、ZMQ robot client、time alignment 都在 Python 里。

## 实现步骤

### 阶段 1：最小 GUI 替代 SaveInterface
- [ ] 新建 GUI save interface。
- [ ] 支持 start/stop episode。
- [ ] 支持 fine scan flag。
- [ ] 显示 D405 RGB 和 force。
- [ ] 结束时优雅退出。

### 阶段 2：路径可视化
- [ ] 读取 `planned_path.json`。
- [ ] 显示路径线和抽样法向。
- [ ] 显示当前 probe tip 和 xyz 轴。
- [ ] 显示最近点和扫查进度。

### 阶段 3：接入拍照/分割/规划
- [ ] `拍一帧 RGB-D` 冻结当前 D405 数据。
- [ ] 点击 seed。
- [ ] 调用 `breast_path_planning/segmentation.py` 的分割函数。
- [ ] 点击 `规划路径` 后调用 `breast_path_planning/path_planner.py` 的规划函数。
- [ ] 生成并加载 `planned_path.json`。
- [ ] 在 3D 视图中显示新规划路径。

### 阶段 4：验收
- [ ] 不用 `Ctrl-C` 能正常停止。
- [ ] pkl 中 `fine_scan_flag` 与按钮一致。
- [ ] `operator_events.jsonl` 记录 start/flag/stop。
- [ ] GUI 显示的进度和离线 `path_nearest_index` 一致。

## 成功标准

- 操作者能在一个窗口里看到采集关键状态。
- 操作者能在可视化界面中点击 `规划路径`，实际调用独立路径规划文件夹里的函数。
- episode 开始、精扫 flag、episode 结束都有明确按钮和日志。
- 采集结束不再靠 `Ctrl-C`。
- GUI 不改变 `obs_t -> action_t` 的数据语义，只是管理状态和可视化。
