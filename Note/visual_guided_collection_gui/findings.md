# 发现：可视化引导采集 GUI

## 任务定位

这个任务不是单独给 `run_env.py` 加路径参数，而是要做采集入口本身：

```text
GUI 中先用 GELLO 摆位到适合拍照的位置，再拍照、规划、确认，最后示教采集。
```

路径 residual / normal 保存是 GUI 采集流程的一部分。

## 关键复用点

- 点云分割、规划、路径可视化已在 `breast_path_planning/` 中实现。
- `path_features.py` 已有 residual / normal 计算函数。
- `RobotEnv.get_obs()` 已有 TCP 位置和 xyz 轴。
- `control_utils.run_control_loop()` 当前保存 `current_obs` 和 `action`，这符合 `obs_t -> action_t`。
- 拍照时刻必须冻结当前 `T_base_tcp`，因为 D405 是手腕相机，后续点云转 base 依赖这个位姿。
- D405 是独占资源；最终 GUI 必须只创建一个 `RealSenseD405` 实例，并在摆位、规划、采集阶段复用。

## 关键风险

- Open3D 事件循环和 GUI 框架事件循环需要谨慎处理。
- Robot 控制循环不能被点云渲染卡住。
- 不能让规划脚本和采集脚本同时打开 D405，否则 RealSense pipeline 可能启动失败或读帧超时。
- 摆位模式只控制 UR5，不应保存 episode 样本；确认规划后才开始正式记录。
- `probe_tip_offset_m` 的正负号必须实物确认。
- 只有真实 D405 在线规划出来的 base 坐标系路径，才能直接用于真实 UR 采集 residual。

## 第一版实现决策

- `Newgello` 环境已有 `open3d 0.19.0`、`PyQt5`、`opencv-python`、`pyrealsense2`，不需要新增大依赖。
- 为了最小化依赖和事件循环复杂度，第一版使用 Open3D 自带 GUI：

```text
open3d.visualization.gui.Application
open3d.visualization.gui.SceneWidget
open3d.visualization.rendering.Open3DScene
```

- seed 选择不再依赖独立 Open3D picker 窗口，而是在主 `SceneWidget` 中用当前相机矩阵把点云投影到屏幕，找 `Shift + 左键` 附近最近的点云点。
- `visual_guided_collection_gui/` 是新入口；原来的命令行规划和采集代码仍保留，便于回退。
