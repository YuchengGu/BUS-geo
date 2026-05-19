# 进度：可视化引导采集 GUI

## 2026-05-14 第一版 Open3D GUI 入口

- 新增 `visual_guided_collection_gui/` 作为独立 GUI 代码文件夹。
- 第一版使用 Open3D GUI 单窗口，不新增 PySide/网页前端依赖。
- 新增左侧 `SceneWidget` 点云视图、右侧 RGB/状态/按钮区。
- 新增 GUI 状态机、投影拾取、episode recorder、设备连接、规划会话、采集循环模块。
- 每帧保存路径 residual / normals / progress / fine_scan_flag 的硬件无关逻辑已加测试。
- 原有 `run_env.py`、`breast_path_planning/` 核心分割规划代码未被替换。
- `开始记录 episode` 前增加显式设备/GELLO 连接检查；拍照规划阶段只停止控制循环，不断开 GELLO。
- GUI 按钮拆成两个：`拍照摆位` 和 `采集前 GELLO 接管`。两者底层都启动 GELLO 控制循环，但在不同状态启用，下一步分别是 `拍照冻结` 和 `开始记录 episode`。
- GUI 右侧预览扩展为 `D405 RGB`、`D405 depth` 伪彩色和 `Ultrasound` 三块图像；默认接入 `UltrasoundCamera(camera_index=5)`，可用 `--disable-ultrasound` 关闭或 `--ultrasound-index` 改编号。

## 2026-05-14

- [x] 明确该任务包含“先用 GELLO 摆位，再拍照分割规划，最后示教采集”的完整 GUI 流程。
- [x] 将路径 residual / normal 保存并入 GUI 采集任务。
- [x] 明确 GUI 采集时仍保持 `obs_t -> action_t`。
- [x] 明确第一版 GUI 应复用 `breast_path_planning/` 的点云分割、规划和可视化能力。
- [x] 明确 D405 是独占资源，GUI 必须统一管理一个 D405 实例，不能规划和采集双进程同时打开。
- [ ] 实现 GUI 入口。
- [ ] 实现 GUI 内 GELLO 拍照摆位模式。
- [ ] 实现 GUI 内规划确认。
- [ ] 实现 GUI 内启动/停止 GELLO episode。
- [ ] 实现每帧保存路径 residual / normal / progress。
