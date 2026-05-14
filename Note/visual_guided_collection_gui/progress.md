# 进度：可视化引导采集 GUI

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
