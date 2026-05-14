# 代码发现：可视化采集窗口

## 当前保存界面

- `gello/data_utils/keyboard_interface.py:12-17` 初始化 pygame 窗口并填充灰色。
- `gello/data_utils/keyboard_interface.py:19-35` 按 `s` 后变绿色并持续返回 `save`。
- 这个界面没有图像显示、路径显示、force 显示或 flag 状态。

## 当前采集入口

- `experiments/run_env.py:270-278` 在 `--use_save_interface` 时创建 `SaveInterface` 并进入 `run_control_loop()`。
- `gello/utils/control_utils.py:129-176` `SaveInterface` 只负责创建保存目录和轮询键盘状态。
- `gello/utils/control_utils.py:315-333` 保存发生在控制循环内部，GUI 需要提供同样的 recording 状态和 episode 路径。

## demo1 可参考的 GUI 行为

- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:567-589` 有相机 color/depth 图像刷新。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1111-1165` 有点云、路径线、路径点局部坐标轴显示。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2255-2259` 有基于轨迹索引的进度条。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:3251-3294` 有记录开始、创建 session 目录、打开输出文件的思路。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:3296-3372` 有停止记录、关闭文件、导出 summary 的思路。

## 当前应该避免的点

- 不把 GUI 渲染放在控制循环里。
- 不用 `Ctrl-C` 作为正常结束路径。
- 不让 GUI 改变 `control` 的语义。
- 不在第一版 GUI 中强行迁移 demo1 的自动力控和 BO。

## 结论

GUI 是第二个大任务，职责是可视化和状态管理；路径规划算法和路径残差字段属于另一个任务。

