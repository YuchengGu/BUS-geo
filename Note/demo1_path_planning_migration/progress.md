# 进度：demo1 路径规划迁移

## 2026-05-13

- [x] 按用户要求将任务收敛为两个大任务之一。
- [x] 保留并合并原先“探头 Z 轴延长”和“路径残差 schema”内容，作为本任务的顺手子项。
- [x] 读取 `/home/ubuntu22/dev/demo1/CMakeLists.txt`。
- [x] 读取 demo1 路径规划、坐标变换、点云、分割相关源码。
- [x] 明确第一版走 `planned_path.json` 文件桥接。
- [x] 新建独立 Python 路径规划文件夹 `breast_path_planning/`。
- [x] 实现从当前 D405 RGB-D frame 生成 base 点云的函数。
- [x] 移除需要手动输入相机内参的 RGB/depth 反投影入口，只保留 librealsense pointcloud 路径。
- [x] 实现 seed + HSV/空间邻域区域生长分割。
- [x] 实现法向估计、adaptive slice、serpentine；normal constrain 已保留为可选参数，默认关闭。
- [x] 实现 `planned_path.json` 读写，第一版只含 `position_base` 和 `normal_base`。
- [x] 实现离线路径特征函数，供 GUI/采集侧以后计算 residual。
- [x] 新增在线终端脚本 `breast_path_planning/live_plan_from_d405.py`：打开 D405、读取 UR 当前 TCP、生成 base 坐标系 RGBXYZ 点云、在 Open3D 点云窗口里选 seed、高亮确认分割并保存结果。
- [x] 新增 PLY 离线测试入口 `breast_path_planning/plan_from_ply.py`，真实 D405 保存的 `raw_cloud_base.ply` 可重新读入复查/重规划。
- [x] 新增单元测试 `tests/test_breast_path_planning.py`。
- [ ] 后续接入可视化界面的“规划路径”按钮。
