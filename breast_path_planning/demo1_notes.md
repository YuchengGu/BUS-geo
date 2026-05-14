# demo1 逻辑对应关系

这个文件记录 Python 版本和 `/home/ubuntu22/dev/demo1/src` 的对应关系。目标是复刻逻辑步骤，不复制 C++/Windows 工程。

## 坐标转换

- demo1：`CoordinationTransform.cpp`
  - `getBaseToCameraTrans() = T_Base_Flange * T_Flange_Camera`
  - `image_acquisition.cpp` 中点云从 mm 转 m 后乘 `T_Base_Camera`
- librealsense 当前帧：
  - `pc.map_to(color_frame)`
  - `points = pc.calculate(depth_frame)`
  - `points.get_vertices()` 得到 D405 相机坐标系下的点云
- Python：
  - `pointcloud_from_d405.realsense_frames_to_point_cloud()`
  - 输入 `T_base_camera = T_base_tcp @ T_tcp_camera`
  - 不再接收手动 `camera_matrix`；D405 内参由 librealsense pointcloud 在内部使用

## 分割

- demo1：`mainwindow.cpp`
  - 点选 seed
  - 用 seed 平均 HSV + 空间半径做区域生长
- Python：
  - `segmentation.segment_region_from_seed_pixels()`

## 表面处理

- demo1：
  - `smoothPointCloudMLS()`
  - `makeCloudWithNormals()`
  - `constrainNormal()`
- Python：
  - `surface_processing.estimate_normals()`
  - `surface_processing.constrain_normals_to_reference()`

第一版不完整复刻 PCL MLS，只做轻量法向估计。法向角度约束默认关闭，因为乳腺侧面真实法向可能超过 30 度；只有显式设置 `max_normal_angle_deg` 时才启用约束。

## 路径规划

- demo1：
  - `planAdaptiveSlicePath()`
  - `generateSerpentinePath()`
  - `geodesic resample` 和后续优化
  - `computePathTangents()`
- Python：
  - `path_planner.plan_serpentine_path()`

第一版只做 adaptive slice + serpentine，并保留真实估计法向。暂不实现测地线优化、tangent、reverse_y。
