# 任务计划：从 GELLO pkl 数据计算 D405 手眼外参

## 目标
基于当前 `gello_software` 已保存的 `.pkl` episode 数据，以及 `/home/ubuntu22/dev/python_opencv_camera_calibration-master` 中的 OpenCV 手眼标定示例，规划一条从“固定标定板的一整段 pkl 数据”到 `T_tcp_camera` 的可靠计算路径。

最终要得到：

```text
T_tcp_camera = ^tcp T_camera
```

也就是 D405 相机坐标系到 UR TCP 坐标系的 4x4 变换矩阵。之后可用于：

```text
P_base = T_base_tcp @ T_tcp_camera @ P_camera
```

把 D405 depth/点云/路径点从相机坐标系转换到机械臂 base 坐标系。

## 当前理解
- 用户希望先拍/采一整段固定标定板数据，然后从 pkl 中离散选帧做标定。
- 用户已有 pkl 字段：
  - `D405_rgb`
  - `D405_depth`
  - `ee_pos_rotvec`
  - `ee_pos_quat`
  - `joint_positions`
  - `joint_velocities`
  - `control`
  - `meta`
- `ee_pos_rotvec = [x, y, z, rx, ry, rz]`，来自 UR `getActualTCPPose()`，表示 TCP 在 robot base 下的位姿，单位位置为米，旋转为 rotvec。
- `D405_depth` 当前是对齐到 D405 color frame 的 raw `z16` 深度图；手眼标定本身主要用 `D405_rgb` 检测标定板角点，depth 不是主输入。
- 标定板不需要提前知道它在 robot base 下的位置；只要采集期间固定不动即可。

## 已读取的关键源码

### 当前 GELLO 仓库
- `gello/robots/ur.py`
  - `get_observations()` 读取 `getActualTCPPose()`。
  - 保存 `ee_pos_rotvec = tcp_pose`。
  - 保存 `ee_pos_quat = [x, y, z, qw, qx, qy, qz]`。
- `gello/cameras/D405.py`
  - D405 当前配置为 `848x480 RGB8 + Z16 depth @ 30 FPS`。
  - depth 已经通过 `rs.align(rs.stream.color)` 对齐到 color frame。
  - `last_metadata` 包含 `frame_new`、`frame_id`、`hardware_timestamp_ms`、`cache_age_ms`。
- `gello/env.py`
  - 每帧 obs 中保存 `D405_rgb`、`D405_depth`、robot lowdim、force。
  - `meta["modalities"]["D405"]` 保存相机帧 freshness。
- `gello/data_utils/format_obs.py`
  - pkl 顶层字段平行保存，`meta` 与 `D405_rgb`、`control` 同级。
- `chack_data.py`
  - 可读取单个 pkl，也可摘要检查 episode。
- `Note.md`
  - 用户的残差方案需要把规划路径点和当前 TCP 位置放在同一坐标系下。
  - 若规划路径来自 D405 点云，则需要 `T_tcp_camera` 把路径点转到 base。

### OpenCV 标定示例
- `/home/ubuntu22/dev/python_opencv_camera_calibration-master/README.md`
  - 说明该项目用 OpenCV `calibrateHandEye()` 做 UR5 手眼标定。
- `camera_calibration.py`
  - 用 `findChessboardCornersSB()` 检测棋盘格。
  - 用 `cv.calibrateCamera()` 求相机内参和畸变。
  - 用 `cv.solvePnP()` 对每张图求 `rvec/tvec`。
  - 用 `cv.calibrateHandEye()` 求相机相对于末端的外参。
- `main.py`
  - 原项目通过按 `S` 拍照，并保存 UR TCP pose 到 `data/xyz.txt` 和 `data/RxRyRz.txt`。
  - 你的 pkl 已经把图像和 TCP pose 存在同一个文件里，不需要额外 txt 同步。
- `robotcontrol.py`
  - 使用 RTDE `getActualTCPPose()` 读取当前 TCP 位姿。
- `util.py`
  - 有 rotvec、rotation matrix、rpy 的互转逻辑。
- 本地 `cv2.calibrateHandEye` 文档确认：
  - 输入 `R_gripper2base/t_gripper2base` 是 `^base T_gripper`。
  - 输入 `R_target2cam/t_target2cam` 是 `^camera T_target`。
  - 输出 `R_cam2gripper/t_cam2gripper` 是 `^gripper T_camera`。

## 示例代码不能直接照搬的地方
| 问题 | 原示例做法 | 当前应怎么做 |
|------|------------|--------------|
| 数据来源 | 图片在 `photo/`，机器人位姿在 txt | 直接从同一个 pkl 读 `D405_rgb` 和 `ee_pos_rotvec` |
| 单位 | 示例把 UR xyz 转成 mm，棋盘格 `checker_size` 用 mm | 当前建议全程用米；棋盘格边长也写成米 |
| 图像颜色 | 示例 `cv.imread()` 得到 BGR | pkl 里的 `D405_rgb` 是 RGB，检测前应转 gray，不要误用 BGR 假设 |
| 相机内参 | 示例用 `calibrateCamera()` 从图片估计 | 可选：用 D405 SDK 内参，或从标定板图片估计；必须和 RGB 分辨率匹配 |
| 缓存帧 | 示例没有 frame freshness | 当前必须过滤 `meta["modalities"]["D405"]["frame_new"] == False` 的帧 |
| 连续运动 | 示例按键拍照，姿态基本静止 | 如果你从连续 episode 中选帧，应过滤高速度帧，最好采集时每个姿态停一下 |
| 方向命名 | README 中有一些转置/求逆描述容易混乱 | 以 OpenCV 文档为准：输入 `T_base_tcp` 和 `T_camera_board`，输出 `T_tcp_camera` |

## 数学链路

每个有效标定帧 `i` 有两类观测：

### 1. 机器人给出的 TCP 位姿
来自 pkl：

```python
ee = frame["ee_pos_rotvec"]
t_base_tcp = ee[:3]          # meters
rvec_base_tcp = ee[3:]       # radians rotvec
R_base_tcp = cv2.Rodrigues(rvec_base_tcp)[0]
T_base_tcp = make_T(R_base_tcp, t_base_tcp)
```

这是：

```text
^base T_tcp
```

也就是 OpenCV hand-eye 参数里的 `gripper2base`。

### 2. RGB 图像检测标定板得到的标定板位姿
从 `D405_rgb` 检测棋盘格 / Charuco / AprilTag board，得到 2D-3D 对应点，然后：

```python
ok, rvec_target2cam, t_target2cam = cv2.solvePnP(
    object_points_board,
    image_points,
    camera_matrix,
    dist_coeffs,
)
R_target2cam = cv2.Rodrigues(rvec_target2cam)[0]
T_camera_board = make_T(R_target2cam, t_target2cam)
```

这是：

```text
^camera T_board
```

也就是 OpenCV hand-eye 参数里的 `target2cam`。

### 3. 手眼标定
多帧送入：

```python
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base=R_base_tcp_list,
    t_gripper2base=t_base_tcp_list,
    R_target2cam=R_camera_board_list,
    t_target2cam=t_camera_board_list,
    method=cv2.CALIB_HAND_EYE_TSAI,
)
```

输出：

```text
R_cam2gripper, t_cam2gripper = ^tcp T_camera
```

也就是：

```text
T_tcp_camera
```

之后 D405 点云转换为 base 坐标：

```text
P_base = T_base_tcp @ T_tcp_camera @ P_camera
```

路径点和法向转换：

```text
p_i^base = T_base_tcp_at_planning @ T_tcp_camera @ p_i^camera
n_i^base = R_base_tcp_at_planning @ R_tcp_camera @ n_i^camera
```

法向量只用旋转，不用平移。

## 从 pkl 数据计算的规划流程

### 阶段 1：采集标定 episode
- 固定标定板，不需要手量它的位置。
- 用手腕 D405 看标定板。
- 手动移动 UR5 到 20-40 个姿态，尽量每个姿态停一下。
- 每个姿态应有明显不同的平移和姿态：
  - 左右、上下、远近变化；
  - roll/pitch/yaw 有变化；
  - 不要全部正对标定板；
  - 不要只平移不旋转。
- 继续使用当前 pkl 保存链路即可，因为每帧已经有 `D405_rgb` 和 `ee_pos_rotvec`。

### 阶段 2：从 episode 中选帧
筛选规则：

```text
必须：
  meta.schema_version == "time_alignment_v1"
  D405_rgb 存在
  ee_pos_rotvec 存在
  meta.modalities.D405.valid == True
  meta.modalities.D405.frame_new == True

建议：
  joint_velocities 范数小，表示采样时机械臂接近静止
  D405 cache_age_ms 很小
  标定板角点检测成功
  姿态之间差异足够大
```

不要使用 `frame_new=False` 的帧。缓存帧的图像属于上一相机时刻，但 TCP pose 是当前时刻，二者会造成手眼标定误差。

### 阶段 3：检测标定板
推荐优先级：

1. **Charuco board**：推荐，鲁棒，角点精度好，部分遮挡也更稳。
2. **AprilTag grid**：也推荐，检测稳定，ID 可防止点顺序混乱。
3. **普通棋盘格**：可用，但需要完整可见，点序和内角点数量必须严格配置。

如果沿用示例项目，则普通棋盘格核心是：

```python
gray = cv2.cvtColor(D405_rgb, cv2.COLOR_RGB2GRAY)
ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
```

其中 `(cols, rows)` 是**内角点数量**，不是棋盘格方格数量。

### 阶段 4：相机内参处理
必须有和 `D405_rgb` 分辨率一致的：

```text
camera_matrix
dist_coeffs
```

可选方案：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A：用 D405 SDK intrinsics | 从 RealSense 当前 RGB stream 读 `fx/fy/ppx/ppy/distortion` | 最贴合硬件 | 当前 pkl 没保存，需要另写小工具查询一次 |
| B：用标定 episode 自己 `calibrateCamera()` | 像示例项目一样从棋盘格图片估计内参 | 不依赖 SDK 内参导出 | 需要足够多且质量好的棋盘格视角 |
| C：先用近似内参 | 临时用 RealSense 常见值或旧标定值 | 快速跑通 | 不推荐用于最终外参 |

推荐：先用 B 跑通流程；后续把 D405 RGB intrinsics 和 depth scale 存进采集 metadata。

### 阶段 5：调用 hand-eye calibration
输入列表：

```text
R_gripper2base_list = [R_base_tcp_i]
t_gripper2base_list = [t_base_tcp_i]
R_target2cam_list = [R_camera_board_i]
t_target2cam_list = [t_camera_board_i]
```

输出：

```text
T_tcp_camera
```

保存结果建议：

```text
calibration_results/
  d405_T_tcp_camera.npy
  d405_T_tcp_camera.yaml
  selected_frames.json
  reprojection_report.json
  validation_report.md
```

### 阶段 6：验证外参
不能只看 `calibrateHandEye()` 返回矩阵，必须验证。

对每个有效帧计算：

```text
T_base_board_i = T_base_tcp_i @ T_tcp_camera @ T_camera_board_i
```

因为标定板固定不动，所以所有 `T_base_board_i` 应该接近同一个值。

检查指标：

```text
translation_std_mm
translation_max_error_mm
rotation_std_deg
rotation_max_error_deg
PnP reprojection_error_px
```

还要做直观 sanity check：

- `t_tcp_camera` 是否接近 D405 在末端上的物理安装位置；
- 相机 z 轴方向是否符合镜头朝向；
- 用结果把棋盘格 3D 坐标轴投影回图片，是否和图像一致；
- 留出几帧不参与标定，只做验证。

## 和路径残差任务的关系

标定完成后，路径规划可按以下方式做：

1. 规划时刻从 D405 depth 反投影局部点云：

```text
P_camera = depth_to_point_cloud(D405_depth, camera_intrinsics, depth_scale)
```

2. 在相机坐标系下生成路径点和法向：

```text
p_i^camera
n_i^camera
```

3. 用规划时刻 TCP 位姿转换到 base：

```text
p_i^base = T_base_tcp_plan @ T_tcp_camera @ p_i^camera
n_i^base = R_base_tcp_plan @ R_tcp_camera @ n_i^camera
```

4. 每帧训练输入的几何部分：

```text
p_tcp_base = ee_pos_rotvec[:3]
R_base_tcp = Rodrigues(ee_pos_rotvec[3:])
x_axis_base = R_base_tcp[:, 0]
y_axis_base = R_base_tcp[:, 1]
z_axis_base = R_base_tcp[:, 2]

k = argmin_i ||p_i^base - p_tcp_base||
delta_p_seq = p_{k:k+K+1}^base - p_tcp_base
z_desired_seq = -n_{k:k+K+1}^base
```

用户已决定第一版不显式输入姿态误差，因此几何输入可为：

```text
p_tcp_base
x_axis_base
y_axis_base
z_axis_base
delta_p_seq
z_desired_seq
```

## 推荐实现方案

### 方案 A：复用当前 pkl，离线标定
从标定 episode 里读 pkl，检测 RGB 标定板，读取同帧 `ee_pos_rotvec`，求 `T_tcp_camera`。

优点：
- 和当前采集链路一致；
- 不需要单独写实时拍照/机械臂控制程序；
- pkl 中已有 `meta`，可过滤缓存帧和运动帧。

缺点：
- 如果采集时机器人一直在动，图像和 pose 仍有时间差；
- 当前 pkl 没保存 D405 intrinsics/depth_scale，需要额外获取或用棋盘格估计。

结论：推荐作为第一版。

### 方案 B：单独写按键采样脚本
像示例项目一样，每个姿态按键保存一张 RGB 和一个 TCP pose。

优点：
- 数据更干净；
- 每个姿态可以稳定静止；
- 更接近手眼标定经典流程。

缺点：
- 需要新脚本；
- 和当前 pkl 数据链路分开，后续还要同步维护。

结论：如果方案 A 因运动/缓存导致验证误差大，再做。

### 方案 C：自动控制机械臂采样
自动生成多组姿态，moveJ/moveL 到位后拍照。

优点：
- 姿态覆盖可控；
- 可重复。

缺点：
- 安全风险和实现复杂度最高；
- 需要工作空间、碰撞、速度限制、标定板视野约束。

结论：当前不推荐第一版做。

## 第一版文件规划
后续如果实现，建议新增：

```text
scripts/hand_eye_from_pkl.py
```

职责：
- 输入 `--episode-dir`
- 输入棋盘格/Charuco 参数；
- 读取 pkl；
- 筛选 `frame_new=True`、速度低、检测成功帧；
- 求 `camera_matrix/dist_coeffs` 或加载已有内参；
- 求每帧 `T_camera_board`；
- 调用 `cv2.calibrateHandEye()`；
- 输出 `T_tcp_camera` 和验证报告。

建议先不要改 `gello/` 业务代码。

## 风险清单
| 风险 | 后果 | 规避 |
|------|------|------|
| 使用缓存帧 | 图像和 TCP pose 错配 | 只用 `frame_new=True` |
| 机器人运动中取帧 | rolling/time skew 造成外参偏差 | 手动停稳或按 `joint_velocities` 过滤 |
| 棋盘格尺寸/内角点数写错 | PnP 尺度和位姿全错 | 实测标定板并写入配置 |
| 米和毫米混用 | 平移外参尺度错 1000 倍 | 全程用米，或全程用毫米，不混用 |
| OpenCV 方向搞反 | `T_tcp_camera` 反向或旋转错误 | 按本 note 的 `T_base_tcp` + `T_camera_board` 传参，并验证 `T_base_board_i` 一致性 |
| TCP 定义不是真实探头安装坐标 | 外参相对于错误 TCP | 确认 UR 控制器 TCP 设置 |
| D405 intrinsics 不匹配分辨率 | PnP 偏差 | 使用和 `D405_rgb` 相同分辨率的内参 |

## 验收标准
- 至少 15 个有效标定帧，推荐 20-40 个。
- 所有选中帧：
  - `frame_new=True`
  - 棋盘格/Charuco/AprilTag 检测成功
  - TCP pose 存在且单位明确
- 输出 `T_tcp_camera` 4x4 矩阵。
- 验证报告中：
  - `T_base_board_i` 平移散布在可接受范围内；
  - 旋转散布在可接受范围内；
  - 重投影误差合理；
  - 矩阵方向通过可视化/物理安装 sanity check。

## 当前不做的事
- 不直接修改业务采集代码。
- 不自动控制机械臂。
- 不从普通扫查数据里强行标定；必须是标定板固定且可见的 episode。
- 不把 depth 当作 hand-eye 主输入。depth 后续用于路径点云和验证。
