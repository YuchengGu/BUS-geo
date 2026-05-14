# 任务计划：迁移 demo1 的路径规划部分

## 目标

这个任务只负责像 `hand_eye_calibration/` 一样，在 `gello_software` 下新建一个独立 Python 文件夹，只实现路径规划本身。

它参考 `/home/ubuntu22/dev/demo1/src` 的逻辑步骤，但不照搬 C++ 代码，因为 demo1 是 Windows/C++/Qt/PCL/VTK 工程，而当前项目是 Ubuntu/Python 采集工程。

建议实现文件夹名：

```text
breast_path_planning/
```

这个文件夹的职责只到：

```text
D405 当前 RGB-D 图像 / 相机帧
    -> base 坐标系点云
    -> 乳腺分割
    -> 路径规划
    -> planned_path.json
```

不负责 GUI，不负责 robot 控制，不负责 episode 记录，不负责训练转换。

最终主要得到一个路径文件：

```text
planned_path[i] = {
    position_base,
    normal_base
}
```

每帧样本里的残差特征由采集/GUI 侧读取 `planned_path.json` 后计算，不作为 `breast_path_planning/` 的核心输出。

## 当前阶段

阶段 1：只做规划，不改业务代码。

## 范围

- 范围内：
  - `/home/ubuntu22/dev/demo1/CMakeLists.txt`
  - `/home/ubuntu22/dev/demo1/src/CommonStructs.h`
  - `/home/ubuntu22/dev/demo1/src/CoordinationTransform.*`
  - `/home/ubuntu22/dev/demo1/src/3DEqualDistance.*`
  - `/home/ubuntu22/dev/demo1/src/GeodesicPath.*`
  - `/home/ubuntu22/dev/demo1/src/PointCloudProcessor.*`
  - `/home/ubuntu22/dev/demo1/src/image_acquisition.*`
  - `/home/ubuntu22/dev/demo1/src/mainwindow.*` 中拍照、分割、规划、路径可视化相关部分
  - 当前 gello 的手眼标定结果、D405 实时 RGB/depth 数据
- 范围外：
  - demo1 的自动力控、贝叶斯优化、Plus/NDI 重建、超声视频校正。
  - 完整 GUI 的按钮和窗口实现不在第一阶段。
  - 摆位、拍照、规划确认、启动 GELLO 示教和路径 residual 保存属于可视化引导采集 GUI 子任务，见 `Note/visual_guided_collection_gui/`。
  - 每帧 pkl 的实时保存和 `obs_t -> action_t` 控制循环。
  - 训练代码。

## demo1 中要迁移的核心链路

```text
相机 RGB-D / 点云
    -> T_Base_Camera 转 base 坐标系
    -> 点击 seed / 区域生长分割乳腺
    -> MLS 平滑
    -> 法向估计
    -> adaptive slice
    -> serpentine path
    -> 可选 normal constrain
    -> smoothPathWindow
    -> planned_path(position, normal)
```

术语解释：

- `adaptive slice`：按乳腺点云真实边界一层一层切片，自适应生成每一行路径点。
- `serpentine path`：蛇形路径顺序，第一行左到右、第二行右到左，减少空走。
- `normal constrain`：可选步骤，用于限制法向不要偏得过大，让探头姿态更稳定；第一版默认关闭，因为乳腺侧面真实法向可能超过 30 度。
- `geodesic resample/测地线优化`：demo1 里更高级的沿曲面等距优化步骤；第一版先不要，避免复杂度过高。
- `tangent`：路径前进方向，demo1 用它生成机器人姿态；本任务第一版不用。
- `reverse_y`：demo1 里配合 tangent/serpentine 做探头局部 Y 轴翻转的姿态标志；不是本任务路径文件必需字段。

第一版预期会生成比较“板正”的参考路径：一行一行的规则蛇形扫描线贴在分割后的乳腺表面上。它不是最终最优轨迹，而是稳定、可解释、便于计算残差的参考路径；模仿学习再学习人在这条参考路径上的局部修正。

## 独立文件夹定位

和 `hand_eye_calibration/` 类似，`breast_path_planning/` 应该是一个独立工具文件夹：

```text
breast_path_planning/
    README.md
    plan_from_frame.py
    path_io.py
    pointcloud_from_d405.py
    segmentation.py
    surface_processing.py
    path_planner.py
    visualize_path.py
    demo1_notes.md
    examples/
    results/
```

建议每个文件职责：

| 文件 | 职责 |
|---|---|
| `plan_from_frame.py` | 主入口；只接收 RealSense 当前 `color_frame/depth_frame` 和 `T_base_camera`，不要求用户输入相机内参。 |
| `path_io.py` | 读写 `planned_path.json`，做 schema 校验。 |
| `pointcloud_from_d405.py` | 用 librealsense pointcloud 生成 base 点云，D405 内参由 SDK 内部使用。 |
| `segmentation.py` | 复刻 demo1 的 seed + HSV/空间邻域区域生长思想。 |
| `surface_processing.py` | 点云滤波、平滑、法向估计。 |
| `path_planner.py` | adaptive slice、serpentine，输出位置和法向；normal constrain 作为可选参数。 |
| `visualize_path.py` | 离线显示点云、路径和法向，便于审查。 |
| `demo1_notes.md` | 记录每一步来自 demo1 哪些函数，不让迁移依据丢失。 |

## 最小可行策略

第一版不要直接把 demo1 C++ 全量编进 Python。原因是 demo1 的 CMake 依赖 Qt、PCL、VTK、OpenCV、OrbbecSDK、ITK、NLopt、PlusLib、自定义 `.lib`，且大量路径是 Windows 风格。

推荐顺序：

1. **先写独立路径规划文件夹骨架。**
   - 只实现路径规划相关工具。
   - 主输入来自可视化界面刚拍到/冻结的 D405 RGB/depth 图像。
   - 不保留手动输入 `camera_matrix` 的离线反投影入口，避免后续误用错误内参。
   - 输出固定为 `planned_path.json`。

2. **按 demo1 逻辑步骤逐步实现。**
   - 代码肯定不同，但步骤要对应 demo1。
   - 优先迁移 zigzag/adaptive slice 主链路。
   - 第一版跳过 geodesic resample/测地线优化。
   - radial/spiral 后续再做。

3. **采集/GUI 只调用这个工具或读取它的输出。**
   - GUI 不内嵌路径规划细节。
   - 采集循环只消费 `planned_path.json` 来算残差。

## `planned_path.json` schema

```python
{
    "schema_version": "planned_path_v1",
    "source": "demo1_or_python_planner",
    "frame": "base",
    "points": [
        {
            "index": 0,
            "position_base": [0.0, 0.0, 0.0],
            "normal_base": [0.0, 0.0, 1.0]
        }
    ],
    "metadata": {
        "T_tcp_camera_source": "hand_eye_calibration/results_0512_222937_calib_11x8_stride10/T_tcp_camera.npy",
        "generated_from": "D405 RGB-D",
        "units": "meters"
    }
}
```

## 探头 Z 轴延长的边界

这个不是本文件夹的核心功能，只是路径文件和后续残差计算必须知道“真实接触点是 probe tip，不是 UR TCP”。

因此在 `breast_path_planning/` 中只需要：

```text
planned_path.json 使用 base 坐标系下的乳腺表面接触点
metadata 记录生成路径时使用的 T_base_camera / T_tcp_camera
```

真正每帧的 probe tip 计算放在采集侧：

```text
probe_tip_position_base = tcp_position_base + probe_tip_offset_m * tcp_z_axis_base
probe_x_axis_base = tcp_x_axis_base
probe_y_axis_base = tcp_y_axis_base
probe_z_axis_base = tcp_z_axis_base
```

注意：

- demo1 里出现过 `0.217 m` 和 `0.223 m` 两个值，后续实现必须配置化。
- 正负号要用实物确认。
- 训练残差使用 `probe_tip_position_base`，不是 `tcp_position_base`。

## 每帧路径残差的边界

每帧路径残差不是 `breast_path_planning/` 的核心输出，但这个文件夹应提供一个纯函数供 GUI/采集调用：

采集时给定当前探头尖端 `p_probe`，在路径上找最近点 `k`：

```text
k = argmin_i ||planned_path[i].position_base - p_probe||
```

保存：

```text
path_nearest_index
path_progress = k / (N - 1)
path_residuals_base[j] = planned_path[k+j].position_base - p_probe
path_normals_base[j] = planned_path[k+j].normal_base
path_lookahead_mask[j]
```

正常扫查时建议使用单调窗口搜索，避免最近点来回跳：

```text
search_window = [last_k - 3, last_k + 7]
```

默认采集特征建议使用：

```text
lookahead = 8
backtrack = 3
forward = 7
```

其中 `lookahead=8` 表示每帧保存从当前最近点 `k` 到 `k+7` 的 8 个 residual/normal；`backtrack=3, forward=7` 表示最近点只在上一帧附近的 `-3` 到 `+7` 窗口里搜索。

## 实现步骤

### 阶段 1：新建独立路径规划文件夹
- [ ] 新建 `breast_path_planning/`。
- [ ] 写中文 `README.md`，说明它像 `hand_eye_calibration/` 一样是独立工具。
- [ ] 写 `demo1_notes.md`，逐步映射 demo1 的 C++ 函数到 Python 模块。
- [ ] 写 `path_io.py` 和 `planned_path_v1` 示例。

### 阶段 2：从当前 D405 图像生成 base 点云
- [ ] 从可视化界面传入当前冻结的 RealSense `color_frame`、`depth_frame`。
- [ ] 同时传入拍照时刻的 `T_base_tcp`。
- [ ] 读取 `T_tcp_camera`；D405 内参不由用户输入，由 librealsense pointcloud 内部使用。
- [ ] 计算 `T_base_camera = T_base_tcp @ T_tcp_camera`。
- [ ] 优先用 librealsense `pointcloud` 自动生成相机坐标系点云。
- [ ] 再把 SDK 点云乘 `T_base_camera` 转成 base 点云。
- [ ] 输出 `raw_cloud_base.ply`。

这里传入拍照时刻的 `T_base_tcp`，不是为了根据机器人位姿规划动作，而是因为 D405 是手腕相机。librealsense pointcloud 得到的是相机坐标系下的 `P_camera`，路径规划和后续残差需要 base 坐标系下的点：

```text
P_base = T_base_tcp @ T_tcp_camera @ P_camera
```

因此 `T_base_tcp` 只承担坐标变换作用，确保 `planned_path.json` 里的 `position_base/normal_base` 和后续采集时的探头位置处于同一个 UR base 坐标系。

如果单独开终端规划，不需要启动 `run_env.py` 或 GELLO，但需要能读到 UR 当前 TCP 位姿。当前仓库里对应代码在：

```text
gello/zmq_core/robot_node.py
```

示例：

```python
from gello.zmq_core.robot_node import ZMQClientRobot
from breast_path_planning.geometry import rotvec_pose_to_transform

robot = ZMQClientRobot(port=6001, host="127.0.0.1")
obs = robot.get_observations()
T_base_tcp = rotvec_pose_to_transform(obs["ee_pos_rotvec"])
T_base_camera = T_base_tcp @ T_tcp_camera
```

`PathPlannerParams` 在：

```text
breast_path_planning/path_planner.py
```

默认：

```python
PathPlannerParams(max_normal_angle_deg=None)
```

表示不启用法向角度限制。只有显式设置 `max_normal_angle_deg=30.0` 时，才会调用 `constrain_normals_to_reference()`。

### 阶段 3：分割乳腺
- [ ] 参考 demo1 的 seed + HSV + 空间邻域区域生长逻辑。
- [ ] 第一版可以用独立规划脚本或可视化界面点击选 seed。
- [ ] 输出 `segmented_breast.ply`。
- [ ] 保存分割配置和 seed。

### 阶段 4：路径规划
- [ ] 实现法向估计、adaptive slice、serpentine。
- [ ] 保留 normal constrain 作为可选参数，默认不启用。
- [ ] 先实现 demo1 zigzag 主链路。
- [ ] 暂不实现 geodesic resample/测地线优化。
- [ ] 输出 `planned_path.json`。
- [ ] 离线可视化检查点云、路径和法向。

### 阶段 5：给 GUI/采集提供接口
- [ ] 提供加载路径和计算最近点/残差的纯函数。
- [ ] GUI 只调用接口或读取 `planned_path.json`。
- [ ] 采集侧再把 residual 写入 pkl。

## 成功标准

- `breast_path_planning/` 可以像 `hand_eye_calibration/` 一样独立运行。
- 从当前 D405 RGB-D 图像、手眼外参和拍照时刻 `T_base_tcp` 出发，能输出 base 坐标系点云。
- 完成 seed 分割后，能输出 `segmented_breast.ply`。
- 完成规划后，能输出 `planned_path.json`。
- `planned_path.json` 可以被 GUI 和采集循环读取。
- 代码逻辑步骤对应 demo1，但实现方式是 Python/Ubuntu，不依赖 Windows C++ 工程。
