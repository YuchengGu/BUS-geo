# 乳腺表面路径规划

这个文件夹是一个独立路径规划工具，定位类似 `hand_eye_calibration/`：它只负责从当前 D405 RealSense 帧生成 `planned_path.json`，不负责 GUI、机器人控制、episode 记录或训练。

当前版本只保留**不需要手动输入相机内参**的入口：通过 librealsense 的 `rs.pointcloud()` 从当前 `color_frame/depth_frame` 直接生成 RGBXYZ 点云。RealSense SDK 会在内部使用 D405 自己的内参；本工具不再接收 `camera_matrix`，也不再提供手动 depth 反投影入口。

## 为什么还需要机器人位姿

这里需要“拍照时刻的机器人位姿”，不是为了根据机器人姿态规划动作，而是为了做坐标变换。D405 是手腕相机，SDK 生成的点云一开始在 D405 相机坐标系下；路径文件需要输出 UR base 坐标系下的路径点和法向：

```text
P_base = T_base_tcp @ T_tcp_camera @ P_camera
```

其中：

- `P_camera`：RealSense SDK 生成的 D405 相机坐标系点云，单位是米。
- `T_tcp_camera`：手眼标定结果，相机到 TCP。
- `T_base_tcp`：拍这张 RGB-D 图时，UR TCP 在 base 坐标系下的位姿。
- `P_base`：转换后的乳腺点云，后续规划路径和残差都用这个坐标系。

所以机器人位姿只用于把相机点云变成 base 点云，保证 `planned_path.json` 里的 `position_base/normal_base` 能和后续采集时的探头位置放在同一个坐标系里。

## 处理流程

```text
当前 D405 color_frame/depth_frame
  -> librealsense pointcloud 生成 D405 相机坐标系 RGBXYZ 点云
  -> 用 T_base_camera 转到 UR base 坐标系
  -> 根据 seed 做乳腺区域生长分割
  -> 估计表面法向
  -> adaptive slice 生成一行一行的路径点
  -> serpentine 连接成蛇形顺序
  -> 可选 normal constrain
  -> planned_path.json
```

第一版**不做** demo1 里的 geodesic resample / 测地线优化，也不输出 `tangent` 或 `reverse_y`。输出路径比较“板正”：它是稳定、可解释、便于计算残差的蛇形参考路径。

默认不会把法向限制在 30 度以内，因为乳腺侧面真实法向可能远大于 30 度。`normal constrain` 只是可选安全/姿态约束；不显式设置时，路径文件保留点云估计出的真实表面法向。

## 在线终端入口

正常使用时，先在一个终端启动 UR robot node：

```bash
python experiments/launch_nodes.py --robot ur
```

然后另开一个终端运行在线规划脚本：

```bash
python breast_path_planning/live_plan_from_d405.py
```

这个脚本会按顺序执行：

```text
连接 UR robot node
  -> 打开 D405
  -> 冻结一张当前 RGB-D frame
  -> 读取当前 ee_pos_rotvec
  -> 加载 T_tcp_camera.npy
  -> 计算 T_base_camera = T_base_tcp @ T_tcp_camera
  -> 生成 base 坐标系 RGBXYZ 点云
  -> 弹出 Open3D 点云窗口
  -> 在点云里选一个 seed
  -> 高亮显示分割到的乳腺点云
  -> 终端输入 y 确认，或 r 重选，或 q 取消
  -> 调用点云规划核心
  -> 保存规划结果
```

默认手眼文件为：

```text
hand_eye_calibration/results_0512_222937_calib_11x8_stride10/T_tcp_camera.npy
```

如需指定输出目录：

```bash
python breast_path_planning/live_plan_from_d405.py \
  --output-dir breast_path_planning/results/test_plan_001
```

输出文件：

```text
raw_cloud_base.ply
segmented_breast.ply
planned_path.json
planning_report.json
```

注意：这个脚本会自己打开 D405。如果 `run_env.py` 已经占用了 D405，可能会打开失败；推荐在正式采集 episode 前先单独运行它生成 `planned_path.json`。

## PLY 离线测试入口

3D 点云选点需要 Open3D：

```bash
python -c "import open3d as o3d; print(o3d.__version__)"
```

如果 `Newgello` 环境里没有：

```bash
pip install open3d
```

如果先不接真实机械臂，可以直接用已有 `.ply` 点云测试同一套分割和规划逻辑：

```bash
python breast_path_planning/plan_from_ply.py \
  --cloud-ply /path/to/raw_cloud_base.ply \
  --output-dir breast_path_planning/results/test_ply_001
```

这个入口会读取 PLY 的 XYZ 和 RGB，转换成内部统一的 `PointCloud(points_base, colors_rgb)`，然后走同一套：

```text
Open3D 点云选 seed
  -> 点云区域生长分割
  -> 高亮预览
  -> 确认后规划
  -> 保存 planned_path.json
```

要求 PLY 最好带 `red/green/blue` 顶点颜色，因为当前分割使用“空间邻域 + RGB/HSV 颜色相似”。真实 D405 在线脚本保存出来的 `raw_cloud_base.ply` 带颜色，因此可以再喂给 `plan_from_ply.py` 复查或重规划。外部 PLY 如果不是 UR base 坐标系，生成的 `planned_path.json` 也会处在该 PLY 原本的坐标系里。

## 内部核心函数

在线 D405 和离线 PLY 都会先转换成统一的 RGBXYZ 点云，然后调用点云规划核心：

```python
from breast_path_planning.plan_from_frame import plan_from_point_cloud

result = plan_from_point_cloud(
    raw_cloud=raw_cloud,
    seed_indices=[seed_index],
    output_dir="breast_path_planning/results/session_001",
)
```

这里没有 `camera_matrix` 参数。点云生成对应 librealsense 逻辑：

```text
pc.map_to(color_frame)
points = pc.calculate(depth_frame)
vertices = points.get_vertices()
```

## 输出格式

`planned_path.json` 只包含每个路径点的位置和法向：

```json
{
  "schema_version": "planned_path_v1",
  "frame": "base",
  "points": [
    {
      "index": 0,
      "position_base": [0.1, 0.2, 0.3],
      "normal_base": [0.0, 0.0, 1.0]
    }
  ]
}
```

后续可视化采集界面点击“规划路径”时，应调用这里的函数，然后加载生成的 `planned_path.json`。

## 文件说明

- `plan_from_frame.py`：点云分割、路径规划和保存核心；也保留 RealSense frame 到点云的函数入口。
- `live_plan_from_d405.py`：在线终端入口，连接 UR、打开 D405、在 Open3D 点云窗口中选 seed。
- `plan_from_ply.py`：离线 PLY 测试入口，读取已有点云并走同一套点云分割/规划。
- `interactive_pointcloud.py`：Open3D 点云选点、分割高亮预览和确认/重选。
- `pointcloud_from_d405.py`：RealSense SDK 点云生成、PLY 写出和 PLY 读取。
- `segmentation.py`：seed + 颜色/空间邻域区域生长分割。
- `surface_processing.py`：点云法向估计、法向方向约束。
- `path_planner.py`：adaptive slice + serpentine 路径生成。
- `path_io.py`：`planned_path.json` 的读写和校验。
- `visualize_path.py`：离线预览点云、路径和法向。
- `demo1_notes.md`：记录这些步骤对应 demo1 的哪些代码。

## 单独终端规划时怎么读 UR 位姿

如果不启动 `run_env.py`，也可以读取 UR 当前 TCP 位姿。先启动 robot node：

```bash
python experiments/launch_nodes.py --robot ur
```

然后规划脚本可以通过 ZMQ 读当前观测：

```python
from gello.zmq_core.robot_node import ZMQClientRobot
from breast_path_planning.geometry import rotvec_pose_to_transform

robot = ZMQClientRobot(port=6001, host="127.0.0.1")
obs = robot.get_observations()
T_base_tcp = rotvec_pose_to_transform(obs["ee_pos_rotvec"])
T_base_camera = T_base_tcp @ T_tcp_camera
```

这段代码用到的文件是：

- `gello/zmq_core/robot_node.py`：定义 `ZMQClientRobot`。
- `breast_path_planning/geometry.py`：定义 `rotvec_pose_to_transform()`。

这里不需要 GELLO，也不需要 `run_env.py`。但如果要输出 base 坐标系下的路径，必须能读到 `T_base_tcp`；否则只能在相机坐标系里预览点云和分割。

## 路径规划参数在哪里

路径参数在：

```text
breast_path_planning/path_planner.py
```

其中：

```python
PathPlannerParams(max_normal_angle_deg=None)
```

表示默认不启用法向角度限制，保留乳腺侧面真实法向。只有显式设置：

```python
PathPlannerParams(max_normal_angle_deg=30.0)
```

才会调用 `constrain_normals_to_reference()` 把法向限制在参考方向 30 度内。
