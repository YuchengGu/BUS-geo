# 代码发现：demo1 路径规划迁移

## CMake 证据

- `/home/ubuntu22/dev/demo1/CMakeLists.txt:12-13` 将 `src/*.cpp`、`src/*.h`、`src/*.ui`、`src/*.qrc` 全量编译进 Qt 程序。
- `/home/ubuntu22/dev/demo1/CMakeLists.txt:17-28` 依赖 Qt5 Widgets/Core/Gui/Network/SerialPort/SerialBus/Xml/Concurrent。
- `/home/ubuntu22/dev/demo1/CMakeLists.txt:38-74` 依赖 VTK、Boost、OrbbecSDK、OpenCV、PCL。
- `/home/ubuntu22/dev/demo1/CMakeLists.txt:76-105` 依赖 ITK、NLopt、PlusLib、Limbo、自定义 `.lib`。
- 多数依赖路径是 Windows 路径，所以第一步不应直接把 demo1 编译进当前 Ubuntu Python 项目。

## 路径点结构

- `/home/ubuntu22/dev/demo1/src/CommonStructs.h:14-19` 定义 `PathPointWithTangent`：
  - `position`
  - `normal`
  - `tangent`
  - `reverse_y`

其中本任务第一版只需要 `position` 和 `normal`。`tangent` 与 `reverse_y` 是 demo1 用来从路径点生成机器人姿态的附加信息，当前不作为 `planned_path.json` 必需字段。

## 点云与坐标转换

- `/home/ubuntu22/dev/demo1/src/CoordinationTransform.cpp:122-139` 用 `T_Base_Camera = T_Base_Flange * T_Flange_Camera`。
- `/home/ubuntu22/dev/demo1/src/image_acquisition.cpp:237-248` 生成点云时把相机点从 mm 转 m，再乘 `T_Base_Camera` 写入 base 坐标系。

当前 gello 已有 D405 手眼标定结果，等价公式应是：

```text
T_base_camera = T_base_tcp @ T_tcp_camera
P_base = T_base_camera @ P_camera
```

## 分割逻辑

- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:594-667` 分割入口先选择并合并 PLY 文件。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:690-720` 在 PCL viewer 里点选 seed。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:724-751` space 执行分割，S 保存，R 重置。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:757-835` 区域生长用 seed 平均 HSV 和空间半径。

## 路径规划主链路

- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1272-1284` 对分割点云做 MLS smoothing 和法向计算。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1306-1311` 调用 `planAdaptiveSlicePath()`。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1323-1333` demo1 还有 geodesic resample 后再生成 serpentine path；本任务第一版先跳过 geodesic resample/测地线优化，直接做规则蛇形参考路径。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1370-1383` 对 normal 做角度约束。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1384-1414` window smoothing。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:1416-1429` 计算 tangent 并转换 robot poses；这一步属于 demo1 的姿态生成，本任务第一版不迁移。

## tangent/reverse_y 与姿态

- `/home/ubuntu22/dev/demo1/src/3DEqualDistance.cpp:390-433` `computePathTangents()` 用全局参考方向 `(1,0,0)` 投影到切平面，得到稳定切向。
- `/home/ubuntu22/dev/demo1/src/CoordinationTransform.cpp:154-156` 目标探头 Z 轴为 `-normal`。
- `/home/ubuntu22/dev/demo1/src/CoordinationTransform.cpp:167-190` 切向投影后作为 Y 轴，X 轴用 `Y x Z`。

结论：

- `serpentine` 决定路径点的扫描顺序。
- `tangent` 决定探头绕法向的朝向。
- `reverse_y` 是 demo1 里配合 tangent 做局部 Y 轴翻转的姿态标志，不是路径点顺序本身。
- 当前只做模仿学习路径上下文时，`position_base + normal_base` 足够作为第一版输出。

## 探头偏移

- `/home/ubuntu22/dev/demo1/src/CoordinationTransform.cpp:107-119` 使用 `T_Flange_Probe(2,3)=0.217`。
- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp:4089-4109` 导出覆盖率时使用 `PROBE_OFFSET_M=0.223`。

这两个值不一致。后续实现时应作为配置项，不能散落硬编码。

## 当前 gello 可承接的点

- `gello/env.py` 已保存 `ee_pos_rotvec` 和 `tcp_x/y/z_axis_base`。
- `gello/utils/control_utils.py` 当前保存语义已经是 `obs_t -> action_t`。
- 已有 `meta` 可以记录路径文件、probe offset、lookahead K、最近点策略。

## 结论

路径规划迁移任务的本质是：

```text
demo1 PathPointWithTangent(position, normal)
    -> planned_path.json
    -> gello 每帧 probe tip + position/normal residual features
```

不需要第一步就迁移 demo1 的整个 GUI 或自动力控。
