# demo1 MainWindow 文件定位

## 结论

`demo1` 的主窗口代码在：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp
/home/ubuntu22/dev/demo1/src/mainwindow.h
/home/ubuntu22/dev/demo1/src/mainwindow.ui
/home/ubuntu22/dev/demo1/src/ui_mainwindow.h
```

其中和力控最相关的是：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp
/home/ubuntu22/dev/demo1/src/mainwindow.h
```

## 关键源码位置

### 主窗口实现

- `/home/ubuntu22/dev/demo1/src/mainwindow.cpp`
- 包含 GUI 回调、路径规划、扫描流程、力传感器启动、机器人控制线程、导纳控制循环。

### 主窗口声明

- `/home/ubuntu22/dev/demo1/src/mainwindow.h`
- 包含 `MainWindow` 类成员变量、力传感器对象、机器人控制线程、导纳状态、调试变量和机器人命令队列。

### UI 文件

- `/home/ubuntu22/dev/demo1/src/mainwindow.ui`
- Qt Designer UI 文件。

### 自动生成 UI 头文件

- `/home/ubuntu22/dev/demo1/src/ui_mainwindow.h`
- Qt 根据 `.ui` 生成的头文件，不建议手动改。

## 和力控直接相关的函数

### 力传感器线程

位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:509
```

函数：

```cpp
void MainWindow::forceSensorLoop()
```

作用：

```text
循环读取 6 维力/力矩数据，写入 currentForceData_。
```

对应成员变量在：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.h:247-256
```

包括：

```cpp
ForceSensorSerial* forceSensor_;
std::thread* forceSensorThread_;
std::atomic<bool> forceSensorRunning_;
std::vector<float> currentForceData_; // [Fx, Fy, Fz, Mx, My, Mz]
std::mutex forceDataMutex_;
```

### 力转 Base 坐标系

位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2407
```

函数：

```cpp
Eigen::Vector3d transformForceToBase(
    const RobotVector6& current_tcp_pose,
    const Eigen::Vector3d& f_sensor_pure
)
```

作用：

```text
用 UR 当前 TCP rotvec 得到 R_base_tcp，然后把传感器系力转换到 base 系：
F_base = R_base_tcp * F_sensor_pure
```

### 机器人力控主循环

位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2422
```

函数：

```cpp
void MainWindow::robotControlLoop()
```

这是当前实际启用的力控核心。

### 扫描启动

位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2888
```

逻辑包括：

```text
1. 清零导纳状态。
2. 连接力传感器 COM8。
3. 启动力传感器线程 forceSensorLoop。
4. 启动机器人控制线程 robotControlLoop。
5. 启动轨迹推进 timer。
```

### 扫描停止

位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2938
```

逻辑包括：

```text
1. 停止轨迹推进。
2. 停止 robotControlLoop。
3. 调用 UR Stop。
4. 停止力传感器线程。
5. 断开力传感器。
```

## 机器人命令发送方式

导纳循环不直接长期阻塞调用 `MoveL`，而是把命令放进队列：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2648
```

```cpp
robotCmdQueue_.push({ targetFlangePos, 1.0, 0.3, 0.4, dt, true });
```

队列消费位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2140-2146
```

如果 `isSoft == true`：

```cpp
s_urRobot_->MoveSoftPose(cmd.pose, cmd.vel, cmd.acc, cmd.blend, cmd.dt);
```

如果 `isSoft == false`：

```cpp
s_urRobot_->MoveL(cmd.pose, cmd.vel, cmd.acc);
```

队列结构在：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.h:429-435
```

```cpp
struct RobotCmd {
    RobotVector6 pose;
    double vel, acc, blend, dt;
    bool isSoft;
};
std::queue<RobotCmd> robotCmdQueue_;
std::mutex robotCmdMutex_;
```

