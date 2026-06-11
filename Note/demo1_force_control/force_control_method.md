# demo1 MainWindow 力控方法

## 结论

`demo1` 当前实际启用的是：

```text
混合位置-力控制，核心是 Z 方向主动恒力导纳控制。
```

更具体地说：

```text
XY 方向：主要做位置/路径跟踪，不响应力，防止横向漂移。
Z 方向：根据目标按压力和实际力的误差，通过导纳模型修正目标高度。
姿态：基本跟随参考轨迹或 BO 给出的姿态，不在当前主循环里做力矩导纳。
```

因此它不是纯阻抗控制，也不是简单力 PID，而是：

```text
hybrid position-force admittance control
路径位置跟踪 + Z 方向恒力导纳
```

## 控制入口

当前实际生效的控制函数是：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2422
```

```cpp
void MainWindow::robotControlLoop()
```

启动时打印：

```cpp
">>> Robot Admittance Control Thread Started (Real-Error Admittance Mode). <<<"
```

说明当前版本作者也把它定义为：

```text
Real-Error Admittance Mode
```

## 控制周期

代码设置：

```cpp
const auto period = std::chrono::milliseconds(8);
const double dt = 0.008;
```

因此控制频率约为：

```text
125 Hz
```

## 力数据链路

### 1. 读取力传感器

`forceSensorLoop()` 读取：

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

写入：

```cpp
currentForceData_
```

### 2. 重力补偿

在 `robotControlLoop()` 中：

```cpp
Eigen::Matrix<double, 6, 1> pure_data =
    compensator.getCompensatedData(raw_f, raw_t, pose_vec);
```

得到：

```cpp
Eigen::Vector3d F_sensor_pure = pure_data.head(3);
Eigen::Vector3d T_sensor_pure = pure_data.tail(3);
```

含义：

```text
F_sensor_pure: 传感器坐标系下去重力后的纯力
T_sensor_pure: 传感器坐标系下去重力后的纯力矩
```

### 3. 转到 Base 坐标系

函数：

```cpp
transformForceToBase(currentFlangePos, F_sensor_pure)
```

计算：

\[
F_{base}=R_{base}^{tcp}F_{sensor}
\]

代码位置：

```text
/home/ubuntu22/dev/demo1/src/mainwindow.cpp:2407-2417
```

## 安全阈值

代码中设置：

```cpp
const double MAX_FORCE = 40.0;
const double MAX_TORQUE = 1.5;
```

当前实际触发条件只启用了力阈值：

```cpp
if (F_sensor_pure.cwiseAbs().maxCoeff() > MAX_FORCE)
```

力矩阈值被注释掉了。

触发后：

```text
1. 停止机器人控制线程。
2. 停止扫描。
3. 停止轨迹 timer。
4. 当前 TCP 沿世界 Z 方向上抬 10 cm。
5. 用 MoveL 执行紧急上抬。
```

## 目标力

代码设置：

```cpp
const double TARGET_FORCE_Z = 3.0;
const double FORCE_GAIN_Z = 10.0;
```

注释写的是：

```text
目标恒定按压力，约 3~4 N。
```

Z 方向力误差：

```cpp
double force_err_z = F_base.z() - TARGET_FORCE_Z;
```

驱动力：

```cpp
F_drive.z() = force_err_z * FORCE_GAIN_Z * final_gain;
```

也就是：

\[
F_{drive,z}
=
(F_{base,z}-F_{target,z})K_fK_{timeout}
\]

其中 `final_gain` 会根据连续超时次数增大，最多放大到 5 倍。

## 导纳模型

代码核心：

```cpp
double a_i = (F_i - B * v_i - K * x_i) / M;
double delta_p = v_i * dt + 0.5 * a_i * dt * dt;
```

对应导纳方程：

\[
M\ddot{x}+B\dot{x}+Kx=F
\]

或者：

\[
\ddot{x}
=
\frac{F-B\dot{x}-Kx}{M}
\]

代码中的变量对应关系：

```text
M: 虚拟质量
B: 虚拟阻尼
K: 虚拟刚度
v_i: 当前 TCP 速度
x_i: 参考工具位姿和当前工具位姿的位置误差
F_i: 力驱动项
delta_p: 本周期导纳产生的位置增量
```

## XY 方向控制

XY 方向代码明确不响应力：

```cpp
F_drive.x() = 0.0;
F_drive.y() = 0.0;
```

XY 参数：

```cpp
const double M_xy = 0.5;
const double B_xy = 40.0;
const double K_xy = 500.0;
```

含义：

```text
XY 方向主要用于路径位置跟踪。
较大的 K_xy 和 B_xy 用来防止横向漂移。
```

虽然循环里对 i=0,1 也计算了导纳公式，但代码只在 `i == 2` 时把 `delta_p` 加到目标位姿：

```cpp
if (i == 2) {
    finalToolTarget[2] += delta_p;
}
```

因此当前实际位置修正只作用在 Z 方向。

## Z 方向控制

Z 方向根据是否接触切换两套参数。

接触判断：

```cpp
bool in_contact = (std::abs(F_base.z()) > 0.5);
```

空气中参数：

```cpp
const double M_z_air = 0.5;
const double B_z_air = 10.0;
const double K_z_air = 20.0;
```

接触后参数：

```cpp
const double M_z_cont = 0.1;
const double B_z_cont = 30.0;
const double K_z_cont = 400.0;
```

代码注释说明：

```text
接触后 Z 向阻尼用于抑制震荡。
接触后 Z 向刚度约 4N 对应 1cm 位移。
```

正常扫描时：

```cpp
finalToolTarget[2] += delta_p;
```

因此实际控制效果是：

```text
如果测得力和目标力有误差，Z 方向目标位置会被导纳模型上抬或下压。
```

## Z 偏移限幅

Z 方向目标相对参考轨迹有限幅：

```cpp
const double Z_OFFSET_MIN = -0.02;
const double Z_OFFSET_MAX = 0.05;
```

也就是：

```text
最多向下 2 cm
最多向上 5 cm
```

代码：

```cpp
double z_offset = finalToolTarget[2] - refToolPos[2];
if (z_offset < Z_OFFSET_MIN) finalToolTarget[2] = refToolPos[2] + Z_OFFSET_MIN;
if (z_offset > Z_OFFSET_MAX) finalToolTarget[2] = refToolPos[2] + Z_OFFSET_MAX;
```

## BO 模式下的特殊逻辑

如果正在 Bayesian Optimization：

```cpp
if (isOptimizing_ && boTargetPoseReady_.load())
```

则：

```text
XY 和姿态参考来自 BO 命令位姿。
Z 不走正常导纳，而是只做轻微力保护补偿。
```

代码：

```cpp
if (std::abs(F_base.z()) > 8.0) {
    finalToolTarget[2] += 0.0005;
}
else if (std::abs(F_base.z()) < 0.3) {
    finalToolTarget[2] -= 0.0003;
}
```

含义：

```text
力太大：上抬 0.5 mm
失去接触：下探 0.3 mm
```

## 控制输出

最终目标是工具位姿：

```cpp
RobotVector6 finalToolTarget = refToolPos;
```

导纳只修改：

```cpp
finalToolTarget[2]
```

然后转换回法兰：

```cpp
RobotVector6 targetFlangePos = toolToFlange(finalToolTarget);
```

再入队：

```cpp
robotCmdQueue_.push({ targetFlangePos, 1.0, 0.3, 0.4, dt, true });
```

最终由队列调用：

```cpp
s_urRobot_->MoveSoftPose(...)
```

## 和阻抗控制的区别

阻抗控制通常是：

```text
输入位置误差，输出力/力矩。
```

导纳控制通常是：

```text
输入外力，输出位置/速度修正。
```

当前 demo1 的代码是：

```text
输入：力误差 F_base.z - TARGET_FORCE_Z
输出：Z 方向目标位置增量 delta_p
```

所以它符合导纳控制。

## 一句话总结

`demo1` 的 MainWindow 力控方法是：

```text
以规划路径为参考，XY 保持高刚度位置跟踪；
Z 方向用重力补偿后的力反馈做恒力导纳控制；
最终通过 MoveSoftPose 持续发送修正后的 TCP/法兰目标位姿。
```

