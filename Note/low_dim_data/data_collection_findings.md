# 发现与决策

## 需求
- 暂不修改业务代码，先产出实现计划。
- 使用 `karpathy-guidelines`：显式写出假设，优先做简单、外科手术式改动，定义可验证的成功标准。
- 使用 `planning-with-files`：把实现计划和调研发现持久化到项目文件中。
- 后续要实现：
  - 真实 UR5 关节速度；
  - 真实 UR5 TCP 位置和姿态；
  - 清晰处理 `rx,ry,rz` 旋转向量；
  - 记录实际发送给 UR5 的控制命令；
  - 记录各模态时间戳；
  - 记录频率/速率诊断；
  - 为 30 Hz 图像与更高频的机器人/控制数据设计对齐策略。

## 源码调研发现
- `experiments/run_env.py` 默认目标频率是 `hz=100`，并在同一个进程里构造 D405、Orbbec、Ultrasound、力传感器和 ZMQ robot client。
- `RobotEnv.get_obs()` 当前顺序读取相机，然后读取机器人观测，然后读取力传感器。
- `RobotEnv.step()` 先发送机器人关节命令，按配置频率 sleep，然后再调用 `get_obs()`。
- `gello/robots/ur.py` 当前用 `getActualQ()` 读取 UR5 关节位置。
- `gello/robots/ur.py` 当前把 `joint_velocities` 设置成 `joints`，这对速度语义是错误的。
- `gello/robots/ur.py` 当前把 `ee_pos_quat` 设置成全 0，因此没有记录 TCP 位姿。
- `gello/robots/ur.py` 当前把 `gripper_position` 设置成 `joints[-1]`；对无夹爪 UR5 来说，这是第六关节角，不是夹爪。
- UR RTDE 可以通过 `getActualQd()` 提供真实关节速度，通过 `getActualTCPPose()` 提供 TCP pose。
- UR TCP pose 使用 `[x, y, z, rx, ry, rz]`，其中 xyz 单位是米，rx/ry/rz 是旋转向量，单位是弧度。
- D405 配置为 848x480、30 FPS、RGB8 + Z16 depth。
- D405 的 `read()` 最多等待 10 ms 获取新 frameset；超时则返回缓存的 `last_color/last_depth`。
- Orbbec 使用默认 profile，也最多等待 10 ms；超时则返回缓存帧。
- Ultrasound 使用 OpenCV `VideoCapture(camera_index=5)`，没有显式设置 FPS 或 buffer；实际频率依赖采集卡和 OpenCV 后端。
- `SaveInterface.update()` 使用 `datetime.datetime.now()` 给保存调用打时间，但没有在 pickle 内保存每个传感器自己的时间戳。
- `save_frame()` 会通过 `obs["control"] = action` 原地修改 `obs`。

## 技术决策
| 决策 | 理由 |
|------|------|
| 在 `URRobot.get_observations()` 中补齐真实状态读取 | 错误的 UR5 低维状态正是在这里产生的，这是最小改动点。 |
| 同时保存 UR 原始 TCP rotvec 和 quaternion pose | 原始 UR pose 可直接追溯到硬件；四元数方便机器学习和现有 `ee_pos_quat` 字段命名。 |
| 使用 `time.monotonic()` 作为对齐元数据 | monotonic 时间适合在一次采集 session 内计算时间差，不受系统时间跳变影响。 |
| 同时保留 wall-clock ISO 时间 | 文件名、日志和人工排查仍然方便。 |
| 只在需要时新增 raw GELLO action | 当前裁剪后的 `control` 已经是发送给 UR5 的命令；raw GELLO 更适合调试 7 维到 6 维裁剪，不是 UR5 命令真实性的必要条件。 |

## 发现的问题
| 问题 | 处理方向 |
|------|----------|
| 当前下游 converter 期望 `wrist_rgb/base_rgb`，但采集写的是 `D405_rgb/Orbbec_rgb/Ultrasound_rgb` | 计划中包含更新 converter 或新增当前 schema 的转换路径。 |
| 当前单个 `.pkl` 暗示同步，但实际没有严格同步 | 计划中包含各模态时间戳和缓存帧 age 元数据。 |
| 相机流比控制循环慢 | 计划中使用低维/action 流作为主时间轴，再用最近时间戳或最近过去帧对齐图像。 |
| 无夹爪 UR5 的 `gripper_position` 语义冲突 | 计划中包含明确的无夹爪表示，或加入 `has_gripper` 元数据。 |

## 相关资源
- `experiments/run_env.py`
- `gello/env.py`
- `gello/robots/ur.py`
- `gello/utils/control_utils.py`
- `gello/data_utils/format_obs.py`
- `gello/cameras/D405.py`
- `gello/cameras/Orbbec.py`
- `gello/cameras/Ultrasound.py`
- `gello/force_sensor_mtcp.py`
- `gello/data_utils/conversion_utils.py`

## 视觉/浏览器发现
- 本次计划任务没有使用浏览器或图像检查。
