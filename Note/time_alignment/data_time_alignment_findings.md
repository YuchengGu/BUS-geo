# 发现与决策

## 需求
- 先不要急着改业务代码。
- 使用 `using-superpowers`、`brainstorming`、`karpathy-guidelines`、`planning-with-files`。
- 尽可能全面读取仓库的一方源代码，而不是只看显然相关的文件。
- 核心目标是判断数据时间是否能对齐，以及如何实现可验证的对齐。
- 规划文件格式和路径要模仿现有 `Note/low_dim_data/` 下的 note 文件。
- 用户确认当前没有使用 ROS；因此 ROS 代码只作为仓库背景，不纳入本轮实现计划。

## 源码调研发现
- 仓库是 GELLO teleoperation 软件，包含 Python runtime、ROS 2 包、robot/camera/agent 抽象、ZMQ 节点、data_utils、测试、configs 和 vendored 第三方代码；当前实际采集实现聚焦 Python pkl 链路，不使用 ROS。
- 排除 `.git`、`third_party`、`Log`、cache 和二进制资源后，一方源文件约 139 个、13,801 行。
- `README.md` 明确数据采集入口：
  - YAM/YAML 推荐路径：`python experiments/launch_yaml.py --left-config-path ... --use-save-interface`
  - 非 YAM 路径：`python experiments/run_env.py --agent=gello --use-save-interface`
  - 后处理：`python gello/data_utils/demo_to_gdict.py --source-dir=<source_dir>`
- `ros2/README.md` 表明 ROS 2 路径主要用于 Franka FR3：GELLO publisher 发布 `/gello/joint_states`，joint impedance controller 订阅该 topic。
- 既有 `Note/low_dim_data/` 计划已经完成 UR5 真实低维状态修正，下一阶段正是时间戳与频率测量。

## 已读关键代码发现
- `experiments/run_env.py`
  - 默认 `hz=100`。
  - 非 mock 路径当前创建 force sensor 和 ZMQ robot client；相机字典目前为空，但注释里保留 D405/Orbbec/Ultrasound 构造路径。
  - 启动时会多次调用 `env.get_obs()`、`agent.act(obs)` 和 `env.step(...)` 做 reset/start 对齐。
  - 主循环交给 `gello.utils.control_utils.run_control_loop()`。
  - 对 7 维 GELLO action 到 6 维 UR5 robot joints 做了裁剪，这会影响 action/control 的语义追踪。
- `experiments/launch_yaml.py`
  - YAML 路径默认控制频率来自 config 的 `hz`，默认 30 Hz。
  - 机器人可能直接作为 hardware server 被包成 ZMQ server，然后再由 ZMQ client 控制。
  - 使用 `SaveInterface` 时数据目录是 config 上两级目录下的 `data`。
- `gello/env.py`
  - `Rate` 用 `time.time()` 控制周期，不是 monotonic。
  - `RobotEnv.step()` 顺序是：发送 joint command、sleep 到周期、再 `get_obs()`。
  - `RobotEnv.get_obs()` 顺序是：先读所有 camera，再读 robot observation，再读 force sensor。
  - force 读取失败时保存全 0，但当前没有记录失败原因或该帧是否有效。
- `gello/utils/control_utils.py`
  - `run_control_loop()` 当前循环是：用上一轮 `obs` 计算 `action`，可选保存 `(obs, action)`，然后调用 `env.step(action)` 得到下一轮 `obs`。
  - 因此保存文件里的 `control` 更接近 `obs_t -> action_t`，不是 `action_t -> obs_{t+1}`。如果训练需要 next-state，需要从相邻文件重新配对。
  - 保存的是裁剪后的 action；raw GELLO action 如果要用于调试 7 维到 6 维裁剪，需要另存。
- `gello/data_utils/format_obs.py`
  - 每帧 pickle 文件名使用 wall-clock `datetime.isoformat()`。
  - 文件内部只保存 `control` 和 obs 字段，没有保存 monotonic 时间戳、保存开始/结束时间、传感器 read 时间或 frame id。
- `gello/zmq_core/robot_node.py` 和 `gello/robots/sim_robot.py`
  - ZMQ robot 使用 REQ/REP 同步请求：每次 `get_observations()`、`command_joint_state()` 都包含序列化、socket 往返、server 执行和反序列化延迟。
  - ZMQ server 内部不附加 server-side 时间戳，因此 client 无法区分本机调用时间与真实 robot 采样时间。
- `gello/cameras/D405.py`
  - D405 配置为 848x480、30 FPS、RGB8 + Z16 depth。
  - `read()` 等待新 frameset 最多 10 ms，超时返回 `last_color/last_depth` 缓存，没有标记缓存 age 或是否新帧。
- `gello/cameras/Orbbec.py`
  - 尝试启用彩色/深度 FULL_FRAME_REQUIRE 和硬件 frame sync，并做深度到彩色空间对齐。
  - `read()` 同样等待最多 10 ms，超时返回缓存，没有暴露帧时间戳、frame id 或缓存新鲜度。
- `gello/cameras/Ultrasound.py`
  - 使用 OpenCV `VideoCapture.read()` 同步读帧。
  - 当前没有设置 FPS、buffer size 或读取时间元数据；失败时返回全 0 图像和全 0 depth，占位值与真实黑图不可区分。
- `gello/force_sensor_mtcp.py`
  - 使用 Modbus TCP 同步读取 12 个寄存器并转换成 6 维力/力矩。
  - 失败返回 `None`，上层 `RobotEnv` 转成全 0；缺少读前/读后时间、错误类型和有效性标记。
- `gello/agents/*`
  - `GelloAgent.act()` 直接读取 Dynamixel leader joint state，没有 action 生成时间。
  - `SpacemouseAgent` 有后台线程每 1 ms 读最新 state，`act()` 读取的是最近缓存；底层 state 有 `t` 字段注释但未保存到 action metadata。
  - `QuestAgent.act()` 从 `OculusReader` 拉 controller pose/button，没有记录 controller 数据时间戳。
- `gello/robots/*`
  - `URRobot.command_joint_state()` 使用 UR `servoJ`，内部 `dt=1/500` 并 `waitPeriod`；外层控制循环频率和 UR servo 内部周期是两套时间。
  - `XArmRobot` 有 50 Hz 后台 command thread，主控制循环设置的是 target command，不是立刻硬件执行；保存 action 时必须区分 target set time 与 hardware apply time。
  - `DynamixelDriver` 有 1 ms 后台读取线程，`get_joints()` 返回最近缓存；没有返回缓存采样时间。
  - `YAMRobot.get_observations()` 当前返回内部 `_joint_state`，而不是每次从硬件新读；如果先 `get_joint_state()` 再 obs 可能更新，否则可能陈旧。
- `gello/data_utils/demo_to_gdict.py` 和 `conversion_utils.py`
  - converter 按 pickle 文件名排序/反序，把每个 `.pkl` 变成一个 timestep。
  - 现有 `preproc_obs()` 硬编码读取 `wrist_rgb/base_rgb/wrist_depth/base_depth`，但当前多模态采集注释和 `chack_data.py` 使用的是 `D405_rgb/Orbbec_rgb/Ultrasound_rgb` 等字段。
  - converter 没有读取任何时间戳字段，也不会检查相邻帧间隔、action/obs skew 或缓存图像重复。
  - `get_act_min_max()` 中 `scale_max = np.maximum(scale_min, curr_scale_factor)` 看起来像笔误，应该用旧的 `scale_max`；这不直接影响时间对齐，但会影响 action normalization。
- `chack_data.py`
  - 目前是单个 pkl 的人工可视化/结构检查脚本，能打印 lowdim 和多模态图像。
  - 文件路径硬编码，且尚不具备 episode 级频率、时间差、缓存帧比例、传感器有效性统计。
- ROS 2 路径
  - `GelloPublisher` 以 25 Hz 发布 `sensor_msgs/JointState`，并设置 `header.stamp = self.get_clock().now()`。
  - `JointImpedanceController` 订阅 `gello/joint_states` 的队列深度是 1，并用 0.5 秒阈值检查 `last_joint_state_time_` 和 `msg.header.stamp` 是否过期。
  - controller update rate 在 `controllers.yaml` 中是 1000 Hz；GELLO publisher 25 Hz，因此控制器在高频循环中持有最近一次 GELLO position。
  - ROS 2 路径有基本过期保护，但不是面向离线模仿学习数据集的多模态对齐记录。
- FACTR 路径
  - `gravity_compensation.py` 主 gravity compensation loop 使用 config `controller.frequency`，常见为 500 Hz。
  - teleop follower loop 独立线程，默认 `teleop.hz=30`，读取 leader state、构建 follower action、平滑后 `teleop_env.step(action)`。
  - FACTR 同样有多线程、ZMQ、不同频率和 `time.time()` 控制周期；如果未来采集 FACTR 轨迹，也应使用同一套 timestamp schema。
- config 与频率
  - Python/YAML teleop 常见 `hz=30`；`run_env.py` 默认 `hz=100`；UR `servoJ` 内部 dt 为 2 ms；XArm 后台线程默认 50 Hz；FACTR 500 Hz；ROS controller 1000 Hz；ROS GELLO publisher 25 Hz；RealSense/Orbbec 常见 30 FPS。
  - 因此“同一帧”不能由循环次数自然保证，必须以时间戳和主时间轴定义为准。

## 初始对齐风险模型
| 风险 | 影响 |
|------|------|
| 控制循环用 `time.time()` 而不是 monotonic | 系统时间跳变可能影响周期统计和时间差计算。 |
| `step()` 返回的是 action 发送并 sleep 之后的 observation | 如果保存逻辑把当前 action 放入返回 obs，语义可能是 `action_t -> obs_{t+1}`，不是 `obs_t -> action_t`。 |
| `get_obs()` 内相机、robot、force 顺序读取 | 单个 obs 字典内部各模态真实采样时间不同。 |
| 相机可能低频或返回缓存 | 图像字段可能不是当前控制周期的新帧。 |
| ZMQ client/server 增加请求响应延迟 | robot state 和 command 的实际硬件时间可能晚于本进程调用时间。 |
| force 读取失败用全 0 | 训练可能把通信失败误学成真实 0 力。 |
| 后台线程缓存最新值但不记录采样时刻 | action/observation 看似同一循环，真实可能来自不同时间。 |
| `step()` 后返回 next obs，但保存发生在 step 前 | 下游如果误把同一文件当作 action 后状态，会造成标签错位。 |

## 技术决策
| 决策 | 理由 |
|------|------|
| 后续实现优先加元数据和诊断，不优先重写控制架构 | 符合 karpathy-guidelines 的外科手术式改动。 |
| 每个模态需要自己的 `read_start/read_end` 或硬件 stamp | 单个 sample 时间戳不足以证明多模态对齐。 |
| 保存 schema 需要显式定义 `obs_time` 和 `action_time` | 模仿学习训练标签必须知道 action 对应的是哪一刻的状态。 |
| 训练样本推荐先定义为 `obs_t -> action_t` | 当前保存发生在 `env.step(action)` 之前，顺应现有代码最小改动；next obs 可由相邻样本派生。 |
| 主时间轴推荐先用 `action_mono_ns` 或 `sample_mono_ns` | 示教标签是 action，且 robot/camera/force 都可计算到该时刻的 skew。 |

## 发现的问题
| 问题 | 处理方向 |
|------|----------|
| 当前路径风格应使用 `Note/<topic>/` | 已将本轮计划迁移到 `Note/time_alignment/`。 |
| 既有低维数据采集计划与本轮目标重叠 | 本轮应继承并细化其阶段 4-7，不重复造新 schema。 |
| 下游 converter 和当前采集 schema 不一致 | 需要新增当前 schema 转换路径，或让 `preproc_obs()` 支持 `D405/Orbbec/Ultrasound` 字段。 |
| 用户当前不使用 ROS | 本轮实现不进入 ROS 2 路径，只保留 Python pkl 采集链路的 timestamp/schema/diagnostic 计划。 |

## 对齐方案比较
| 方案 | 做法 | 优点 | 缺点 | 结论 |
|------|------|------|------|------|
| 方案 A：只用文件名 wall-clock 对齐 | 继续用 pickle 文件名的 ISO 时间，离线按文件排序 | 改动最少 | 无法知道各模态真实采样时刻；缓存帧和读取失败不可见 | 不推荐，只能用于粗略人工查看。 |
| 方案 B：单循环统一 timestamp | 每个 sample 加 `sample_mono_ns`，所有字段默认共享该时间 | 简单，兼容当前结构 | 仍然掩盖 camera/force/robot 顺序读取的 skew | 可作为过渡，但不足以证明多模态对齐。 |
| 方案 C：每个模态独立 timestamp + 主时间轴诊断 | 保存 action、robot、force、camera 的 read_start/read_end/hardware stamp/cache age/valid 标志，并离线报告 skew | 能定量判断是否对齐，适合模仿学习 | 需要较多元数据字段和检查脚本 | 推荐。最小改动可先从 Python pkl 路径实现。 |

## 推荐对齐定义
- 一个训练样本先定义为：`obs_t` 中的各模态读数，加上由该 `obs_t` 计算并实际发送的 `action_t/control_t`。
- 这同时回答 `control` 的锚点问题：当前 `control` 应解释为“发送给机器人前、由当前 obs 计算出来的动作”，而不是“已经造成下一帧状态变化的动作”。
- `obs_t + action_t -> obs_{t+1}` 不应塞进同一个当前样本文件里硬解释；如果训练或分析需要 next-state，应由离线 converter 用相邻样本显式构造。
- `action_t` 的时间语义：`action_compute_start/end` 记录 `agent.act(obs_t)` 时间，`action_send_start/end` 记录 `env.step(action_t)` 内部 command 时间。
- `obs_t` 的每个子模态单独记录：
  - `robot_read_start/end`，如有硬件 stamp 则保留；
  - `force_read_start/end`、`force_valid`、`force_error`；
  - 每个 camera 的 `read_start/end`、`frame_new`、`frame_age_ms`、`frame_id`、`hardware_timestamp`；
  - `sample_index`、`sample_mono_ns`、`wall_time_iso`。
- 离线诊断时默认以 `action_send_start` 作为主时间轴计算 skew；`robot_state_time`、`frame_save_time` 和 camera frame time 作为被比较的模态时间。若某些硬件拿不到真实 frame time，则使用 read_start/read_end 区间近似并明确标记。

## `meta` 的含义
- `meta` 是每个样本的旁路元数据，不是新的机器人观测值，也不是必须直接喂给策略网络的训练输入。
- 它用于回答“这条样本能不能信”：什么时候读 obs、什么时候算 action、什么时候发送 action、相机是不是新帧、force 是否有效、robot state 是否可能陈旧。
- `meta` 里确实有数据，但这些数据是小体量标量/字符串/布尔值，例如 timestamp、sample_index、frame_id、frame_new、cache_age_ms、valid、error；不是 RGB 图、depth、force 数值或关节状态本体。
- 保持现有顶层字段不变的原因是兼容旧脚本；新增 `meta` 后，训练脚本可以先忽略它，检查脚本和 converter 可以用它过滤错位样本。

## 100Hz 控制与 30Hz D405 的结论
- 100Hz 机械臂/控制循环和 30Hz D405 不是整数整除，因此同一张相机图像天然会对应多个控制样本，且重复次数可能是 3 或 4 个周期交替。
- 这不代表不能训练，但必须在数据里显式记录 `frame_new`、`frame_id`、`cache_age_ms` 和 action/obs timing，否则训练时无法区分新图和缓存图。
- 可调频率：
  - 控制循环可通过 `--hz` / `RobotEnv(control_rate_hz=...)` 改到 30、60、90 或 100Hz。
  - D405 fps 在 `D405.py` 的 `enable_stream(..., 30)` 设置中，但可用 fps 取决于相机分辨率和硬件支持，不能假设任意 fps 都可用。
- 推荐先不靠频率整数倍解决问题，而是先记录真实 timestamp 并跑短 episode 诊断；如果图像是主输入，再考虑 30Hz 控制或离线只取新图帧。

## 相关资源
- `Note/low_dim_data/data_collection_task_plan.md`
- `Note/low_dim_data/data_collection_findings.md`
- `Note/low_dim_data/data_collection_progress.md`
- `README.md`
- `ros2/README.md`
- `experiments/run_env.py`
- `experiments/launch_yaml.py`
- `gello/env.py`

## 视觉/浏览器发现
- 本轮没有使用浏览器或图像检查。

## 实现结果
- Python pkl 采集链路已实现 `time_alignment_v1` 元数据，ROS 路径未修改。
- `save_frame()` 现在支持可选 `meta`，但旧字段和 `control` 仍保持顶层平行结构。
- `RobotEnv.get_obs()` 仍只返回 obs dict，不把 `meta` 塞进 obs；对应元数据通过 `env.last_obs_meta` 暴露。
- `RobotEnv.step()` 仍返回 next obs；对应 action send timing 通过 `env.last_step_timing` 暴露。
- `run_control_loop()` 在 action 发送后保存旧的 `obs_t + action_t + meta`，因此保存文件语义仍是 `obs_t -> action_t`，同时能记录同一 action 的 send timing。
- D405、Orbbec、Ultrasound 和 force sensor 新增 `last_metadata`，用于标记 `frame_new`、`cache_age_ms`、`valid`、`error` 等状态。
- `chack_data.py` 新增 `summarize_episode()` 和目录路径 CLI 模式，可统计 legacy 帧数、schema 版本、action interval、camera cache ratio 和 invalid modality 计数。
- 验证结果：`PYTHONPATH=. pytest tests` 为 9 passed；`py_compile` 和 `git diff --check` 通过。
