# 任务计划：GELLO 数据时间对齐审计与实现规划

## 目标
全面理解 `gello_software` 仓库的一方源代码，建立数据采集链路的时间语义模型，并规划一套可验证的数据时间对齐方案，服务后续模仿学习。

## 当前阶段
阶段 6：实现与验证

## 范围
- 范围内：项目根目录、`gello/`、`experiments/`、`scripts/`、`tests/`、`configs/`、`ros2/src/` 下的一方 Python、C++、ROS 2 launch/config、shell、YAML/XML/config 源文件。
- 当前实现范围：用户确认不使用 ROS，因此后续代码实现只考虑 Python pkl 采集链路；ROS 相关阅读只作为仓库背景，不进入本轮实现。
- 边界：`third_party/`、二进制资源、日志、cache、生成元数据不作为一方业务实现逐行审计；但如果一方代码通过接口依赖它们，需要记录接口边界。
- 前一轮不修改业务代码，只做理解、头脑风暴、风险分析和实现计划；用户随后明确要求实现，因此本轮进入 Python pkl 采集链路代码修改。

## 阶段

### 阶段 1：需求与全仓库源码阅读
- [x] 阅读指定技能：`using-superpowers`、`brainstorming`、`karpathy-guidelines`、`planning-with-files`
- [x] 检查历史规划与会话上下文
- [x] 盘点一方源文件范围
- [x] 读取主 README、ROS 2 README、`Note.md` 和既有低维数据采集计划
- [x] 分批完整读取所有一方源文件
- [x] 记录所有与时间、频率、缓存、阻塞、保存、obs/action 配对有关的发现
- **状态：** 已完成

### 阶段 2：数据流与时间流建模
- [x] 识别数据生产者：robot、agent、camera、force sensor、keyboard、ZMQ node；ROS node 已阅读但不纳入当前实现范围
- [x] 识别数据消费者：控制循环、保存接口、转换脚本、检查脚本、测试
- [x] 标出每条链路的时间源、采样频率、阻塞点、缓存语义、队列语义
- [x] 明确 observation、action、control、next observation 的配对关系
- **状态：** 已完成

### 阶段 3：对齐风险与需求定义
- [x] 把“时间对齐”转成可验证的不变量
- [x] 明确主时间轴候选：robot state、action sent、frame saved、camera frame stamp
- [x] 标出当前代码可能造成错位的数据字段和原因
- [x] 提出 2-3 种对齐方案并比较取舍
- **状态：** 已完成

### 阶段 4：实现计划
- [x] 规划最小业务代码改动：时间戳、元数据、缓存帧 freshness、保存 schema、离线检查
- [x] 规划测试：合成时间戳、错位样例、正确对齐样例、schema 兼容
- [x] 规划硬件短采集验证：5-10 秒频率和 skew 报告
- [x] 规划对旧 `.pkl` 或现有下游脚本的兼容路径
- **状态：** 已完成

### 阶段 5：自审与交付
- [x] 检查假设是否显式
- [x] 检查是否存在“帧”“动作”“同步”等歧义词未定义
- [x] 检查每个成功标准是否可测量
- [x] 向用户汇报分析结论并等待批准后再改业务代码
- **状态：** 已完成

### 阶段 6：实现与验证
- [x] `save_frame()` 支持可选 `meta`，并保持旧顶层字段和 `control`
- [x] `RobotEnv` 记录 `last_obs_meta` 和 `last_step_timing`
- [x] `run_control_loop()` 保存 `obs_t -> action_t`，并在 action 发送后落盘同一条样本的 timing
- [x] D405、Orbbec、Ultrasound、force sensor 暴露 `last_metadata`
- [x] `chack_data.py` 新增 episode 级 `summarize_episode()`
- [x] 新增 time-alignment schema 测试
- [x] 运行 `PYTHONPATH=. pytest tests`
- [x] 运行 `PYTHONPATH=. python -m py_compile ...`
- [x] 运行 `git diff --check`
- **状态：** 已完成

## 关键问题
1. 样本语义问题：一个模仿学习样本应定义为 `obs_t -> action_t`，还是 `obs_t + action_t -> obs_{t+1}`？这与 `control` 应锚定为“由当前 obs 计算并即将发送的动作”，还是“导致下一帧状态变化的动作”是同一个问题。
2. 主时间轴问题：在已经确定样本语义后，离线诊断和过滤应以 `robot_state_time`、`action_sent_time`、`frame_save_time` 还是 camera frame time 作为参考时间。
3. 对训练来说可接受的 skew 是多少：毫秒级、一个控制周期内，还是只要求最近过去帧？
4. 未来数据集主形态是低维为主、图像为主，还是多模态同时训练？

## 已做决定
| 决定 | 理由 |
|------|------|
| 初始轮次不修改业务代码 | 用户要求先全面理解、brainstorm 和规划；用户确认后才进入实现。 |
| 使用 `Note/time_alignment/` 存放本轮规划 | 用户要求模仿现有 `Note/low_dim_data/` 的路径与格式。 |
| 一方源码优先，第三方代码作为接口边界 | `third_party/` 体量很大，直接逐行审计会稀释对采集时间语义的判断。 |
| 对齐问题必须用显式时间戳和离线诊断验证 | 仅凭同一个 `.pkl` 或同一轮循环不能证明多模态严格同步。 |
| 当前不实现 ROS 数据链路 | 用户确认没有使用 ROS，本轮只修 Python pkl 采集路径。 |
| 当前训练样本语义固定为 `obs_t -> action_t` | `run_control_loop()` 用当前 obs 算 action，并在 action 发送后把同一对 obs/action 连同 timing 保存。 |

## 遇到的问题
| 问题 | 尝试 | 处理 |
|------|------|------|
| 曾按根目录读取 `data_collection_*`，实际不存在 | 1 | 已记录路径错误，改为读取 `Note/low_dim_data/...`。 |
| 曾按 `Note/data_collection_*` 读取，实际路径多一层 `low_dim_data` | 2 | 已列出 `Note/` 并确认真实路径。 |
| 初始按 planning-with-files 默认在根目录创建规划文件 | 1 | 根据用户最新要求迁移到 `Note/time_alignment/`，删除根目录副本。 |

## 备注
- 当前一方源码范围初步统计为 139 个文件、约 13,801 行，排除了 `.git`、`third_party`、`Log`、cache、二进制资源。
- 既有 `Note/low_dim_data/` 计划已经指出：当前关键风险包括相机缓存帧、force 同步读取、缺少各模态时间戳、`save_frame()` 的保存语义、`control` 与真实发送命令的关系。
- 后续计划必须和既有低维数据采集计划兼容，不重复制造平行 schema。

## 代码改动落点规划

### 不属于本轮实现的改动
- `experiments/run_env.py`
  - 当前 diff 是用户把注释掉的 D405、Orbbec、Ultrasound 相机恢复到 `camera_clients`。
  - 这不是时间对齐实现的一部分，不应在本轮回滚或重写。
  - 后续最多只读取其中的 `hz`、camera name 和 save interface 使用方式，不把它作为主要修改点。
- `ros2/`
  - 用户确认当前没有使用 ROS，本轮不修改 ROS 代码。

### 第一层：保存 schema 和样本语义
- `gello/data_utils/format_obs.py`
  - 修改 `save_frame()` 的参数，允许接收可选 `meta`。
  - 继续保存现有顶层 obs 字段和 `control`，保证旧训练脚本能继续读。
  - 新增 `frame["meta"] = meta`，其中写明 `sample_semantics: "obs_t_to_action_t"`。
  - 不在这里重新解释 `control`，只负责忠实落盘。
- `gello/utils/control_utils.py`
  - 修改 `SaveInterface.update()`，接收可选 `meta` 并传给 `save_frame()`。
  - 在 `run_control_loop()` 中定义样本编号 `sample_index`。
  - 在 `agent.act(obs)` 前后记录 `action_compute_start/end`。
  - 保存时明确使用当前 `obs` 和裁剪后的 `action`，即 `obs_t -> action_t`。

### 第二层：主时间轴和 action 发送时间
- `gello/env.py`
  - 在 `RobotEnv.step()` 中记录 `action_send_start/end`，即调用 `robot.command_joint_state(joints)` 前后。
  - `step()` 当前返回 next obs；这个行为不改，只把 command timing 暴露给上层。
  - 建议返回值保持兼容：可以让 `RobotEnv` 维护 `last_step_timing` 属性，而不是马上改 `step()` 返回结构，降低破坏面。
- `gello/utils/control_utils.py`
  - 从 `env.last_step_timing` 或等价结构取出上一轮 action send timing，写入下一次保存所需的 timing。
  - 如果实现时发现这会让语义复杂，优先选择在 `env.step(action, return_timing=True)` 加可选参数，默认保持旧行为。

### 第三层：obs 内各模态读数时间
- `gello/env.py`
  - 在 `get_obs()` 中记录 `obs_read_start/end`。
  - 对每个 camera 记录 `read_start/end`、是否拿到新帧、是否使用缓存帧。
  - 对 robot observation 记录 `robot_read_start/end`。
  - 对 force sensor 记录 `force_read_start/end`、`force_valid`、`force_error`。
  - 建议同样用 `self.last_obs_meta` 暴露元数据，保持 `get_obs()` 仍返回原来的 obs dict。

### 第四层：相机 freshness
- `gello/cameras/D405.py`
  - 保持 `read()` 返回 `(rgb, depth)` 不变。
  - 新增 `self.last_metadata`，记录 `frame_new`、`read_start/end`、`cache_age_ms`、如果 SDK 可取则记录 frame number/hardware timestamp。
  - `wait_for_frames(10)` 超时时继续返回缓存，但必须把 `frame_new=False` 写入 metadata。
- `gello/cameras/Orbbec.py`
  - 同 D405，重点记录 FULL_FRAME_REQUIRE 成功拿到的新 frameset，或者超时返回缓存。
  - 如果 pyorbbecsdk 能取 timestamp/frame index，则写入 metadata；取不到也要明确为 `None`。
- `gello/cameras/Ultrasound.py`
  - 保持 `read()` 返回 `(rgb, dummy_depth)`。
  - 记录 OpenCV `cap.read()` 是否成功、失败时 `valid=False`，不要只用全 0 图像表达失败。
  - 可设置 `CAP_PROP_BUFFERSIZE=1`，但这属于采集行为变更，建议作为第二步并用短采集验证。

### 第五层：force 有效性
- `gello/force_sensor_mtcp.py`
  - 保持 `read_values()` 兼容返回 list 或 None。
  - 新增 `self.last_metadata`，记录 `read_start/end`、`valid`、`error`。
  - 上层 `gello/env.py` 可继续在失败时填零，但必须在 `meta` 里写 `force_valid=False`，避免训练把通信失败误认为真实 0 力。

### 第六层：离线检查和转换
- `chack_data.py`
  - 保留单帧可视化能力。
  - 新增 episode 目录模式：按 pkl 文件名排序，统计 sample interval、action_send_time 间隔、camera cache ratio、force invalid ratio、各模态相对主时间轴的 skew。
- `gello/data_utils/demo_to_gdict.py` 和 `gello/data_utils/conversion_utils.py`
  - 不作为第一步强改。
  - 等新 pkl schema 验证后，再让 converter 读取或保留 `meta`，并支持当前 `D405_rgb/Orbbec_rgb/Ultrasound_rgb` 字段。

### 第七层：测试
- `tests/test_env_lowdim.py`
  - 扩展 fake env/robot/camera/force，验证 `get_obs()` 不改变原 obs 字段，同时产生 timing metadata。
- 新增 `tests/test_time_alignment_schema.py`
  - 验证 `save_frame()` 写出的 pkl 仍包含旧字段和 `control`。
  - 验证新增 `meta.schema_version`、`sample_semantics`、`sample_index` 和 timing 字段。
  - 验证 legacy pkl 无 `meta` 时检查脚本不崩溃。

## 实现计划

### 1. 定义统一数据 schema
- 新增顶层 `meta` 字段，不改变现有图像、lowdim、`control` 的顶层字段，降低下游破坏面。
- `meta` 的含义：每个 `.pkl` 样本的“说明书/旁路元数据”，记录这个样本是什么时候采的、各模态是否有效、是否使用缓存帧、动作何时计算和发送。它不是新的训练输入本体，默认训练仍读原来的 obs 字段和 `control`；`meta` 用于检查、过滤、调试和必要时构造更严格的数据集。
- `meta` 至少包含：
  - `schema_version: "time_alignment_v1"`
  - `episode_id`
  - `sample_index`
  - `wall_time_iso`
  - `sample_mono_ns`
  - `control_loop_hz_config`
  - `sample_semantics: "obs_t_to_action_t"`
- `meta["timing"]` 包含：
  - `obs_read_start_mono_ns`
  - `obs_read_end_mono_ns`
  - `agent_act_start_mono_ns`
  - `agent_act_end_mono_ns`
  - `action_send_start_mono_ns`
  - `action_send_end_mono_ns`
  - `step_return_obs_start/end` 或 `next_obs_read_start/end`（如果选择保存 next obs）
- `meta["modalities"]` 按模态保存 read timing、valid、error、hardware stamp、cache age。

#### `meta` 示例
`meta` 里有数据，但不是图像、深度、力、关节这种训练观测本体；它是小体量的元数据，主要是时间、有效性和来源说明。示例：

```python
meta = {
    "schema_version": "time_alignment_v1",
    "episode_id": "0319_153159",
    "sample_index": 42,
    "sample_semantics": "obs_t_to_action_t",
    "wall_time_iso": "2026-03-19T15:32:01.397344",
    "sample_mono_ns": 123456789000000,
    "control_loop_hz_config": 100,
    "timing": {
        "obs_read_start_mono_ns": 123456780000000,
        "obs_read_end_mono_ns": 123456786000000,
        "agent_act_start_mono_ns": 123456786100000,
        "agent_act_end_mono_ns": 123456787000000,
        "action_send_start_mono_ns": 123456787200000,
        "action_send_end_mono_ns": 123456787600000,
    },
    "modalities": {
        "robot": {
            "read_start_mono_ns": 123456784000000,
            "read_end_mono_ns": 123456785000000,
            "valid": True,
        },
        "force": {
            "read_start_mono_ns": 123456785100000,
            "read_end_mono_ns": 123456786000000,
            "valid": True,
            "error": None,
        },
        "D405": {
            "read_start_mono_ns": 123456780000000,
            "read_end_mono_ns": 123456782000000,
            "valid": True,
            "frame_new": False,
            "frame_id": 1007,
            "hardware_timestamp_ms": 89123.4,
            "cache_age_ms": 18.7,
        },
    },
}
```

默认训练仍使用原来的顶层字段，例如 `joint_positions`、`D405_rgb`、`force`、`control`。`meta` 用于训练前检查和过滤，例如丢掉 force invalid、camera cache age 太大或 skew 超阈值的样本。

### 1.1 100Hz 机械臂与 30Hz D405 的频率策略
- 事实：100Hz 控制周期是 10ms，D405 30Hz 图像周期约 33.33ms，二者不是整数整除。即使改成 90Hz/30Hz，也仍有 USB、SDK、系统调度和相机硬件时钟 jitter，所以不能只靠整数倍关系证明对齐。
- 策略 A：保持机械臂/控制循环 100Hz，相机保持 30Hz。
  - 每个控制样本保存最新一帧图像，同时在 `meta["modalities"]["D405"]` 标记 `frame_new`、`frame_id`、`cache_age_ms`。
  - 结果会出现同一张图像连续对应 3 或 4 个 action，这是正常现象，训练/诊断时可以看见。
  - 优点是低维控制最平滑；缺点是图像策略训练时需要处理重复图像。
- 策略 B：把控制循环改成 30Hz、60Hz 或 90Hz。
  - 30Hz：最接近“一次控制对应一帧新图”，适合图像主导策略，但机器人动作更新变慢。
  - 60Hz：每张图大约对应两次控制，折中。
  - 90Hz：理论上每张图约对应三次控制，比 100Hz 更规整，但仍需 timestamp 验证。
  - 在当前代码里，控制频率主要通过 `RobotEnv(control_rate_hz=args.hz)` 控制；`run_env.py` 有 `--hz` 参数，D405 fps 在 `D405.py` 的 `enable_stream(..., 30)` 中配置。相机 fps 是否能改到其他值取决于硬件支持的 stream profile。
- 策略 C：采集保持原频率，离线重采样。
  - 对 100Hz 低维/action 数据保留完整时间戳。
  - 训练图像策略时只选择 `frame_new=True` 的样本，或以 camera frame time 为 anchor 选最近的 robot/action。
  - 训练低维策略时可以忽略图像重复，直接用 100Hz lowdim/action。
- 推荐顺序：
  1. 先实现 `meta` 和 episode 级诊断，真实测出 D405 frame_new 比例、cache_age、action skew。
  2. 如果训练是低维为主，保持 100Hz。
  3. 如果训练是图像为主，优先试 30Hz 或离线只取新图帧。
  4. 如果想兼顾控制平滑和图像规整，再试 60Hz 或 90Hz，并用诊断报告验证。

### 2. 最小代码改动点
- `gello/env.py`
  - 将 `Rate` 内部时间源从 `time.time()` 改成 `time.monotonic()` 或新增 monotonic 诊断，不改变外部 API。
  - 在 `get_obs()` 内采集每个 camera、robot、force 的 read_start/read_end。
  - 对 force 失败保存 `force_valid=False` 和错误信息，不再只靠全 0 表达失败。
- `gello/cameras/D405.py`、`Orbbec.py`、`Ultrasound.py`
  - 保持 `read()` 兼容返回 `(image, depth)`。
  - 新增可选 `read_with_metadata()` 或对象属性 `last_metadata`，记录 `frame_new`、`cache_age_ms`、`frame_id`、硬件时间戳（如果 SDK 可取）。
  - 超时返回缓存时必须显式标记，不阻塞 100 Hz 控制循环。
- `gello/utils/control_utils.py`
  - 在 `run_control_loop()` 中记录 `agent.act()` 前后时间。
  - 在调用 `save_interface.update(obs, action)` 时传入 sample/action timing。
  - 在 `env.step(action)` 内或外部记录 command send 前后时间。
- `gello/data_utils/format_obs.py`
  - 保存 `meta`，继续保存现有顶层字段和 `control`。
  - 保持旧 pickle 能被读取，新 pickle 能被 `chack_data.py` 和 converter 识别。
- `chack_data.py`
  - 从单帧检查升级为可选 episode 检查：统计采样频率、相邻间隔、每个模态 skew、缓存帧比例、invalid force 比例。
- `gello/data_utils/demo_to_gdict.py` / `conversion_utils.py`
  - 支持当前字段名 `D405/Orbbec/Ultrasound`。
  - 保留或导出 timing/skew，用于训练前过滤。

### 3. 测试计划
- 单元测试：fake camera 返回新帧/缓存帧，验证 `frame_new`、`frame_age_ms`。
- 单元测试：fake force 返回 None，验证 `force_valid=False` 且数值占位不会被误认为真实有效。
- 单元测试：fake robot/agent/env 验证 `obs_t -> action_t` 的 sample 语义和 `control` 保存时机。
- 离线诊断测试：构造一个包含故意错位 timestamps 的临时 episode，检查脚本必须报告超阈值 skew。
- 兼容测试：旧 pickle 没有 `meta` 时，检查脚本应提示 legacy schema，而不是崩溃。

### 4. 硬件验证计划
- 采集 5-10 秒短 episode。
- 打印和保存：
  - 控制循环实际 Hz、p50/p95/p99 周期；
  - robot/force/camera 各自 read latency；
  - camera 新帧比例、缓存 age p50/p95/p99；
  - 各模态相对 `action_send_start` 的 skew；
  - invalid force 或 camera failure 计数。
- 验收标准：
  - 所有 sample 有 `meta.schema_version`、`sample_index`、`control`、robot lowdim 和 timing 字段；
  - robot/action skew 可解释并稳定；
  - 30 FPS 图像在 100 Hz 控制循环下允许重复缓存，但每次重复必须可见；
  - force 通信失败不能静默伪装成真实 0 力；
  - converter 不再因字段名不匹配而无法处理当前数据。

### 5. 与既有规划的关系
- 本计划继承 `Note/low_dim_data/` 中阶段 4-7 的目标。
- 如果继续实现，只修改 Python pkl 采集路径；ROS 2 路径本轮不做。
