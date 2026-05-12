# 进度日志

## 会话：2026-05-12

### 阶段 1：需求与全仓库源码阅读
- **状态：** 已完成
- **开始时间：** 2026-05-12
- 已执行操作：
  - 阅读指定技能：`using-superpowers`、`brainstorming`、`karpathy-guidelines`、`planning-with-files`。
  - 运行 planning session catchup；没有发现根目录可恢复的 active planning session。
  - 检查 `git status --short`；创建本轮规划文件前工作区干净。
  - 盘点仓库文件，确认 `third_party/` 体量很大，应作为 vendored 边界处理。
  - 统计一方源文件范围：139 个源/配置文件、约 13,801 行。
  - 阅读 `README.md`、`ros2/README.md`、`Note.md`。
  - 读取既有 `Note/low_dim_data/` 三个规划文件，确认它们已经覆盖 UR5 低维状态修正，下一阶段是时间戳和频率测量。
  - 阅读 `experiments/run_env.py`、`experiments/launch_yaml.py`、`gello/env.py`。
  - 阅读主控制与保存链路：`gello/utils/control_utils.py`、`gello/data_utils/format_obs.py`。
  - 阅读 robot/ZMQ 基础链路：`gello/robots/robot.py`、`gello/robots/ur.py`、`gello/zmq_core/robot_node.py`、`gello/zmq_core/camera_node.py`。
  - 阅读 camera/force 链路：`gello/cameras/camera.py`、`D405.py`、`Orbbec.py`、`Ultrasound.py`、`realsense_camera.py`、`gello/force_sensor_mtcp.py`。
  - 阅读 agent 和部分 robot 后端：`gello/agents/*`、`gello/robots/dynamixel.py`、`yam.py`、`panda.py`、`sim_robot.py`、`xarm_robot.py`、`robotiq_gripper.py`、`gello/dynamixel/driver.py`。
  - 阅读 data_utils、检查脚本和测试：`demo_to_gdict.py`、`conversion_utils.py`、`keyboard_interface.py`、`plot_utils.py`、`chack_data.py`、`tests/test_ur_lowdim.py`、`tests/test_env_lowdim.py`。
  - 阅读 launch、脚本、configs、ROS 2 publisher/controller/gripper manager、FACTR 和 dm_control 仿真任务代码。
  - 将方案比较、推荐 schema、实现计划、测试计划和硬件验证计划写入 note 文件。
  - 根据用户要求，改用 `Note/time_alignment/` 路径和中文 note 格式保存本轮规划。
- 创建/修改文件：
  - `Note/time_alignment/data_time_alignment_task_plan.md`
  - `Note/time_alignment/data_time_alignment_findings.md`
  - `Note/time_alignment/data_time_alignment_progress.md`
  - 删除了本轮误建的根目录 `task_plan.md`、`findings.md`、`progress.md`

### 阶段 2：数据流与时间流建模
- **状态：** 已完成
- 已执行操作：
  - 建立控制循环语义：保存发生在 `env.step(action)` 之前，当前 pickle 是 `obs_t -> action_t`。
  - 建立模态读取顺序：camera -> robot -> force。
  - 标出 ZMQ、后台线程、相机缓存、force 同步读取、ROS 2 topic queue 的时间风险。
- 创建/修改文件：
  - `Note/time_alignment/data_time_alignment_findings.md`

### 阶段 3：对齐风险与需求定义
- **状态：** 已完成
- 已执行操作：
  - 比较 3 种对齐方案。
  - 推荐每模态独立 timestamp + 主时间轴 skew 诊断。
  - 定义首版训练样本语义为 `obs_t -> action_t`。
- 创建/修改文件：
  - `Note/time_alignment/data_time_alignment_findings.md`

### 阶段 4：实现计划
- **状态：** 已完成
- 已执行操作：
  - 写出 schema、最小代码改动点、测试计划、硬件短采集验证标准。
- 创建/修改文件：
  - `Note/time_alignment/data_time_alignment_task_plan.md`

### 阶段 5：自审与交付
- **状态：** 已完成
- 已执行操作：
  - 检查计划中是否明确了 frame/action/同步语义。
  - 检查验收标准是否可测量。
  - 根据用户澄清，确认 `experiments/run_env.py` 的 camera diff 是用户恢复相机，不属于时间对齐实现。
  - 将后续业务代码修改落点拆成七层：保存 schema、控制循环、环境 timing、相机 freshness、force 有效性、离线检查、测试。
- 创建/修改文件：
  - `Note/time_alignment/data_time_alignment_task_plan.md`

### 阶段 6：实现与验证
- **状态：** 已完成
- 已执行操作：
  - 新增 `tests/test_time_alignment_schema.py`，先确认缺失功能导致测试失败。
  - 修改 `gello/data_utils/format_obs.py`，让 `save_frame()` 支持可选 `meta`，保持旧字段兼容。
  - 修改 `gello/env.py`，让 `Rate` 使用 monotonic，并记录 `last_obs_meta`、`last_step_timing`。
  - 修改 `gello/utils/control_utils.py`，拆分键盘状态检查和落盘，在 action 发送后保存旧 `obs_t + action_t + meta`。
  - 修改 D405、Orbbec、Ultrasound 和 force sensor，添加 `last_metadata`。
  - 修改 `chack_data.py`，新增 `summarize_episode()`、摘要打印 helper 和目录路径 CLI 模式，同时避免 import 时触发 matplotlib。
- 创建/修改文件：
  - `tests/test_time_alignment_schema.py`
  - `gello/data_utils/format_obs.py`
  - `gello/env.py`
  - `gello/utils/control_utils.py`
  - `gello/cameras/D405.py`
  - `gello/cameras/Orbbec.py`
  - `gello/cameras/Ultrasound.py`
  - `gello/force_sensor_mtcp.py`
  - `chack_data.py`

## 测试结果
| 测试 | 输入 | 预期 | 实际 | 状态 |
|------|------|------|------|------|
| planning catchup | `session-catchup.py` | 如有历史上下文则输出 | 无输出 | 通过 |
| 初始工作区状态 | `git status --short` | 干净或仅用户已有改动 | 无输出 | 通过 |
| 一方源文件统计 | `find ... | wc -l` | 得到可审计文件规模 | 139 | 通过 |
| time alignment 红灯测试 | `PYTHONPATH=. pytest tests/test_time_alignment_schema.py tests/test_env_lowdim.py` | 新功能缺失导致失败 | 4 个预期失败：`meta` 参数、`last_obs_meta`、`last_step_timing`、`summarize_episode` 缺失 | 通过 |
| 目标测试 | `PYTHONPATH=. pytest tests/test_time_alignment_schema.py tests/test_env_lowdim.py` | 全部通过 | 6 passed | 通过 |
| 全部仓库测试 | `PYTHONPATH=. pytest tests` | 全部通过 | 9 passed | 通过 |
| 语法检查 | `PYTHONPATH=. python -m py_compile ...` | 无输出，退出码 0 | 无输出，退出码 0 | 通过 |
| diff 空白检查 | `git diff --check` | 无输出，退出码 0 | 无输出，退出码 0 | 通过 |

## 错误日志
| 时间戳 | 错误 | 尝试 | 处理 |
|--------|------|------|------|
| 2026-05-12 | 按根目录读取 `data_collection_*` 文件失败 | 1 | 记录为路径错误，继续精确列出 `Note/`。 |
| 2026-05-12 | 按 `Note/data_collection_*` 读取失败 | 2 | 确认真实路径为 `Note/low_dim_data/data_collection_*`。 |
| 2026-05-12 | 初始规划文件路径不符合用户偏好 | 1 | 迁移到 `Note/time_alignment/` 并删除根目录副本。 |

## 5 问恢复检查
| 问题 | 回答 |
|------|------|
| 我现在在哪？ | 阶段 5：准备向用户汇报并等待批准后再改业务代码。 |
| 接下来去哪？ | 用户确认样本语义和实现范围后，进入代码实现。 |
| 目标是什么？ | 规划可验证的数据时间对齐方案，服务模仿学习数据采集。 |
| 我学到了什么？ | 见 `Note/time_alignment/data_time_alignment_findings.md`。 |
| 我做了什么？ | 建立符合用户路径/格式偏好的 note 文件，并完成入口代码初读。 |
