# 进展记录：曲面路径约束的 GELLO/UR5 末端遥操作

## 2026-05-19

- 明确用户目标：不是让 UR5 逐关节跟随 GELLO，而是让 GELLO 作为末端输入设备，控制 UR5 TCP 沿曲面蛇形路径前进。
- 明确路径构造思路：

```text
蛇形路径只保留拐点
  -> 每两个拐点之间生成曲面曲线
  -> 多段曲线拼接成一个完整采集任务
```

- 比较了主曲率线、测地线和论文扩散方程：
  - 第一版推荐测地线 / kNN graph shortest path。
  - 主曲率线只作为方向参考。
  - 扩散方程可作为后续增强。
- 明确微调方案：

```text
p_tip_des = Gamma(S) + y_offset * b(S) + h_offset * n(S)
```

- 明确 GELLO 行程不足方案：
  - 使用增量控制，不使用绝对位移。
  - 增加 clutch/recenter。
- 明确 `gello_get_offset.py` 在新模式中不再承担 UR5-GELLO 关节对齐职责。
- 创建任务文件夹：

```text
Note/surface_path_cartesian_teleop/
```

- 创建文件：

```text
task_plan.md
findings.md
progress.md
design.md
```

## 下一步

建议先做离线实现和验证：

```text
planned_path.json + segmented_breast.ply
  -> 拐点抽取
  -> graph geodesic segments
  -> global Gamma(S)
  -> Darboux frame visualization
```

确认曲线和 frame 合理后，再接 GELLO 增量输入和 UR5 `servoL`。

## 2026-05-19 记号修正

- 将综合 Markdown 中的 Darboux frame 记号从 `t,v,n` 统一改为 `t,n,b`。
- 采用：

```text
t: 路径切向
n: 表面外法向
b = t x n: 曲面内横向副法向
D = [t,n,b]: Darboux TNB frame
R_tcp_base = [t,b,-n]: TCP +z axis points opposite to outward normal
```

- 补充解释了 `h_contact`、`p_start`、探头延伸长度和自动接近起点流程。
- 根据用户确认，探头尖端沿 UR5 TCP +z 轴延伸，且 TCP +z 与曲面外法向相反；文档已改为先计算探头尖端目标 `p_tip_des`，再用 `p_tcp_des = p_tip_des + probe_length * n` 反推 UR5 TCP 目标。
