---
title: "曲面乳腺机器人超声扫查实验设计方案"
subtitle: "测地线优化 - 贝叶斯姿态优化 - Darboux 示教采集"
author: "整理版"
date: "2026-06-12"
lang: zh-CN
mainfont: "FreeSerif"
CJKmainfont: "Noto Serif CJK SC"
geometry: margin=2cm
fontsize: 10.5pt
colorlinks: false
linkcolor: black
urlcolor: black
---

# 0. 总体主线

整篇文章不要写成三个孤立模块，而要围绕一个统一问题展开：

> 曲面几何 backbone 能否同时提升乳腺机器人超声自动扫查的稳定性与图像质量，并提升示教采集的平滑性与一致性？

你的三块方法应形成如下层级：

1. **测地线优化**：提供全局曲面扫描 backbone，解决路径在自由曲面上的扭转和姿态突变问题。
2. **贝叶斯优化**：在测地线 backbone 的局部点上优化探头姿态，提升图像质量，同时考虑力/力矩安全。
3. **Darboux/TNB 映射示教**：在同一个曲面 backbone 上，把人的操作映射到“沿路径前进、侧向微调、法向压迫”的局部任务坐标中，获得更平滑、更稳定、更一致的示教数据。

一句话主张：

> 测地线优化不是为了替代力控，而是为了给力控、局部成像优化和示教采集提供更好的曲面几何参考。

实验设计必须从一开始就按**多路径、多点位、多 trial**来组织，而不是先展示一条路径或一个点，再在最后补一句“我们也做了很多”。具体来说：测地线优化的样本单位是一条完整路径，贝叶斯优化的样本单位是一个局部优化点位，真实扫查的样本单位是一次完整 robot scan trial，示教采集的样本单位是一个用户在一条路径上的一次任务。

| 模块 | 样本单位 | 推荐样本量 | 统计方式 |
|---|---|---:|---|
| 测地线优化 | 一个 phantom/ROI 上的一条路径 | 15-25 条路径 | 每条路径先计算 summary metrics，再跨路径统计 |
| 真实机械臂扫查 | 一次完整扫描 trial | 50-75 次 trial | 每次 trial 计算力/力矩/图像指标，再跨 trial 统计 |
| 贝叶斯优化 | 一个扫描点位上的局部姿态优化任务 | 120-150 个点位 | 每个点位做配对比较，再跨点位统计 |
| Darboux 示教 | 一个用户在一条路径上的一次任务 | 100 次以上 trial | 按 user/path/method 建模，比较平滑性和一致性 |

# 1. 实验总览

建议整篇实验分成六大组。每一组实验都要从设计阶段就包含多个对象、多个路径、多个点位或多个用户，而不是只展示一个代表性例子。

| 实验组 | 核心问题 | 要证明的结论 | 样本设计 |
|---|---|---|---|
| A. 测地线优化过程与几何验证 | 能量函数是否真的收敛？路径点如何移动？ | 测地线优化在多条路径上降低曲面内扭转和姿态突变，同时保持覆盖 | 多 phantom、多 ROI、多路径 |
| B. 真实机械臂扫查验证 | 加力控后，测地线效果会不会被覆盖？ | 在相同法向力控下，测地线路径降低力控负担、切向拖拽和力矩波动 | 多路径、多重复 trial |
| C. 图像质量指标验证 | 自定义 Q 分数是否有意义？ | Q 与人工评价一致，能代表图像可用性 | 多 phantom、多姿态、多质量等级图像 |
| D. 贝叶斯优化验证 | BO 是否真的比其他搜索好？惩罚项是否必要？ | BO 在多点位上更省样本；force/torque 惩罚项提高安全性 | 多区域、多点位配对实验 |
| E. 示教映射验证 | Darboux/TNB 映射是否真的更好操作？ | 示教轨迹更平滑、力更稳定、跨人一致性更好 | 多用户、多路径、多任务 |
| F. 附加实验 | 系统是否鲁棒、安全、医学相关？ | target visibility、噪声鲁棒性、安全性、失败案例分析 | 多扰动、多 target、多失败类型 |

推荐的总体规模如下：

| 项目 | 最低可接受 | 比较漂亮 |
|---|---:|---:|
| breast phantom 数量 | 3 个 | 5-8 个 |
| 每个 phantom 的 ROI/路径 | 3 条 | 4-5 条 |
| 测地线路径总数 | 9-15 条 | 20 条以上 |
| 每种真实扫查方法重复 | 5 次 | 8-10 次 |
| BO 测试点位 | 90-120 个 | 150 个以上 |
| 示教用户 | 8-10 人 | 12-15 人 |
| 示教路径 | 2-3 条 | 3-5 条 |
| 图像盲评者 | 2 人 | 3 人，最好有超声经验 |

# 2. A 组实验：测地线优化过程与几何验证

本组实验不要只围绕某一条路径展开。实验单位应定义为：

> 一个 phantom 的一个 ROI 上生成的一条完整扫描路径。

例如，准备 5 个 phantom，每个 phantom 选 4 个 ROI，则共有 20 条路径：

$$
\Gamma_1,\Gamma_2,\ldots,\Gamma_{20}.
$$

每一条路径都生成 initial serpentine、spline/NURBS 和 geodesic-optimized 三种版本，然后分别计算几何指标。主文只展示少量代表性路径，统计图必须包含全部路径。

## 2.1 多路径能量函数迭代过程可视化

你的能量函数可以写成：

$$
E(\Gamma)=\sum_{i=2}^{M-1}\left(\kappa_{g,i}^2 + m\|r_i-r_i^0\|^2\right).
$$

这里第一项惩罚曲面内弯曲，第二项防止路径点偏离初始扫描路径太远。

这部分不是只画一条路径的能量下降曲线，而是对所有路径都记录迭代过程。对每一条路径 $\Gamma_k$，记录三条曲线：

| 曲线 | 含义 |
|---|---|
| total energy | 总能量是否收敛 |
| curvature term | 测地曲率项是否下降 |
| fidelity term | 路径偏移项是否上升过多 |

推荐画法分两层：

1. **代表性路径图**：选择 3 条典型路径，分别来自平滑表面、高曲率表面和边界复杂区域，画 total/curvature/fidelity 三条能量曲线。
2. **全部路径统计图**：对所有路径的能量曲线按 iteration 归一化后画 mean $\pm$ 95% CI，横轴为 normalized iteration，纵轴为 normalized energy。

这样证明的是：

> 不是某一条路径收敛，而是大多数路径的能量函数都稳定收敛，并且曲率项整体下降。

## 2.2 多路径路径点移动过程

不要只放一条路径的初始和最终状态。建议在 3 个代表性 surface 上分别展示多帧过程：

- iteration 0：初始路径；
- iteration 10%：早期移动；
- iteration 30%：中间状态；
- iteration 60%：接近稳定；
- final：最终路径。

同时，对每条路径记录路径点移动量：

$$
\Delta r_i = r_i^* - r_i^0.
$$

推荐画法：

| 图 | 内容 | 目的 |
|---|---|---|
| 代表性 3D 图 | 3 个 surface 上的 initial/intermediate/final paths | 直观展示点如何移动 |
| 位移箭头图 | 每个点的 $\Delta r_i$，箭头长度表示移动距离 | 展示哪些区域移动最多 |
| 全路径位移统计图 | 所有路径的 mean/max displacement boxplot | 证明位移没有失控 |

这样可以说明：蛇形路径拐弯处或高曲率区域通常移动最多，而平滑区域移动较少。这个结果比只展示一条路径更有说服力。

## 2.3 不同参数 m 的多路径消融实验

建议测试：

$$
m \in \{0,10^{-4},10^{-3},10^{-2},10^{-1},1\}.
$$

每一个 $m$ 都要在全部路径上运行，而不是只在一条路径上调参。对每条路径和每个 $m$ 统计以下指标：

| 指标 | 作用 |
|---|---|
| mean geodesic curvature | 平均曲面内扭转 |
| max geodesic curvature | 最大局部突变 |
| accumulated geodesic curvature | 整条路径总扭转 |
| path length | 路径是否变长 |
| average displacement | 偏离初始路径程度 |
| coverage preservation | 是否仍覆盖目标区域 |
| adjacent probe-frame rotation | 探头姿态连续性 |

推荐主图为 trade-off curve：

- x 轴：所有路径的 average displacement from initial path 的均值；
- y 轴：所有路径的 accumulated geodesic curvature 的均值；
- 每个点代表一个 $m$；
- 误差条表示跨路径标准差或 95% CI。

可以再在 supplementary 中放参数热图：行是 $m$，列是 curvature reduction、path deviation、coverage、length change 等指标。这样可以证明参数选择不是只适合某一条路径，而是在多路径上取得稳定折中。

## 2.4 多路径几何指标

测地线优化本身要证明：它在多条路径上从几何上降低了扭转，同时没有牺牲覆盖。

对每条路径 $\Gamma_k$，每种方法都输出一行 summary metrics：

| Path ID | Phantom | ROI | Method | Acc. $\kappa_g$ | Max frame rotation | Length | Coverage |
|---|---|---|---|---:|---:|---:|---:|
| P01 | phantom 1 | full | initial |  |  |  |  |
| P01 | phantom 1 | full | spline |  |  |  |  |
| P01 | phantom 1 | full | geodesic |  |  |  |  |
| P02 | phantom 1 | upper | initial |  |  |  |  |
| P02 | phantom 1 | upper | geodesic |  |  |  |  |

推荐指标如下：

| 指标 | 解释 |
|---|---|
| mean/max/accumulated geodesic curvature | 曲面内转弯程度 |
| path length | 路径长度是否增加过多 |
| adjacent tangent angle | 相邻路径方向是否连续 |
| adjacent normal angle | 表面法向变化 |
| adjacent probe-frame rotation angle | 探头姿态是否跳变 |
| tangent/frame total variation | 曲面 frame 稳定性 |
| coverage rate | 是否漏扫 |
| overlap ratio | 是否重复扫太多 |
| deviation from initial path | 是否保持原始扫描意图 |

推荐图形组织：

| 图 | 用途 |
|---|---|
| 3 个代表性 3D surface 路径图 | 展示路径形状 |
| 全部路径的 boxplot/violin plot | 证明统计优势 |
| normalized path progress 曲线 | 比较整条路径上的 $|\kappa_g|$ 分布 |
| surface heatmap | 展示高曲率区域是否减少 |

其中 normalized path progress 的横轴为

$$
s_{norm}=\frac{s}{L}\in[0,1].
$$

不同路径长度不同，必须先归一化弧长后再画 mean $\pm$ 95% CI。推荐结论句：

> Compared with the initial serpentine path and spline smoothing, geodesic optimization reduced accumulated geodesic curvature and adjacent probe-frame rotation across multiple breast surfaces and ROIs, while preserving surface coverage and maintaining limited path deviation.

# 3. B 组实验：真实机械臂扫查验证

## 3.1 关于力控是否会覆盖测地线效果

这个问题必须正面处理。实验逻辑应是：

> 所有路径方法都使用同一个法向力控。测地线的价值不是替代力控，而是在相同力控下减少力控负担、切向拖拽、力矩波动和姿态突变。

不要把“无力控 initial path”与“有力控 geodesic path”作为主对照，这不公平。无力控实验只作为动机实验，不作为证明测地线优越性的主实验。

## 3.2 E0 动机实验：只靠相机不可靠

这个实验可以做得很短，放在 motivation 或 supplementary。它可以在多条代表性路径上做，但不需要大规模展开。

| 方法 | 目的 |
|---|---|
| camera-only position following | 证明只靠相机可能悬空或过压 |
| camera path + normal contact search | 证明接触搜索有必要 |
| camera path + normal force control | 证明法向力闭环可以保证安全接触 |

指标：

| 指标 | 说明 |
|---|---|
| contact acquisition success rate | 是否能建立接触 |
| contact loss count | 扫描中是否脱离 |
| over-pressure count | 是否压太深 |
| low-quality frame ratio | 是否产生大量低质量图像 |
| max normal force | 安全性 |

这个实验的结论是：

> Camera-only surface following is insufficient for contact-rich ultrasound scanning.

## 3.3 多路径真实扫查主实验：相同力控下比较路径

真实机械臂实验也要从一开始设计成多 trial，而不是只跑一条路径。实验单位为：

> 一个 phantom 上，一条路径，用一种方法完成一次扫描 trial。

推荐规模：

$$
5\ \text{phantoms} \times 3\ \text{methods} \times 5\ \text{repeats} = 75\ \text{scan trials}.
$$

所有路径使用完全相同的法向力控参数。

| 组别 | 路径 | 接触控制 |
|---|---|---|
| B1 | Initial serpentine | same normal force controller |
| B2 | Spline/NURBS smoothed path | same normal force controller |
| B3 | Geodesic path | same normal force controller |

力控只做局部法向补偿，不做切向重规划、不做大范围姿态搜索、不做图像寻优。也就是说，路径仍然决定沿哪里扫和姿态如何变化。

每个 scan trial 都要输出一行 summary metrics：

| Trial ID | Phantom | Path ID | Method | Force error | Tangential ratio | Torque max | Q mean |
|---|---|---|---|---:|---:|---:|---:|
| T001 | phantom 1 | P01 | initial |  |  |  |  |
| T002 | phantom 1 | P01 | spline |  |  |  |  |
| T003 | phantom 1 | P01 | geodesic |  |  |  |  |

这样证明的是：在同一批路径和同一套力控下，geodesic 是否整体降低接触负担。

## 3.4 推荐控制相关指标

不要只看 normal force mean，因为法向力控会主动把 $F_n$ 拉到目标值，导致不同路径之间差异被部分抹平。应重点看**力控负担**和**切向接触质量**。

| 类别 | 指标 | 说明 |
|---|---|---|
| 法向接触 | normal force mean/std | 接触稳定性 |
| 力控负担 | normal compensation displacement | 力控修正了多少 |
| 力控负担 | normal compensation velocity/jerk | 是否频繁救火 |
| 力误差 | $|F_n-F_{target}|$ mean/std | 力控跟踪难度 |
| 切向拖拽 | $(|F_x|+|F_y|)/|F_z|$ | 侧向接触不稳定 |
| 扭矩 | torque mean/max/std | 探头扭转风险 |
| 接触稳定 | contact loss count | 是否掉接触 |
| 姿态稳定 | orientation tracking error | 姿态是否稳定 |
| 成像 | Q / low-quality frame ratio | 路径是否影响图像 |

推荐画法：

| 图 | 内容 |
|---|---|
| 代表性时间序列 | 只选 1-2 个 trial，画 $F_n$、tangential ratio、torque、Q |
| 全部 trial 统计图 | 用 boxplot/violin plot 画 force error、torque max、normal compensation jerk |
| normalized time curve | 把每次 scan 时间归一化后画 mean $\pm$ CI |

推荐结论句：

> Under identical normal force control, the geodesic path reduced required normal compensation, tangential wrench fluctuation, and probe-frame discontinuity across multiple scan trials.

## 3.5 多路径力控增益扫描实验

建议做三档力控增益：

| 增益 | 含义 |
|---|---|
| Low gain | 法向补偿弱，路径影响明显 |
| Medium gain | 正常使用 |
| High gain | 法向补偿强，可能掩盖路径差异 |

这个实验也不要只在一条路径上做。建议选择至少 3 个 phantom、每个 phantom 2 条代表性路径，比较 initial path 与 geodesic path。

理想结果是：

- Low/Medium gain 下，geodesic 明显更稳；
- High gain 下，差异变小，但 force overshoot、torque 或抖动增加。

这个实验能支持一句很强的结论：

> Geodesic optimization reduces the need for aggressive force control.

## 3.6 力传感器零漂、探头重量和重力补偿

真实实验必须处理这些问题。力传感器原始读数可以理解为：

$$
W_{raw}=W_{contact}+W_{gravity}+W_{bias}+W_{drift}.
$$

建议写入实验流程：

1. **每次实验前无接触清零**。
2. **工具重力补偿**：估计探头、夹具、传感器下方组件质量和质心位置，根据当前姿态扣除重力项。
3. **姿态相关基线标定**：在无接触下摆到若干典型姿态，记录 wrench，拟合姿态相关基线。
4. **低通滤波**：对力/力矩信号做 5-10 Hz 低通滤波，具体取决于扫描速度。
5. **多用相对指标**：例如 tangential-to-normal ratio、force std、torque std、compensation magnitude，比绝对值更抗零漂。

# 4. C 组实验：图像质量指标 Q 是否有意义

不要只证明“优化 Q 后 Q 变高”。这不够。你必须先证明 Q 与人类评价一致。

## 4.1 Q 与人工评分相关性

采集 200-500 张超声图像，图像应来自多个 phantom、多个 ROI、多个接触力范围和多个探头姿态，不要只来自某一次 BO 实验。数据应覆盖：

- 好图像、中等图像、坏图像；
- 不同 phantom；
- 不同接触力；
- 不同探头姿态；
- 不同 target 可见程度；
- 平坦区域、高曲率区域、边界区域和 target 附近区域。

让 2-3 名评分者盲评。

| 评分项 | 分数 |
|---|---|
| image clarity | 1-5 |
| acoustic coupling | 1-5 |
| target visibility | 1-5 |
| diagnostic usefulness | 1-5 |

统计：

| 分析 | 目的 |
|---|---|
| Q vs human score 的 Pearson/Spearman correlation | 验证 Q 是否符合人类判断 |
| 单特征 D/E/C/S vs human score | 验证每个特征贡献 |
| ICC 或 Fleiss kappa | 验证评分者一致性 |
| good/bad image classification AUC | 验证 Q 是否能区分好坏图 |

推荐图：

- scatter plot: Q vs human score；
- boxplot: 不同人工评分等级对应的 Q 分布；
- bar chart: 各图像特征与人工评分的相关系数；
- 按区域类型分层显示 Q 与人工评分的相关性，说明 Q 在不同点位类型上都稳定。

## 4.2 Q 特征消融

比较不同图像质量指标：

| 方法 | 含义 |
|---|---|
| D only | 只用 confidence-map response |
| E only | 只用 entropy |
| C only | 只用 contrast |
| S only | 只用 speckle index |
| unweighted average | 简单平均 |
| entropy-weighted + TOPSIS | 你的方法 |

比较：

- 与人工评分相关性；
- 区分好图/坏图的 AUC；
- 在不同 phantom 上的稳定性；
- 在不同区域类型上的稳定性。

结论目标：

> The fused Q score better agreed with human-rated image quality than individual features or unweighted fusion across multiple phantoms and region types.

# 5. D 组实验：贝叶斯优化连接机械臂后的效果

BO 部分必须从一开始设计成多点位实验。一个 BO 样本是：

> 一个扫描点位上的局部姿态优化任务。

推荐按区域分层选点：

| 点位类型 | 数量建议 |
|---|---:|
| 平坦区域 | 30 |
| 高曲率区域 | 30 |
| 边界区域 | 30 |
| target 附近 | 30 |
| 初始 Q 较低区域 | 30 |

总点位数建议为 120-150 个，来自多个 phantom、多个 ROI 和多条路径。每个点位都要做同样的方法比较，而不是只展示一个 before/after。

## 5.1 为什么用贝叶斯优化，而不是随机搜索或网格搜索

在相同局部搜索空间、相同 evaluation budget 下比较：

| 方法 | 目的 |
|---|---|
| Fixed normal pose | 基础对照 |
| Random search | 基础随机试探 |
| Grid search | 穷举式搜索 |
| Bayesian optimization | 你的方法 |

每个点位输出以下数据：

| Point ID | Phantom | Region | Method | Initial Q | Final Q | Human score | Evaluations | Violations |
|---|---|---|---|---:|---:|---:|---:|---:|
| X001 | phantom 1 | flat | fixed |  |  |  |  |  |
| X001 | phantom 1 | flat | BO |  |  |  |  |  |
| X002 | phantom 1 | boundary | fixed |  |  |  |  |  |

指标：

| 指标 | 说明 |
|---|---|
| best achieved Q | 最佳图像质量 |
| number of evaluations | 物理试探次数 |
| time to reach target quality | 达到目标质量所需时间 |
| force/torque violations during search | 搜索过程是否危险 |
| final target visibility | 最终目标可见性 |

推荐画法：

- 代表性 before/after 图只选 4 个点：flat、high-curvature、boundary、target；
- 全部点位用 paired plot 显示 fixed normal 到 full BO 的 Q 或人工评分变化；
- 用 boxplot/violin plot 比较 fixed、random、grid、BO 的 final Q；
- 用 mean $\pm$ CI 曲线比较 BO、random、grid 的 best-so-far Q 随 iteration 的变化。

BO 的核心优势应该写成：

> BO reaches comparable or better image quality with fewer physical evaluations across stratified scan points.

## 5.2 force/torque 惩罚项是否必要

这部分也要在同一批多点位上做消融，而不是只挑一个点展示。方法组如下：

| 方法 | 目的 |
|---|---|
| Image-only BO | 看只追求图像质量会不会不安全 |
| Image + force penalty | 看 force penalty 的作用 |
| Image + force + torque penalty | 完整方法 |

指标：

| 指标 | 说明 |
|---|---|
| final Q | 最终图像质量 |
| normal force max/std | 压力是否稳定 |
| tangential force ratio | 侧向拖拽 |
| torque max/std | 扭矩风险 |
| violation count | 安全超限次数 |
| target visibility | 目标可见性 |

理想现象：

- Image-only BO 可能 Q 高，但过压、扭矩或切向力更大；
- Image + force 改善压力，但不一定控制扭矩；
- Full BO 在 Q 和安全性之间取得更好平衡。

推荐画法：

| 图 | 内容 |
|---|---|
| 多点位 boxplot | final Q、force violation、torque max |
| paired plot | 同一点位 image-only 与 full BO 的 violation 对比 |
| region-wise analysis | 平坦、高曲率、边界、target 区域分别统计 |

## 5.3 为什么惩罚项用当前形式

你当前形式包含比值项和二次项。可以设计替代形式对照：

| 惩罚形式 | 作用 |
|---|---|
| ratio + quadratic | 当前方法 |
| pure quadratic | 只惩罚绝对力/力矩 |
| absolute linear | 线性惩罚 |

这些惩罚形式也应在同一批点位上比较。比较：

- Q 是否下降太多；
- tangential force ratio 是否被抑制；
- over-pressure 是否减少；
- torque 是否更稳；
- 在不同区域类型上的稳定性。

解释逻辑：

> Ratio terms penalize tangential interaction relative to normal loading, while quadratic terms suppress excessive absolute force/torque.

## 5.4 参数扫描

不要只拍脑袋设置 $\lambda_f,\lambda_\tau$。建议扫描：

$$
\lambda_f \in \{0,0.1,0.5,1,2\},
$$

$$
\lambda_\tau \in \{0,0.1,0.5,1,2\}.
$$

参数扫描不必在全部 150 个点上都完整运行，可以选取一个代表性子集，例如每种区域类型各 10 个点，共 40-50 个点。画 heatmap：

- 横轴：$\lambda_f$；
- 纵轴：$\lambda_\tau$；
- 颜色：综合 performance score，例如 Q - violation penalty。

也可以分别画 final Q、force violation、torque violation 三张 heatmap。

## 5.5 BO 与测地线优化的关系

它们不是同一个层面的算法：

| 模块 | 作用层级 |
|---|---|
| 测地线优化 | 全局几何路径层，决定沿哪里扫、局部 frame 如何变化 |
| BO | 局部姿态成像层，决定某个点附近怎么微调探头 |

可以增加一个联系实验，但也要设计成多点位/多路径：

| 方法 | 比较内容 |
|---|---|
| Initial path + BO | baseline |
| Geodesic path + BO | 看测地线路径是否让 BO 更省 |

指标：

- BO 触发次数；
- 每次 BO 所需迭代数；
- 初始 Q 到最终 Q 的提升幅度；
- BO correction magnitude；
- force/torque violation count。

如果 geodesic path 的初始姿态更稳定，BO 可能需要更少修正。推荐结论句：

> The geodesic path provides a better initialization for Bayesian local pose optimization, reducing trigger frequency, correction magnitude, and unsafe evaluations.

# 6. E 组实验：Darboux/TNB 映射示教采集

你不做模仿学习，那么示教部分要证明：

> Darboux/TNB 映射让示教更平滑、更稳定、更一致、更容易操作。

这部分也不能只让一个人采一条路径。实验单位为：

> 一个用户在一条路径上使用一种映射方式完成一次示教任务。

推荐规模：

$$
10\ \text{users} \times 3\ \text{paths} \times 2\ \text{methods} \times 2\ \text{repeats}
=120\ \text{demonstration trials}.
$$

## 6.1 不同人采多条路径的一致性

被试建议：

| 版本 | 人数 |
|---|---|
| 最低版 | 8 人 |
| 漂亮版 | 12-15 人 |
| 有超声经验者 | 1-3 人，有最好 |

路径建议至少包含 3 种难度：

| 路径 | 特点 |
|---|---|
| easy path | 表面平滑、曲率低 |
| medium path | 普通曲率和普通转弯 |
| hard path | 高曲率、边界附近或 target 附近 |

方法组：

| 方法 | 说明 |
|---|---|
| Cartesian teleoperation | 世界坐标系映射 |
| Darboux/TNB-frame teleoperation | 你的局部曲面坐标映射 |
| Normal-constrained teleoperation | 可选中间 baseline |

任务：

| 任务 | 目的 |
|---|---|
| Path following | 沿指定曲面路径扫 |
| Target finding | 找 phantom 中 target |
| Image holding | 在 target 处保持清晰图像 10 秒 |
| Correction task | 从差姿态调整到好图像 |

## 6.2 一致性指标

为了比较不同人是否采得一致，建议把所有示教轨迹投影到同一条 reference path 的路径坐标中：

$$
(\ell,d_b,d_n),
$$

其中 $\ell$ 是沿路径进度，$d_b$ 是侧向偏移，$d_n$ 是法向偏移。这样不同人的轨迹可以在同一个 $\ell$ 轴上对齐。

| 类别 | 指标 |
|---|---|
| 位置一致性 | path deviation to reference path |
| 位置一致性 | inter-user trajectory variance |
| 位置一致性 | lateral/normal offset variance |
| 姿态一致性 | orientation variance across users |
| 姿态一致性 | angular jerk |
| 力一致性 | force std across users |
| 力一致性 | force violation count variance |
| 图像一致性 | Q score variance |
| 图像一致性 | target visibility variance |

推荐画法：

| 图 | 内容 |
|---|---|
| 多用户轨迹叠加图 | 同一路径下 Cartesian vs Darboux 的所有用户轨迹 |
| 路径坐标曲线 | $d_b(\ell)$、$d_n(\ell)$ 的 mean $\pm$ std band |
| 全 trial 统计图 | trajectory variance、force std、Q fluctuation |

推荐结论句：

> Darboux mapping leads to more repeatable demonstration trajectories and interaction patterns across operators and path difficulties.

## 6.3 平滑性指标

| 类别 | 指标 | 意义 |
|---|---|---|
| 位置 | velocity variance | 速度是否忽快忽慢 |
| 位置 | acceleration RMS | 加速度是否大 |
| 位置 | jerk RMS | 轨迹是否抖 |
| 姿态 | angular velocity RMS | 姿态变化速度 |
| 姿态 | angular jerk RMS | 姿态抖动 |
| 路径 | curvature variance | 示教路径是否乱扭 |
| 力 | force std | 接触力是否稳定 |
| 力 | force jerk | 力是否忽大忽小 |
| 力矩 | torque std/jerk | 扭矩是否稳定 |
| 成像 | Q fluctuation | 图像质量是否稳定 |

即使不做模仿学习，也可以写：

> Smoother trajectories and more stable force profiles indicate higher-quality demonstrations for future learning-based control.

## 6.4 主观评分

用 1-7 分 Likert 量表即可。每个用户在每种映射方式、每条路径后都评分。

| 问题 |
|---|
| 操作是否直观 |
| 是否容易沿表面移动 |
| 是否容易控制压迫 |
| 是否容易获得清晰图像 |
| 是否适合采集示教数据 |
| 整体负担是否低 |
| 更愿意使用哪个模式 |

可以画雷达图或条形图。统计时建议把 user 作为随机效应，或者先对每个 user 聚合后做配对比较。

# 7. F 组附加加分实验

## 7.1 Target visibility 实验

在 phantom 中放 target：

| target | 模拟 |
|---|---|
| 低回声小球 | 肿块 |
| 囊样目标 | cyst |
| 硬小球 | 高回声结构 |
| 不同深度 target | 深浅病灶 |
| 边界附近 target | 难扫区域 |

每个 phantom 放 2-5 个 target，且应分布在不同曲率和不同深度区域。指标：

- target detection rate；
- target visibility score；
- number of usable frames containing target；
- best-frame CNR/SNR；
- target localization error。

这能把文章从机器人运动验证提升到医学扫查结果验证。

## 7.2 鲁棒性实验

加入扰动：

| 扰动 | 大小 |
|---|---|
| 点云噪声 | 1, 2, 3 mm |
| 法向误差 | 5, 10, 15 degrees |
| 手眼标定平移误差 | 2, 5 mm |
| 手眼标定旋转误差 | 2, 5 degrees |
| phantom 姿态变化 | 平移/旋转 |
| 耦合变化 | 少胶/正常/多胶 |
| 表面软硬变化 | 不同硅胶硬度 |

鲁棒性实验也应在多条路径上测试。指标：

- scan success rate；
- coverage drop；
- Q drop；
- force tracking error；
- normal correction magnitude；
- violation count；
- contact loss count。

## 7.3 失败案例分析

建议主动展示失败案例：

| 失败类型 | 可能原因 |
|---|---|
| 高曲率边界处姿态不稳 | 表面法向变化过快 |
| BO 陷入局部最优 | 搜索空间或初值问题 |
| target 太深导致 Q 提升有限 | 超声物理限制 |
| 力传感器漂移导致误判 | 零漂/重力补偿不足 |
| 点云分割错误 | 相机遮挡或反光 |

每类失败报告发生次数、原因和处理方式，会显得论文更真实可信。

# 8. 推荐图表设计

## 8.1 测地线优化图

| 图 | 内容 |
|---|---|
| Fig. 1 | 系统整体框架：geodesic backbone - autonomous scan - demonstration collection |
| Fig. 2 | 3 个代表性 3D breast surface 上 initial/intermediate/final paths |
| Fig. 3 | 多路径 energy convergence：total/curvature/fidelity 的 mean $\pm$ CI |
| Fig. 4 | path point displacement vector field + 全路径 displacement boxplot |
| Fig. 5 | 多路径 geodesic curvature along normalized path progress |
| Fig. 6 | parameter $m$ trade-off curve，基于全部路径 |
| Fig. 7 | 全部路径的 curvature、frame rotation、coverage、length 统计图 |

## 8.2 机械臂扫查图

| 图 | 内容 |
|---|---|
| Fig. 8 | 代表性 trial 下不同路径的 force/torque 时间序列 |
| Fig. 9 | 全部 trial 的 normal compensation displacement 和 jerk |
| Fig. 10 | 全部 trial 的 force std、torque max、tangential ratio boxplot/violin plot |
| Fig. 11 | normalized scan progress 上 force/Q 的 mean $\pm$ CI |
| Fig. 12 | surface heatmap: force 或 Q 沿路径分布 |

## 8.3 BO 图

| 图 | 内容 |
|---|---|
| Fig. 13 | Q vs human score scatter plot，覆盖多 phantom、多区域图像 |
| Fig. 14 | 4 类代表点位的 BO before/after ultrasound images |
| Fig. 15 | BO vs random vs grid 的多点位 convergence curve |
| Fig. 16 | 全部点位的 fixed vs full BO paired plot |
| Fig. 17 | image-only vs full BO 的 Q 和 safety 多点位统计 |
| Fig. 18 | $\lambda_f,\lambda_\tau$ 参数 heatmap |
| Fig. 19 | $\Delta Q$ surface heatmap，显示 BO 在哪些区域最有用 |

## 8.4 示教采集图

| 图 | 内容 |
|---|---|
| Fig. 20 | 多用户多路径轨迹叠加：Cartesian vs Darboux |
| Fig. 21 | $d_b(\ell)$、$d_n(\ell)$ 的 mean $\pm$ std band |
| Fig. 22 | trajectory jerk、force std、Q fluctuation 的 violin plot |
| Fig. 23 | 示教轨迹在 3D surface 上的误差热图 |
| Fig. 24 | 用户主观评分雷达图或条形图 |

## 8.5 图像风格建议

- 主文图只放代表性案例和总体统计，不要把所有路径或所有点位都硬画在一张 3D 图上。
- 所有路径、所有 BO 点位、所有用户轨迹可以放 supplementary。
- 统一颜色：Initial 用灰色，Spline/NURBS 用蓝色，Geodesic 用红色，Full BO/Darboux 用绿色。
- 统一字体和字号。
- 尽量导出 PDF/SVG 矢量图。
- 每张图只讲一个主结论。
- 曲线图中少放文字，详细解释写到 caption。
- 统计图优先用 boxplot、violin plot、paired scatter plot、spaghetti plot + confidence band。
- surface heatmap 很高级，建议多用在 Q、force、coverage 的空间分布上。

# 9. 最终实验执行清单

## 第一阶段：测地线优化

1. 准备多 phantom、多 ROI，形成 15-25 条路径。
2. 对每条路径运行 initial、spline/NURBS、geodesic 三种方法。
3. 对每条路径画能量函数迭代曲线，并汇总 mean $\pm$ CI。
4. 对代表性路径画路径点移动过程和位移箭头。
5. 在全部路径上做不同 $m$ 参数扫描。
6. 对全部路径计算 geodesic curvature、path length、frame rotation、coverage。

## 第二阶段：真实机械臂扫查

7. 做 camera-only 动机实验，说明不能只靠视觉表面跟踪。
8. 在多路径、多重复 trial 上比较 initial/spline/geodesic，所有方法使用相同法向力控。
9. 计算 force/torque/normal compensation workload。
10. 在多路径上做力控增益扫描。
11. 写清楚力传感器零漂、重力和姿态补偿。

## 第三阶段：贝叶斯优化

12. 采集 200-500 张覆盖多 phantom、多区域、多质量等级的超声图像，做 Q 与人工评分相关性。
13. 做 Q 的单特征和融合方式消融。
14. 选取 120-150 个分层点位，比较 BO、random search、grid search、fixed normal。
15. 在同一批点位上比较 image-only、image+force、image+force+torque。
16. 在代表性点位子集上做 $\lambda_f,\lambda_\tau$ 参数扫描。
17. 做 initial path + BO vs geodesic path + BO 的衔接实验，统计 BO 触发次数、迭代数和修正幅度。

## 第四阶段：示教采集

18. 设计 3 条不同难度路径，招募 8-15 名用户。
19. 每个用户分别使用 Cartesian 和 Darboux 完成 path following、target finding、image holding 等任务。
20. 比较位置、姿态、力/力矩、图像质量的一致性。
21. 比较 jerk、force std、trajectory variance。
22. 做主观评分。
23. 报告跨用户重复性和失败案例。

## 第五阶段：附加加分项

24. 做 target visibility。
25. 做噪声/标定误差鲁棒性。
26. 做安全性统计。
27. 做失败案例分析。

# 10. 最终结果应支撑的核心结论

整篇论文最后要用数据支撑四句话：

1. **测地线优化有效**：它在多 phantom、多 ROI、多路径上保持覆盖，同时降低曲面内扭转和探头姿态突变。
2. **真实扫查更稳**：在相同法向力控下，测地线路径在多次真实 scan trial 中减少力控补偿负担、切向拖拽和力矩波动。
3. **BO 图像优化有效**：Q 与人工评价一致；BO 在多区域、多点位上比随机/网格搜索更省样本；force/torque 惩罚项提高接触安全性。
4. **Darboux 示教有效**：相比 Cartesian 映射，它在多用户、多路径、多任务中产生更平滑、更稳定、更一致、更容易操作的示教数据。

如果这四句话都被实验钉住，这篇文章就不只是“流程跑通”，而是一套完整的曲面乳腺机器人超声框架验证。目标期刊可以优先冲 T-ASE / IJCARS / TBME 边缘尝试，T-MRB 作为兜底。
