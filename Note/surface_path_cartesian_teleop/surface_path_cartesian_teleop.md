# 基于曲面路径约束的 GELLO/UR5 末端笛卡尔遥操作

## 1. 目标

本方案用于乳腺表面超声采集任务。路径规划流程保持不变：先由 D405 点云生成乳腺表面的蛇形 coverage 路径，用户确认路径后，系统自动把 UR5 探头移动到路径起点附近，再进入 GELLO 引导的曲面路径采集模式。

核心目标是：

```text
GELLO 末端做直观的前后/左右/上下操作
  -> 系统解释为路径进度和微调
  -> UR5 探头末端沿乳腺曲面上的蛇形路径运动
```

也就是说，GELLO 不再直接逐关节控制 UR5。GELLO 只作为末端空间输入设备，真实 UR5 接收的是系统计算出的 TCP 目标位姿。

旧模式：

```text
GELLO joint q_gello
  -> UR5 joint q_ur
  -> UR5 servoJ(q_ur)
```

新模式：

```text
GELLO joint q_gello
  -> 虚拟 UR5 FK 得到 T_virtual_tcp
  -> 解码为路径进度和微调
  -> 根据曲面 Darboux frame 得到 T_real_tcp_des
  -> UR5 servoL(T_real_tcp_des)
```

## 2. 整体流程

完整采集流程为：

```text
路径规划流程不变
  -> 用户 confirm path
  -> 系统计算曲面全局路径 Gamma(S)
  -> UR5 自动移动到第一个路径点外法向安全距离处
  -> UR5 沿外法向慢速接近到路径起点
  -> 标定 GELLO 输入方向
  -> 开始记录 episode
  -> GELLO 增量控制路径进度和微调
  -> UR5 跟踪 TCP 目标位姿
```

这里的主语始终是末端，但要区分两个点：

- GELLO 末端：输入设备的虚拟末端。
- UR5 TCP：机器人法兰/工具中心点，`servoL` 真正控制的点。
- 探头尖端：沿 UR5 末端 z 轴延伸出去的点，是真正需要落在曲面路径上的点。

UR5 不需要知道 GELLO 的关节含义。UR5 只执行目标 TCP 位姿。但路径约束应作用在探头尖端上，而不是 UR5 TCP 原点上。

## 3. 从蛇形路径到曲面全局路径

当前 `planned_path.json` 中已有蛇形路径点：

$$
\{p_0, p_1, \ldots, p_M\}, \qquad p_i \in \mathbb{R}^3
$$

每个点还带有表面法向：

$$
\{n_0, n_1, \ldots, n_M\}, \qquad n_i \in \mathbb{S}^2
$$

为了避免直接逐点硬跟踪，可以只保留蛇形路径中的拐点：

$$
P_0, P_1, P_2, \ldots, P_N
$$

例如原始路径是：

```text
1, 2, 3, ..., 10, 11, 12, ..., 20
```

可以保留：

```text
1, 10, 11, 20
```

然后对每两个相邻拐点生成一条曲面路径：

$$
\gamma_k : [0, L_k] \rightarrow \mathbb{R}^3,
\qquad
\gamma_k(0) = P_k,\quad \gamma_k(L_k) = P_{k+1}
$$

所有曲线段拼接后得到一条全局曲面路径：

$$
\Gamma : [0, L_{\mathrm{total}}] \rightarrow \mathbb{R}^3
$$

其中：

$$
L_{\mathrm{total}} = \sum_{k=0}^{N-1} L_k
$$

全局路径参数 \(S\) 表示沿整条蛇形路径的弧长进度。

## 4. 拐点之间的曲面路径生成

第一版推荐使用点云图上的测地线近似。给定 segmented breast point cloud，在点云上建立 kNN graph：

$$
G = (V, E)
$$

每个顶点是一个点云点：

$$
v_i \leftrightarrow x_i \in \mathbb{R}^3
$$

基础边代价为欧氏距离：

$$
c(i,j) = \|x_i - x_j\|_2
$$

可加入法向变化惩罚：

$$
c(i,j)
= \|x_i - x_j\|_2
+ \lambda_n \left(1 - n_i^\top n_j\right)
$$

然后对每段拐点 \(P_k \rightarrow P_{k+1}\) 做 Dijkstra 或 A*：

$$
\gamma_k^\star
=
\arg\min_{\gamma_k}
\sum_{(i,j)\in \gamma_k} c(i,j)
$$

这样得到的是一条贴合点云表面的离散曲线。后续再进行：

- 弧长重采样；
- 曲线平滑；
- 拐点圆角化；
- 法向滤波；
- frame 连续性修正。

不建议第一版直接用主曲率线作为路径生成器，因为主曲率线不保证连接指定两个拐点。也不建议第一版直接实现论文 DOF 的完整 workspace diffusion，因为当前任务的输入已经是一条 coverage 路径，沿路径构造 frame 更直接。

## 5. 曲面 Darboux TNB frame

对全局路径 \(\Gamma(S)\)，定义路径切向：

$$
t(S)
=
\frac{\frac{d\Gamma}{dS}}
{\left\|\frac{d\Gamma}{dS}\right\|}
$$

设表面外法向为：

$$
n(S)
$$

曲面内横向副法向方向定义为：

$$
b(S)
=
\frac{t(S) \times n(S)}
{\|t(S) \times n(S)\|}
$$

于是路径上的曲面 Darboux 标架仍然采用 \(TNB\) 顺序：

$$
D(S)
=
\begin{bmatrix}
t(S) & n(S) & b(S)
\end{bmatrix}
\in SO(3)
$$

其中：

- \(t(S)\)：沿路径前进方向；
- \(n(S)\)：表面外法向方向。
- \(b(S)\)：曲面内横向副法向方向。

注意：曲面 Darboux 标架 \(D(S)=[t,n,b]\) 不等于 UR5 末端姿态。因为探头沿 UR5 TCP 的 \(+z\) 轴伸出，而接触曲面时探头方向应指向曲面，即和表面外法向 \(n\) 相反。因此 TCP 的 \(+z\) 轴应满足：

$$
R_{\mathrm{tcp}} e_z = -n
$$

同时希望 UR5 TCP 的 \(+x\) 轴沿路径切向 \(t\)，因此机器人真正使用的基础 TCP 姿态是：

$$
R_{\mathrm{tcp,base}}(S)
=
\begin{bmatrix}
t(S) & b(S) & -n(S)
\end{bmatrix}
$$

这里 \(x_{\mathrm{tcp}}=t\)，\(z_{\mathrm{tcp}}=-n\)，为了保持右手系就必须有 \(y_{\mathrm{tcp}}=b\)。若实机测试发现 GELLO 往右但探头往左，不应改 Darboux 标架，而应改变 GELLO 输入到 \(b\) 方向的增益符号。

## 6. 起点自动对齐

用户 confirm 路径后，系统不要立刻进入 GELLO 采集，而应先把 UR5 探头移动到第一个路径点。

设路径起点为：

$$
p_0 = \Gamma(0)
$$

起点外法向为：

$$
n_0 = n(0)
$$

起点 frame 为：

$$
F_0 = F(0)
$$

定义安全预接近距离，例如：

$$
h_{\mathrm{approach}} = 0.20\ \mathrm{m}
$$

则预接近点为：

$$
p_{\mathrm{pre}}
=
p_0 + h_{\mathrm{approach}} n_0
$$

接触或采集起始高度为：

$$
h_{\mathrm{contact}}
$$

它表示起点相对于曲面点 \(p_0\) 沿外法向 \(n_0\) 的距离。若希望探头尖端正好落在曲面点上，可取：

$$
h_{\mathrm{contact}} = 0
$$

若希望探头先停在曲面外侧一点点，可取一个小正值，例如 \(2\sim5\ \mathrm{mm}\)。因此起始目标点为：

$$
p_{\mathrm{start}}
=
p_0 + h_{\mathrm{contact}} n_0
$$

目标姿态由起点 Darboux frame 派生，但不是直接使用 \(D_0=[t_0,n_0,b_0]\)。因为这里假设探头就是沿 UR5 末端 \(+z\) 轴延伸，而探头应朝向曲面，所以应让 UR5 TCP 的 \(+z\) 轴对齐 \(-n_0\)。因此可以令：

$$
R_{\mathrm{start}}
=
R_{\mathrm{tcp,base}}(0)
=
\begin{bmatrix}
t_0 & b_0 & -n_0
\end{bmatrix}
$$

这表示 UR5 TCP 的 x 轴对齐路径切向，y 轴对齐曲面内横向副法向，z 轴指向曲面内侧，即和表面外法向相反。若实机工具坐标轴还有额外固定旋转，再额外乘一个固定偏置 \(R_{\mathrm{mount}}\)，但第一版先按探头沿 TCP \(+z\) 轴延伸处理。

设探头尖端相对 UR5 TCP 沿 TCP z 轴延伸长度为：

$$
\ell_{\mathrm{probe}}
$$

则探头尖端位置和 UR5 TCP 位置满足：

$$
p_{\mathrm{tip}}
=
p_{\mathrm{tcp}}
+ \ell_{\mathrm{probe}} R_{\mathrm{tcp}} e_z
$$

其中：

$$
e_z =
\begin{bmatrix}
0\\
0\\
1
 \end{bmatrix}
$$

因为当前约定 \(R_{\mathrm{tcp,base}}=[t,b,-n]\)，第三列是 TCP 的 z 轴，也就是 \(-n\)。因此：

$$
R_{\mathrm{tcp}} e_z = -n
$$

为了避免代码实现时混淆，建议在实现中显式定义探头延伸方向：

$$
a_{\mathrm{probe}} = -n
$$

即探头延伸方向在 base 坐标系下就是当前表面外法向的反方向。

于是当希望探头尖端到达 \(p_{\mathrm{pre}}\) 或 \(p_{\mathrm{start}}\) 时，实际要发送给 UR5 的 TCP 位置是：

$$
p_{\mathrm{tcp,pre}}
=
p_{\mathrm{pre}}
+ \ell_{\mathrm{probe}} n_0
$$

$$
p_{\mathrm{tcp,start}}
=
p_{\mathrm{start}}
+ \ell_{\mathrm{probe}} n_0
$$

自动移动流程：

```text
moveL(T_pre)              # 先到曲面外法向安全距离处
moveL 或 servoL(T_start)  # 再沿外法向慢速接近起点
reset path state          # S=0, 横向/法向微调清零
reset GELLO reference     # 当前 GELLO 虚拟末端作为输入零点
进入方向标定
```

其中：

$$
T_{\mathrm{pre}}
=
\begin{bmatrix}
R_{\mathrm{start}} & p_{\mathrm{tcp,pre}} \\
0 & 1
\end{bmatrix}
$$

$$
T_{\mathrm{start}}
=
\begin{bmatrix}
R_{\mathrm{start}} & p_{\mathrm{tcp,start}} \\
0 & 1
\end{bmatrix}
$$

## 7. GELLO 虚拟 UR5 末端输入

当前普通 `GelloAgent` 只能拿到 GELLO 关节角：

$$
q_{\mathrm{gello}}
$$

经过已有 offset/sign 标定后，它本来会生成一个可直接发给 UR5 的关节 control。新方案中，不再把这个关节 control 发给真实 UR5，而是把它看作虚拟 UR5 的关节角：

$$
q_{\mathrm{virtual}}
$$

通过 UR5 正运动学得到虚拟末端位姿：

$$
T_{\mathrm{virtual}}(q_{\mathrm{virtual}})
=
\begin{bmatrix}
R_{\mathrm{virtual}} & p_{\mathrm{virtual}} \\
0 & 1
\end{bmatrix}
$$

这个虚拟末端位姿只用于解释 GELLO 操作意图，不用于直接控制真实 UR5。

相邻时刻虚拟末端位移为：

$$
\Delta p_{\mathrm{virtual},t}
=
p_{\mathrm{virtual},t}
-
p_{\mathrm{virtual},t-1}
$$

相邻时刻虚拟末端旋转增量为：

$$
\Delta R_{\mathrm{virtual},t}
=
R_{\mathrm{virtual},t-1}^{\top}
R_{\mathrm{virtual},t}
$$

## 8. GELLO 输入方向标定

为了避免纠结 GELLO 模型中 TCP 的 x 轴到底朝哪里，建议在每次开始采集前做一次操作方向标定。

标定前应该允许操作者把 GELLO 自由移动到舒服的位置。这个阶段 GELLO 只被读取，不发送控制给 UR5。操作者摆到舒服姿态后点击 `set reference` 或 `start calibration`，系统记录当前虚拟 TCP 位姿：

$$
T_{\mathrm{virtual,ref}}
=
\begin{bmatrix}
R_{\mathrm{virtual,ref}} & p_{\mathrm{virtual,ref}}\\
0 & 1
\end{bmatrix}
$$

之后所有 GELLO 输入都用相对于这个 reference 的增量解释。这样 GELLO 不需要一直维持在某个固定机械零位，也不需要在开始时和 UR5 末端姿态完全一致。

记录标定开始时的虚拟 TCP 位置：

$$
g_0 = p_{\mathrm{virtual},0}
$$

用户将 GELLO 末端往自己认为的“前进方向”短推 2 到 3 cm，记录：

$$
g_1 = p_{\mathrm{virtual},1}
$$

定义 GELLO 前进方向：

$$
e_x
=
\frac{g_1 - g_0}{\|g_1 - g_0\|}
$$

竖向方向不应默认使用虚拟 TCP 初始 z 轴。更稳妥的做法是再做一次人工标定：用户将 GELLO 末端沿真实世界或 UR base 坐标系下的 \(+z\) 方向短推 2 到 3 cm，记录：

$$
g_{z,0}=p_{\mathrm{virtual},z0},
\qquad
g_{z,1}=p_{\mathrm{virtual},z1}
$$

得到原始竖向输入方向：

$$
\tilde e_z
=
\frac{g_{z,1}-g_{z,0}}
{\|g_{z,1}-g_{z,0}\|}
$$

这里的意思不是说 GELLO 自己知道真实世界 z 轴，而是把“操作者实际往世界 \(+z\) 推 GELLO 时，虚拟 TCP 在 FK 空间中的位移方向”记录下来。以后只要 GELLO 虚拟 TCP 位移投影到这个方向，就解释为标定后的 z 输入。

为了让 \(e_z\) 和 \(e_x\) 正交，先去掉 \(\tilde e_z\) 中沿 \(e_x\) 的分量：

$$
e_z'
=
\tilde e_z
-
(\tilde e_z^\top e_x)e_x
$$

$$
e_z
=
\frac{e_z'}{\|e_z'\|}
$$

为了保证正交，横向方向可定义为：

$$
e_y
=
\frac{e_z \times e_x}{\|e_z \times e_x\|}
$$

然后重新正交化：

$$
e_z
=
\frac{e_x \times e_y}{\|e_x \times e_y\|}
$$

这样得到 GELLO 输入坐标系：

$$
E_{\mathrm{gello}}
=
\begin{bmatrix}
e_x & e_y & e_z
\end{bmatrix}
$$

其中：

- \(e_x\)：GELLO 前推方向；
- \(e_y\)：GELLO 横向微调方向；
- \(e_z\)：GELLO 上下/法向微调方向。

## 9. 路径进度和微调更新

每个控制周期，先把虚拟 TCP 位移投影到标定后的 GELLO 输入坐标系：

$$
\Delta g_t
=
E_{\mathrm{gello}}^\top
\Delta p_{\mathrm{virtual},t}
=
\begin{bmatrix}
\Delta g_x \\
\Delta g_y \\
\Delta g_z
\end{bmatrix}
$$

这里所谓“识别 GELLO 往标定后的 x 方向走”，不是做离散分类，而是做连续投影。若：

$$
\Delta g_x = e_x^\top \Delta p_{\mathrm{virtual},t} > \epsilon_x
$$

则认为操作者正在沿标定后的 \(+x\) 方向推进；若：

$$
\Delta g_x < -\epsilon_x
$$

则认为在往 \(-x\) 方向退回。若 \(|\Delta g_x|\le\epsilon_x\)，则把这部分输入视为手抖或噪声，不更新路径进度。

路径进度更新：

$$
S_t
=
\mathrm{clip}
\left(
S_{t-1} + k_x \Delta g_x,\,
0,\,
L_{\mathrm{total}}
\right)
$$

曲面内横向微调：

$$
y_t
=
\mathrm{clip}
\left(
y_{t-1} - k_y \Delta g_y,\,
-y_{\max},\,
y_{\max}
\right)
$$

法向高度/压入微调：

$$
h_t
=
\mathrm{clip}
\left(
h_{t-1} + k_z \Delta g_z,\,
h_{\min},\,
h_{\max}
\right)
$$

所以：

- GELLO 往标定后的 x 方向移动：探头沿当前路径切向 \(t(S)\) 前进；
- GELLO 往标定后的 z 方向移动：探头沿当前曲面外法向 \(n(S)\) 微调高度；
- 为了保持 GELLO 输入坐标系右手性，如果 \(x_{\mathrm{gello}}\mapsto t\)，\(z_{\mathrm{gello}}\mapsto n\)，则必然有 \(y_{\mathrm{gello}}\mapsto -b\)。

因此 GELLO 平移输入到曲面方向的映射是：

$$
M_{\mathrm{trans}}(S)
=
\begin{bmatrix}
t(S) & -b(S) & n(S)
\end{bmatrix}
$$

也就是说：

$$
\Delta p_{\mathrm{tip}}
=
k_x\Delta g_x\, t(S)
- k_y\Delta g_y\, b(S)
+ k_z\Delta g_z\, n(S)
$$

如果把一次 GELLO 末端动作写成齐次变换：

$$
\Delta T_{\mathrm{gello}}
=
\begin{bmatrix}
\Delta R_g & \Delta p_g\\
0 & 1
\end{bmatrix}
$$

这里的 \(\Delta p_g\) 和 \(\Delta R_g\) 都是在 GELLO 虚拟 FK 空间中表达的增量。先把平移增量投影到标定输入轴：

$$
\begin{bmatrix}
\Delta g_x\\
\Delta g_y\\
\Delta g_z
\end{bmatrix}
=
E_{\mathrm{gello}}^\top \Delta p_g
$$

然后更新路径状态：

$$
S^+
=
\mathrm{clip}
\left(
S+k_x\Delta g_x,\,
0,\,
L_{\mathrm{total}}
\right)
$$

$$
y^+
=
\mathrm{clip}
\left(
y-k_y\Delta g_y,\,
-y_{\max},\,
y_{\max}
\right)
$$

$$
h^+
=
\mathrm{clip}
\left(
h+k_z\Delta g_z,\,
h_{\min},\,
h_{\max}
\right)
$$

新的探头尖端目标为：

$$
p_{\mathrm{tip}}^+
=
\Gamma(S^+)
+ y^+ b(S^+)
+ h^+ n(S^+)
$$

在小步长近似下，也可以理解为：

$$
p_{\mathrm{tip}}^+
\approx
p_{\mathrm{tip}}
+ k_x\Delta g_x\,t
- k_y\Delta g_y\,b
+ k_z\Delta g_z\,n
$$

## 10. 目标 UR5 TCP 位姿

真实希望约束的是探头尖端位置：

$$
p_{\mathrm{tip\_des},t}
=
\Gamma(S_t)
+ y_t b(S_t)
+ h_t n(S_t)
$$

基础目标姿态按当前路径的机器人 TCP frame 给出：

$$
R_{\mathrm{base},t}
=
R_{\mathrm{tcp,base}}(S_t)
=
\begin{bmatrix}
t(S_t) & b(S_t) & -n(S_t)
\end{bmatrix}
$$

如果使用 GELLO wrist rotation 做姿态微调，也按同一套映射关系处理。这里处理的是一次控制周期内的姿态增量，不是角速度。

先从 GELLO 虚拟 TCP 姿态得到世界系下的小旋转增量：

$$
\delta R_{\mathrm{virtual},t}
=
R_{\mathrm{virtual},t}
R_{\mathrm{virtual},t-1}^{\top}
$$

$$
\delta \phi_{\mathrm{virtual},t}
=
\log(\delta R_{\mathrm{virtual},t})^\vee
$$

其中 \(\delta \phi_{\mathrm{virtual},t}\) 是 axis-angle 形式的旋转增量向量，单位是 rad，不是 rad/s。

再投影到标定后的 GELLO 输入轴：

$$
\begin{bmatrix}
\delta \theta_x\\
\delta \theta_y\\
\delta \theta_z
\end{bmatrix}
=
E_{\mathrm{gello}}^\top
\delta \phi_{\mathrm{virtual},t}
$$

然后映射到曲面任务方向：

$$
\delta \phi_{\mathrm{task},t}
=
k_{rx}\delta \theta_x\, t(S_t)
- k_{ry}\delta \theta_y\, b(S_t)
+ k_{rz}\delta \theta_z\, n(S_t)
$$

其中 \((\delta \theta_x,\delta \theta_y,\delta \theta_z)\) 是 GELLO 虚拟末端旋转增量投影到标定后输入轴得到的三个小角度。它的含义是：

- GELLO 绕标定后的 x 轴转：UR5 TCP 绕 \(t(S)\) 小角度旋转；
- GELLO 绕标定后的 z 轴转：UR5 TCP 绕 \(n(S)\) 小角度旋转；
- GELLO 绕标定后的 y 轴转：UR5 TCP 绕 \(-b(S)\) 小角度旋转。

注意最后一项是负号，原因和位移一样：当 \(x_{\mathrm{gello}}\mapsto t\)，\(z_{\mathrm{gello}}\mapsto n\) 时，右手系要求 \(y_{\mathrm{gello}}\mapsto -b\)。

若直接使用增量姿态控制，最终目标姿态可写为：

$$
R_{\mathrm{des},t}
=
\exp\left(\widehat{\delta \phi_{\mathrm{task},t}}\right)
R_{\mathrm{base},t}
$$

上式是“世界系左乘”写法：\(\delta \phi_{\mathrm{task},t}\) 用 base/world 坐标表达，表示绕世界系中的 \(t,-b,n\) 这些方向转。

等价地，也可以在 UR5 TCP 局部坐标系里写成“局部右乘”形式。此时增量向量不是用世界系表达，而是用 TCP 自己的 \(x,y,z\) 轴表达：

$$
\delta r_{\mathrm{tcp}}
=
\begin{bmatrix}
k_{rx}\delta\theta_x\\
-k_{ry}\delta\theta_y\\
-k_{rz}\delta\theta_z
\end{bmatrix}
$$

$$
R_{\mathrm{des},t}
=
R_{\mathrm{base},t}
\exp\left(\widehat{\delta r_{\mathrm{tcp}}}\right)
$$

这里第三项也是负号，因为 \(z_{\mathrm{tcp}}=-n\)。所以“GELLO 绕标定 z 正方向转，UR5 绕 \(+n\) 转”，在 TCP 局部坐标中等价于绕 \(-z_{\mathrm{tcp}}\) 转。

### GELLO 绕标定 y 轴旋转的例子

先明确几个符号。

当前路径点处的 UR5 基础 TCP 姿态是：

$$
R_{\mathrm{base}}
=
\begin{bmatrix}
t & b & -n
\end{bmatrix}
$$

这表示：

$$
x_{\mathrm{tcp}}=t,
\qquad
y_{\mathrm{tcp}}=b,
\qquad
z_{\mathrm{tcp}}=-n
$$

标定后的 GELLO 旋转输入和曲面任务方向之间的约定是：

$$
x_{\mathrm{gello}}\mapsto t,
\qquad
y_{\mathrm{gello}}\mapsto -b,
\qquad
z_{\mathrm{gello}}\mapsto n
$$

因此，如果 GELLO 绕标定后的 \(+y_{\mathrm{gello}}\) 轴按右手定则旋转 \(a\) 度，先把角度换成弧度：

$$
a_{\mathrm{rad}}
=
a\frac{\pi}{180}
$$

那么 GELLO 标定输入坐标中的姿态增量向量是：

$$
\delta \theta_{\mathrm{gello}}
=
\begin{bmatrix}
0\\
a_{\mathrm{rad}}\\
0
\end{bmatrix}
$$

这个向量的意思是“只绕 GELLO 标定后的 \(+y\) 轴转 \(a_{\mathrm{rad}}\)”。

因为 \(+y_{\mathrm{gello}}\mapsto -b\)，所以同一个动作在曲面任务空间中的姿态增量向量是：

$$
\delta\phi_{\mathrm{task}}
=
\left(0\right)t
+\left(a_{\mathrm{rad}}\right)(-b)
+\left(0\right)n
=
-a_{\mathrm{rad}}\, b
$$

这里 \(b\) 是一个三维单位向量，\(-a_{\mathrm{rad}}\,b\) 表示：旋转轴是 \(-b\)，旋转角度是 \(a_{\mathrm{rad}}\)。这就是“GELLO 绕标定后的 \(+y\) 轴转 \(a\) 度，UR5 末端绕 \(-b\) 方向转 \(a\) 度”的数学含义。

如果一个三维旋转增量向量为：

$$
v=
\begin{bmatrix}
v_x\\
v_y\\
v_z
\end{bmatrix}
$$

则帽子算子定义为：

$$
\widehat v
=
\begin{bmatrix}
0 & -v_z & v_y\\
v_z & 0 & -v_x\\
-v_y & v_x & 0
\end{bmatrix}
$$

\(\exp(\widehat v)\) 是由这个 axis-angle 增量生成的旋转矩阵。

因此，用世界系左乘实现时：

$$
R_{\mathrm{des}}
=
\exp\left(\widehat{-a_{\mathrm{rad}}\, b}\right)
R_{\mathrm{base}}
$$

也可以把同一个旋转写到 UR5 TCP 局部坐标里。因为 \(R_{\mathrm{base}}\) 的第二列就是 \(b\)，所以：

$$
R_{\mathrm{base}}^\top b
=
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix}
$$

所以：

$$
R_{\mathrm{base}}^\top
\left(
-a_{\mathrm{rad}}\,b
\right)
=
\begin{bmatrix}
0\\
-a_{\mathrm{rad}}\\
0
\end{bmatrix}
$$

因此等价的 TCP 局部旋转向量是：

$$
\delta r_{\mathrm{tcp}}
=
\begin{bmatrix}
0\\
-a_{\mathrm{rad}}\\
0
\end{bmatrix}
$$

用 TCP 局部右乘实现时：

$$
R_{\mathrm{des}}
=
R_{\mathrm{base}}
\exp\left(\widehat{\delta r_{\mathrm{tcp}}}\right)
$$

左乘和右乘这两种写法等价，使用的是下面这个恒等式：

$$
\exp\left(\widehat v\right)R
=
R
\exp\left(\widehat{R^\top v}\right)
$$

在这个例子里：

$$
v=-a_{\mathrm{rad}}\,b,
\qquad
R=R_{\mathrm{base}}
$$

所以：

$$
\exp\left(\widehat{-a_{\mathrm{rad}}\,b}\right)R_{\mathrm{base}}
=
R_{\mathrm{base}}
\exp\left(\widehat{\delta r_{\mathrm{tcp}}}\right)
$$

实现时通常采用右乘形式，因为它直接对应 UR5 TCP 局部坐标中的姿态增量：

```python
a_rad = np.deg2rad(a_deg)
delta_r_tcp = np.array([0.0, -a_rad, 0.0])
R_des = R_base @ exp_so3(delta_r_tcp)
tcp_rotvec = rotvec_from_matrix(R_des)
```

实机上更建议把 \(\delta \theta_x,\delta \theta_y,\delta \theta_z\) 积分成三个有界的小姿态偏置，并限制在 \(5^\circ\sim15^\circ\) 内，避免探头大角度离开法向接触。

由于 `servoL` 控制的是 UR5 TCP，不是探头尖端，所以需要从探头尖端目标反推 TCP 目标。探头沿当前 \(-n(S_t)\) 方向延伸，因此 TCP 位于探头尖端的外法向一侧：

$$
p_{\mathrm{tcp\_des},t}
=
p_{\mathrm{tip\_des},t}
+ \ell_{\mathrm{probe}} n(S_t)
$$

最终发送给 UR5 的目标齐次变换为：

$$
T_{\mathrm{real\_tcp\_des},t}
=
\begin{bmatrix}
R_{\mathrm{des},t} & p_{\mathrm{tcp\_des},t} \\
0 & 1
\end{bmatrix}
$$

发送给 UR5 RTDE 时，转换为 UR 需要的 pose vector：

$$
u_{\mathrm{servoL},t}
=
\begin{bmatrix}
x_t & y_t & z_t & r_{x,t} & r_{y,t} & r_{z,t}
\end{bmatrix}^{\top}
$$

其中：

$$
\begin{bmatrix}
r_{x,t} & r_{y,t} & r_{z,t}
\end{bmatrix}^{\top}
=
\mathrm{rotvec}(R_{\mathrm{des},t})
$$

真实 UR5 执行：

```python
servoL([x, y, z, rx, ry, rz], velocity, acceleration, dt, lookahead_time, gain)
```

UR 控制器内部完成从 TCP 目标位姿到关节运动的求解。

## 11. GELLO 行程不足和 clutch

GELLO 不需要物理上一直向前推完整条路径。必须使用增量式输入：

$$
S_t = S_{t-1} + k_x \Delta g_x
$$

而不是绝对位移：

$$
S_t \neq k_x (g_t - g_0)
$$

为了让用户能把 GELLO 拉回舒适位置，需要 clutch/recenter：

```text
clutch pressed:
  更新 GELLO reference
  不更新 S, y, h, R_micro
  不改变 UR5 目标路径状态

clutch released:
  继续使用 GELLO 增量驱动路径状态
```

这样 GELLO 的使用方式类似鼠标：

```text
前推一小段 -> UR5 沿路径前进一段
按住 clutch 拉回 -> UR5 不动
松开后再前推 -> UR5 继续前进
```

## 12. 数据记录语义

旧模式中，control 通常是关节目标：

$$
a_t = q_{\mathrm{ur},t}^{\mathrm{target}}
$$

新模式中，control 应该变成 UR5 TCP 目标位姿：

$$
a_t
=
u_{\mathrm{servoL},t}
=
\begin{bmatrix}
x_t & y_t & z_t & r_{x,t} & r_{y,t} & r_{z,t}
\end{bmatrix}^{\top}
$$

建议每帧额外保存：

```text
surface_path_progress_S
surface_path_total_length
surface_path_segment_index
surface_path_local_s
surface_path_y_offset
surface_path_h_offset
surface_path_clutch_state
target_tcp_pose_base
actual_tcp_pose_base
target_frame_t_base
target_frame_n_base
target_frame_b_base
gello_virtual_tcp_pose
gello_virtual_tcp_delta
tracking_error_position
tracking_error_rotation
```

跟踪误差可定义为：

$$
e_p
=
p_{\mathrm{actual}} - p_{\mathrm{des}}
$$

旋转误差可定义为：

$$
R_e
=
R_{\mathrm{des}}^\top R_{\mathrm{actual}}
$$

或用 rotation vector 表示：

$$
e_R
=
\mathrm{rotvec}(R_e)
$$

## 13. `gello_get_offset.py` 的角色

旧命令示例：

```bash
python scripts/gello_get_offset.py \
    --start-joints -1.57 -1.57 -1.57 -1.57 1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAW77E7-if00-port0
```

旧用途：

```text
让 GELLO joint 和 UR5 joint 对齐
```

新模式中，真实 UR5 不再逐关节跟随 GELLO，所以不需要“UR5-GELLO 关节对齐”作为采集前必要步骤。

但仍然可以使用现有 offset/sign 逻辑，因为我们需要：

```text
GELLO joint
  -> q_virtual_ur
  -> UR5 FK
  -> T_virtual_tcp
```

因此 `gello_get_offset.py` 的新意义是：

- 让虚拟 UR5 关节角语义合理；
- 让 FK 算出的虚拟末端运动方向符合手感；
- 检查 joint signs 是否正确。

它不再用于把真实 UR5 关节调到和 GELLO 一一对应。

## 14. 安全限制

建议第一版加入以下限制：

$$
|y_t| \leq y_{\max}
$$

$$
h_{\min} \leq h_t \leq h_{\max}
$$

$$
\|\Delta p_{\mathrm{des},t}\| \leq \Delta p_{\max}
$$

$$
\|\mathrm{rotvec}(R_{\mathrm{des},t-1}^{\top}R_{\mathrm{des},t})\|
\leq
\Delta \theta_{\max}
$$

建议初始参数：

```text
y_max = 0.005 ~ 0.010 m
h_min / h_max = 依据探头接触策略设置
法向微调范围 = 0.002 ~ 0.005 m
姿态微调范围 = 5 ~ 15 deg
预接近距离 h_approach = 0.20 m
```

实机前必须先验证：

- 曲线 \(\Gamma(S)\) 连续；
- Darboux frame \(D(S)\) 和 TCP frame \(R_{\mathrm{tcp,base}}(S)\) 无翻转；
- servoL 目标位姿连续；
- 速度和姿态变化有限幅；
- 起点预接近方向确实是表面外法向；
- 探头延伸长度 \(\ell_{\mathrm{probe}}\) 正确。
- 探头确实沿 UR5 TCP 的法向轴延伸；如果实际轴不是文档里的第二列 \(n\)，需要用固定安装旋转重新排列。

## 15. 与论文 DOF 的关系

论文中的 DOF 方法是：

$$
\text{point cloud} + \text{keypoints}
\rightarrow
\text{diffusion PDE}
\rightarrow
u_\tau
\rightarrow
\nabla u_\tau
\rightarrow
\text{orientation field}
$$

它生成的是整个工作空间上的局部参考系场：

$$
u(x): \Omega \rightarrow SO(3)
$$

本方案不是完整 DOF。当前任务已经有规划好的 coverage path，因此只需要在路径上构造曲面 Darboux 标架：

$$
D(S)
=
\begin{bmatrix}
t(S) & n(S) & b(S)
\end{bmatrix}
$$

然后由它派生 UR5 真正使用的 TCP 姿态：

$$
R_{\mathrm{tcp,base}}(S)
=
\begin{bmatrix}
t(S) & b(S) & -n(S)
\end{bmatrix}
$$

所以本方案更准确地叫：

```text
基于曲面路径约束的末端笛卡尔遥操作
```

或者：

```text
path-constrained Cartesian teleoperation on a surface
```

它借用了论文中“在局部曲面坐标系中表达动作”的思想，但工程实现完全适配当前项目。

## 16. 当前代码支持情况

当前项目还没有完整支持“只给 UR5 发末端位置和姿态”的控制链路。

已经支持的部分：

- `URRobot.get_observations()` 可以通过 RTDE 读取真实 UR5 TCP pose；
- 保存数据链路可以保存 action 数组；
- GELLO 可以读取关节，并通过 offset/sign 得到虚拟 UR5 关节。

目前不支持的部分：

- `RobotEnv.step(action)` 当前只接受 joint action；
- `RobotEnv.step()` 内部调用的是 `robot.command_joint_state(joints)`；
- `URRobot.command_joint_state()` 当前使用 `servoJ`；
- `URRobot` 里还没有 `command_tcp_pose()` 或 `servoL()` 封装；
- `DeviceManager.clamp_action()` 当前也是按 joint delta 做限幅。

因此要实现本方案，需要新增一条 Cartesian action path：

```text
GELLO increment
  -> path state
  -> T_real_tcp_des = [R_des, p_tcp_des]
  -> URRobot.command_tcp_pose(tcp_pose_rotvec)
  -> RTDE servoL
```

同时 `RobotEnv.step()` 或新的 `CartesianRobotEnv.step()` 应该允许 action 为：

```text
[x, y, z, rx, ry, rz]
```

而不是 6 个关节角。

## 17. 第一版实现边界

第一版做：

- 从 `planned_path.json` 和 segmented point cloud 生成全局曲面路径；
- 构造 Darboux TNB frame；
- 使用虚拟 UR5 FK 解释 GELLO 末端增量；
- 自动移动 UR5 到路径起点；
- 使用 GELLO 增量控制路径进度和微调；
- 使用 UR5 `servoL` 执行 TCP 目标；
- 保存 TCP target control 和路径状态。

第一版不做：

- 不实现完整 DOF workspace diffusion；
- 不用主曲率线作为硬路径生成器；
- 不让真实 UR5 逐关节跟随 GELLO；
- 不在没有 mock/sim 验证前直接实机贴合采集。
