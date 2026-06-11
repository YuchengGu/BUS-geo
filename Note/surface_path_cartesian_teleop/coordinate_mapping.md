# Surface Cartesian Darboux 坐标映射

## 结论

当前 `visual_guided_collection_gui` 的 Surface Cartesian Darboux 控制中有三套坐标系：

```text
1. GELLO 标定后输入坐标系
2. 曲面路径 TNB 坐标系
3. UR5 TCP 末端坐标系
```

当前控制语义是：

```text
GELLO +x 平移 -> UR5 tip 沿路径 t 前进
GELLO +y 平移 -> UR5 tip 沿 -b 横向移动
GELLO +z 平移 -> UR5 tip 沿 +n 离开曲面

GELLO 绕 x 转 -> UR5 TCP 绕自身局部 y 转
GELLO 绕 y 转 -> UR5 TCP 绕自身局部 x 转
GELLO 绕 z 转 -> UR5 TCP 绕自身局部 -z 转

UR5 TCP z 轴 -> -n
UR5 TCP x 轴 -> 默认取世界 Y 轴在切平面内的投影，不是路径切向 t
```

对应源码主要在：

```text
visual_guided_collection_gui/surface_teleop.py
```

## 1. GELLO 标定后输入坐标系

### Set neutral

点击 `Set neutral` 时，当前 GELLO TCP 位姿会被记录为零点：

```cpp
neutral_gello_pose
```

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:93-104
```

含义：

```text
后续所有 GELLO 平移和旋转，都不是用世界固定零点，而是相对于这一次 neutral 的变化量。
```

### Calibrate +X

点击 `Calibrate +X` 时，代码取：

\[
e_x^{gello}
=
\frac{p_{gello,x}-p_{gello,neutral}}
{\|p_{gello,x}-p_{gello,neutral}\|}
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:107-114
```

所以 `+X` 不是 GELLO 机械末端原始 x 轴，而是你从 neutral 推出去的那个方向。

### Calibrate +Z

点击 `Calibrate +Z` 时，代码先取：

\[
z_{raw}
=
\frac{p_{gello,z}-p_{gello,neutral}}
{\|p_{gello,z}-p_{gello,neutral}\|}
\]

然后把它相对 \(e_x^{gello}\) 正交化：

\[
e_z^{gello}
=
\operatorname{normalize}
\left(
z_{raw}
-
(z_{raw}\cdot e_x^{gello})e_x^{gello}
\right)
\]

再自动生成：

\[
e_y^{gello}
=
e_z^{gello}\times e_x^{gello}
\]

最后输入坐标系为：

\[
F_{gello}
=
[e_x^{gello}, e_y^{gello}, e_z^{gello}]
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:116-127
```

注意：

```text
GELLO 标定后坐标系是通过两次平移标定出来的操作坐标系。
它不是 GELLO 末端模型自带的原始局部坐标系。
```

## 2. 曲面路径 TNB 坐标系

路径上每个位置都有：

\[
(t,n,b)
\]

其中：

```text
t = 路径切向，沿规划路径前进方向
n = 曲面外法向
b = t × n
```

代码实现：

```text
visual_guided_collection_gui/surface_teleop.py:283-293
```

代码返回：

\[
F_{tnb}
=
[t,n,b]
\]

实现细节：

```text
1. 先归一化 normal 得到 n。
2. 把 tangent 投影到 n 的切平面内，得到 t。
3. 用 b = t × n 得到横向轴。
4. 再用 t = n × b 做一次正交修正。
```

## 3. GELLO 平移到曲面运动的映射

代码先把 GELLO 当前平移相对 neutral 的变化量投影到标定输入坐标系：

\[
\Delta p_{gello}^{calib}
=
F_{gello}^{T}
\left(
p_{gello}-p_{gello,neutral}
\right)
\]

再乘平移增益：

\[
[\Delta x,\Delta y,\Delta z]
=
K_p\Delta p_{gello}^{calib}
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:152-156
```

对应关系：

```text
GELLO +x -> 路径弧长进度 S 增加
GELLO +y -> lateral_offset 减小，也就是沿 -b
GELLO +z -> normal_offset 增加，也就是沿 +n
```

数学形式：

\[
S
=
S_0+k_x\Delta x
\]

\[
d_b
=
d_{b,0}-k_y\Delta y
\]

\[
d_n
=
d_{n,0}+k_z\Delta z
\]

路径采样：

\[
\Gamma(S),\quad t(S),\quad n(S),\quad b(S)
\]

目标探头 tip：

\[
p_{tip}
=
\Gamma(S)
+d_b b(S)
+d_n n(S)
\]

源码中的辅助函数也给出同样方向关系：

```text
visual_guided_collection_gui/surface_teleop.py:410-420
```

```python
return delta[0] * t - delta[1] * b + delta[2] * n
```

## 4. UR5 TCP 坐标系和 TNB 的关系

当前默认参考 TCP 姿态不是：

```text
TCP x = t
TCP y = b
TCP z = -n
```

而是：

```text
TCP z = -n
TCP x = 世界 Y 轴投影到当前切平面后的方向
TCP y = 由 x,z 叉乘补出来
```

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:295-314
visual_guided_collection_gui/surface_teleop.py:317-337
```

数学上：

\[
z_{tcp}
=
-n
\]

设世界 Y 轴为：

\[
e_y^{world}=[0,1,0]^T
\]

则：

\[
x_{tcp}
=
\operatorname{normalize}
\left(
e_y^{world}
-
(e_y^{world}\cdot n)n
\right)
\]

如果模式是 `-world-y`：

\[
x_{tcp}
=
-
\operatorname{normalize}
\left(
e_y^{world}
-
(e_y^{world}\cdot n)n
\right)
\]

然后：

\[
y_{tcp}
=
\operatorname{normalize}
\left(
z_{tcp}\times x_{tcp}
\right)
\]

最终：

\[
R_{ref}
=
[x_{tcp},y_{tcp},z_{tcp}]
\]

这样做的目的：

```text
避免 TCP x 轴严格跟随路径切向 t。
如果 x=t，蛇形路径折返时 TCP 可能在拐角附近突然转 180 度。
当前默认让 TCP x 轴尽量保持在“世界 Y 投影到切平面”的方向附近，更稳定。
```

## 5. 探头 tip 和 UR5 TCP 的位置关系

目标 tip 是：

\[
p_{tip}
=
\Gamma(S)
+d_b b(S)
+d_n n(S)
\]

UR5 TCP 位置是：

\[
p_{tcp}
=
p_{tip}
+L_{probe}n(S)
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:447-460
```

代码：

```python
tcp_position = probe_tip + probe_length_m * n
```

因为当前参考姿态有：

\[
z_{tcp}=-n
\]

所以 TCP 在 tip 的外法向一侧。

## 6. GELLO 旋转到 UR5 TCP 旋转的映射

代码先计算 GELLO 当前姿态相对于 neutral 的旋转：

\[
R_{\Delta}^{gello}
=
R_{gello}
R_{gello,neutral}^{T}
\]

再转成旋转向量，并投影到 GELLO 标定坐标系：

\[
\Delta\theta_{gello}^{calib}
=
F_{gello}^{T}
\operatorname{Log}
\left(
R_{\Delta}^{gello}
\right)
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:173-178
```

然后映射成 TCP 局部旋转：

\[
[\Delta\theta_{tcp,x},\Delta\theta_{tcp,y},\Delta\theta_{tcp,z}]
=
[
\Delta\theta_{gello,y},
\Delta\theta_{gello,x},
-\Delta\theta_{gello,z}
]
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:423-430
```

代码：

```python
return np.array([delta[1], delta[0], -delta[2]], dtype=float)
```

所以当前旋转语义是：

```text
GELLO 绕标定后 x 轴转 -> UR5 TCP 绕自身局部 y 轴转
GELLO 绕标定后 y 轴转 -> UR5 TCP 绕自身局部 x 轴转
GELLO 绕标定后 z 轴转 -> UR5 TCP 绕自身局部 -z 轴转
```

姿态最终更新：

\[
R_{cmd}
=
R_{ref}
R_{residual}
\exp
\left(
\widehat{\Delta\theta_{tcp}}
\right)
\]

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:181
```

其中：

```text
R_ref:
    当前路径点和曲面法向给出的参考 TCP 姿态。

R_residual:
    recenter/clutch 时记录的人工姿态残差。

exp(hat(delta_theta_tcp)):
    GELLO 当前旋转输入映射后的 TCP 局部微调姿态。
```

## 7. Clutch / Recenter 对映射的影响

当 clutch 开启时：

```text
controller.update() 不继续按 GELLO 偏移驱动 UR5。
它会调用 recenter，把当前 GELLO 位姿重新作为 neutral，并同步当前 UR5 TCP pose。
```

代码位置：

```text
visual_guided_collection_gui/surface_teleop.py:146-147
visual_guided_collection_gui/surface_teleop.py:129-136
```

这意味着：

```text
clutch 后的下一次控制，不会强制回到旧 Darboux 参考姿态。
它会以当前 UR5 TCP pose 为新的参考残差，继续局部控制。
```

这也是为什么每个随机 local episode 前后可以重新 Set neutral / Calibrate +X / Calibrate +Z。

## 8. 总表

### 平移

| GELLO 标定后输入 | UR5 tip 运动 | 曲面坐标解释 |
|---|---|---|
| +x | 沿路径前进 | \(+t\), 增加 \(S\) |
| +y | 横向负方向 | \(-b\) |
| +z | 离开曲面 | \(+n\) |

### 旋转

| GELLO 标定后旋转 | UR5 TCP 局部旋转 |
|---|---|
| 绕 x | 绕 TCP local y |
| 绕 y | 绕 TCP local x |
| 绕 z | 绕 TCP local -z |

### 参考姿态

| 轴 | 当前默认含义 |
|---|---|
| TCP z | \(-n\) |
| TCP x | 世界 Y 在切平面内的投影，或其反向 |
| TCP y | 由 \(z_{tcp}\times x_{tcp}\) 补成右手系 |

