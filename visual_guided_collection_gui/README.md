# 可视化引导采集 GUI

这个文件夹是第一版 Open3D 单窗口采集界面。它不替换 `experiments/launch_nodes.py`，也不删除原来的 `experiments/run_env.py` 采集方式。

## 启动方式

先启动 UR robot node：

```bash
cd /home/ubuntu22/dev/gello_software
conda activate Newgello
python experiments/launch_nodes.py --robot ur
```

再启动 GUI：

```bash
cd /home/ubuntu22/dev/gello_software
conda activate Newgello
python -m visual_guided_collection_gui.main \
  --point-stride 2 \
  --probe-tip-offset-m 0.0
```

默认会尝试打开超声采集卡 `video5`。如果暂时不接超声：

```bash
python -m visual_guided_collection_gui.main --disable-ultrasound
```

如果超声采集卡不是 `video5`：

```bash
python -m visual_guided_collection_gui.main --ultrasound-index 4
```

如果要指定新的手眼标定结果：

```bash
python -m visual_guided_collection_gui.main \
  --T-tcp-camera hand_eye_calibration/results_0512_222937_calib_11x8_stride10/T_tcp_camera.npy \
  --point-stride 2
```

## 界面流程

1. 点 `连接服务`
   - 连接已启动的 UR ZMQ robot node。
   - 打开 D405。
   - 默认打开超声采集卡。
   - 初始化 GELLO。
   - 初始化力传感器，除非传入 `--disable-force`。

2. 点 `拍照摆位`
   - 进入 GELLO 摆位模式。
   - 这个阶段只移动 UR5，不保存 episode。

3. 点 `拍照冻结`
   - 停止摆位循环。
   - 默认等待 `--capture-settle-s 0.5` 秒，让 UR5 停稳。
   - 冻结当前 D405 RGB-D。
   - 读取当前 UR TCP。
   - 生成 base 坐标系 RGBXYZ 点云。

4. 在左侧 3D 点云里按住 `Shift + 鼠标左键` 点 seed。

5. 点 `规划路径`
   - 分割乳腺区域。
   - 规划路径。
   - 在同一个 3D 视图里显示分割点云、路径线、路径法向。
   - 保存 `raw_cloud_base.ply`、`segmented_breast.ply`、`planned_path.json`、`planning_report.json`。

6. 点 `确认路径`。
   - 这一步会检查设备服务仍然连接。
   - GELLO 必须已经连接；第一版是在 `连接服务` 时初始化 GELLO，并在拍照/规划阶段保持连接。

7. 点 `采集前 GELLO 接管`
   - 重新启动 GELLO 控制循环。
   - 这一步只控制 UR5，不保存 episode。
   - 目的是先让 GELLO 接管稳定，避免第一帧 recording 同时发生接管和保存。

8. 点 `开始记录 episode`
   - 正式进入 GELLO 示教采集。
   - 只有 GELLO 已经接管控制后，按钮才允许开始保存。
   - 每帧保存原始 obs、control、meta。
   - 每帧额外保存路径 residual、路径法向、最近路径点、进度、精扫 flag。

9. 采集中可以点 `结节精扫 flag` 切换 0/1。

10. 点 `结束 episode` 停止当前 episode。

11. 点 `安全停止` 停止循环并释放 D405。

## 第一版边界

- D405 只在 GUI 内打开一次，避免规划脚本和采集脚本抢相机。
- 左侧 3D 视图使用 Open3D `SceneWidget`。
- 右侧显示三路图像预览：`D405 RGB`、`D405 depth` 伪彩色、`Ultrasound`。
- `D405 depth` 预览是按当前帧深度值做百分位归一化后的伪彩色图，只用于观察形状；保存到 pkl 的 `D405_depth` 仍是相机返回的原始 `uint16` 深度图。
- 状态区显示力、力矩、探头尖端位置和当前 control。探头尖端位置使用 `tcp_position_base + probe_tip_offset_m * tcp_z_axis_base`，也就是沿 UR 末端自身 z 轴延长。
- seed 选择是嵌入式投影拾取：`Shift + 鼠标左键` 会从当前屏幕投影中找最近点云点。
- 采集保存语义仍是 `obs_t -> action_t`。
- 原来的 `run_env.py` 没有被删除，仍然可以用原来的终端方式采集。
- `拍照冻结` 只停止 GUI 的 GELLO 控制循环，不物理断开 GELLO 串口；确认路径后需要先点 `采集前 GELLO 接管`，再点 `开始记录 episode`。

## 需要实物确认的参数

`--probe-tip-offset-m` 表示超声探头尖端相对 UR TCP 沿 `tcp_z_axis_base` 的延长距离。默认是 `0.0`，后面需要用实物确认正负号和长度。
