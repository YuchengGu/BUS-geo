# 从 GELLO pkl 做手腕相机手眼标定

这个文件夹用于从 GELLO 采集出来的 `.pkl` 数据段中离线计算手腕相机的手眼外参。当前脚本支持：

```text
D405
Orbbec
```

目标输出是：

```text
T_tcp_camera
```

它表示 **相机坐标系到 UR TCP 坐标系** 的 4x4 变换矩阵。之后可以用于：

```text
P_base = T_base_tcp @ T_tcp_camera @ P_camera
```

也就是把 depth 反投影出来的点云、路径点、法向量从相机坐标系转换到机械臂 base/world 坐标系。

## 文件说明

### `hand_eye_from_pkl.py`

主标定脚本。

它会从一个 pkl 数据段目录中读取每一帧：

```text
<camera_name>_rgb，例如 D405_rgb 或 Orbbec_rgb
ee_pos_rotvec
meta
```

然后做这些事：

1. 过滤掉相机缓存帧，只使用 `frame_new=True` 的帧。
2. 从 `<camera_name>_rgb` 自动检测棋盘格角点。
3. 从 `ee_pos_rotvec` 计算 `T_base_tcp`。
4. 用棋盘格角点估计每帧的 `T_camera_board`。
5. 调用 OpenCV `cv2.calibrateHandEye()` 计算 `T_tcp_camera`。
6. 输出外参矩阵和验证报告。

### `.gitignore`

忽略运行脚本产生的临时文件和结果目录：

```text
__pycache__/
results*/
```

也就是说，`results_...` 这类标定输出默认不会进入 git。

### `results_0512_210114_detect/`

这是我用你的数据：

```text
/home/ubuntu22/bc_data/gello/0512_210114/
```

跑 `--detect-only` 生成的检测结果目录。它只是检测棋盘格是否能被识别，还没有计算最终外参，因为正式标定还需要你提供棋盘格小方格的真实边长。

这个目录里通常会有：

```text
detected_chessboards.jpg
selected_frames.json
run_config.json
```

这些是脚本输出，不是源码。

## 输出文件说明

### `detected_chessboards.jpg`

棋盘格检测可视化图。

脚本会把若干帧拼成一张图，并把检测到的角点画在棋盘格上。你可以用它确认：

- 棋盘格是否检测正确；
- 角点有没有偏；
- 用的 `--board-cols` / `--board-rows` 是否正确；
- 标定板在图像里是否清晰。

### `selected_frames.json`

记录本次标定/检测用到了哪些 pkl 帧。

每一项包含：

```text
path
sample_index
frame_id
```

它用于追溯：最终外参到底是由哪些原始 pkl 算出来的。

### `run_config.json`

记录本次运行的参数，比如：

```text
episode_dir
board_cols
board_rows
square_size_m
stride
max_frames
require_new_frame
```

它用于保证标定结果可复现。

### `T_tcp_camera.npy`

正式标定后输出的核心结果。

这是一个 4x4 numpy 矩阵：

```text
T_tcp_camera = ^tcp T_camera
```

也就是相机坐标系到 TCP 坐标系的变换。

只有在提供 `--square-size-m` 并完成正式标定时才会生成。

### `camera_intrinsics.npz`

正式标定时估计出的相机内参和畸变参数：

```text
camera_matrix
dist_coeffs
```

当前脚本会从检测到的棋盘格图像中估计内参。后续也可以改成读取相机 SDK 给出的 RGB 内参。

### `calibration_report.json`

机器可读的标定报告。

包含：

- `T_tcp_camera`
- `camera_matrix`
- `dist_coeffs`
- 重投影误差；
- 标定板在 base 下的一致性误差；
- 本次 hand-eye validation 的统计结果。

### `calibration_report.md`

人可读的标定报告。

里面会把 `T_tcp_camera` 和主要验证指标打印出来，方便你直接查看。

## 目前这批数据是否能用

我已经用你的数据段：

```text
/home/ubuntu22/bc_data/gello/0512_210114/
```

做了初步检测。

你的棋盘格边界是：

```text
黑色 5 个，白色 4 个
```

所以每条边一共 9 个方格，对应：

```text
8 x 8 内角点
```

检测命令：

```bash
/home/ubuntu22/dev/anaconda3/envs/Newgello/bin/python hand_eye_calibration/hand_eye_from_pkl.py \
  --episode-dir /home/ubuntu22/bc_data/gello/0512_210114 \
  --output-dir hand_eye_calibration/results_0512_210114_detect \
  --camera-name D405 \
  --board-cols 8 \
  --board-rows 8 \
  --max-frames 40 \
  --detect-only
```

结果：

```text
检测到棋盘格：40 帧。
```

说明这批 pkl 的 RGB 图像可以自动检测出棋盘格。

## 正式标定命令

正式标定需要你量一下棋盘格**每个小方格的真实边长**，单位用米。

例如，如果一个小方格边长是 10mm：

```text
10mm = 0.010m
```

运行：

```bash
/home/ubuntu22/dev/anaconda3/envs/Newgello/bin/python hand_eye_calibration/hand_eye_from_pkl.py \
  --episode-dir /home/ubuntu22/bc_data/gello/0512_210114 \
  --output-dir hand_eye_calibration/results_0512_210114 \
  --camera-name D405 \
  --board-cols 8 \
  --board-rows 8 \
  --square-size-m 0.010 \
  --max-frames 40
```

把 `0.010` 换成你真实测量的小格边长。

## Orbbec 重新标定

Orbbec 使用同一个脚本，只需要把 `--camera-name` 改成 `Orbbec`，并把输出目录放到类似 `Results_Orbbec_YYYYMMDD` 的结果文件夹。

先做检测：

```bash
/home/ubuntu22/dev/anaconda3/envs/Newgello/bin/python hand_eye_calibration/hand_eye_from_pkl.py \
  --episode-dir /home/ubuntu22/bc_data/gello/<你的Orbbec标定episode> \
  --output-dir hand_eye_calibration/Results_Orbbec_20260529_detect \
  --camera-name Orbbec \
  --board-cols 8 \
  --board-rows 8 \
  --max-frames 40 \
  --detect-only
```

确认 `detected_chessboards.jpg` 里的角点检测正确后，正式标定：

```bash
/home/ubuntu22/dev/anaconda3/envs/Newgello/bin/python hand_eye_calibration/hand_eye_from_pkl.py \
  --episode-dir /home/ubuntu22/bc_data/gello/<你的Orbbec标定episode> \
  --output-dir hand_eye_calibration/Results_Orbbec_20260529 \
  --camera-name Orbbec \
  --board-cols 8 \
  --board-rows 8 \
  --square-size-m 0.010 \
  --max-frames 40
```

输出结果仍然是：

```text
hand_eye_calibration/Results_Orbbec_20260529/T_tcp_camera.npy
```

GUI 可以直接指定这个新外参：

```bash
python -m visual_guided_collection_gui.main \
  --wrist-camera Orbbec \
  --t-tcp-camera hand_eye_calibration/Results_Orbbec_20260529/T_tcp_camera.npy
```

## 注意事项

- 不要用普通扫查数据做标定，必须是标定板固定且相机能看到标定板的数据。
- 默认只使用 `frame_new=True` 的相机帧，避免图像是缓存帧而 TCP pose 是当前帧造成错配。
- D405 的 `T_tcp_camera.npy` 不能给 Orbbec 用；相机拆装后也应该重新标定。
- 棋盘格参数 `--board-cols` 和 `--board-rows` 指的是**内角点数量**，不是方格数量。
- 如果棋盘格每边 9 个方格，那么内角点就是 8。
- `--square-size-m` 必须是真实小方格边长，否则外参平移尺度会错。
- 输出的 `T_tcp_camera` 需要看验证报告，不能只看矩阵是否生成。
