# BUS-geo

BUS-geo is a robotic breast ultrasound research codebase built on top of
GELLO. It combines visual-guided data collection, breast surface path planning,
geodesic path variants, Bayesian optimization scan utilities, UR robot control,
and NDI-recorded trajectory replay.

The original GELLO teleoperation framework remains in this repository as the
robot interface foundation. The added project-specific code focuses on robotic
breast ultrasound workflows.

## Main Components

- `visual_guided_collection_gui/`: Open3D-based GUI for RGB-D capture, breast
  surface path planning, GELLO/UR control, ultrasound preview, force telemetry,
  and episode recording.
- `breast_path_planning/`: point-cloud processing, breast segmentation, path
  planning, geodesic paths, smoothing, and visualization utilities.
- `ndi_relative_replay/`: tools for replaying relative NDI marker trajectories
  from `ndi_pose_native.csv` on the current UR TCP frame.
- `gello/`: robot interfaces and ZMQ robot node infrastructure inherited from
  GELLO, including UR support.
- `experiments/`: launch scripts for robot nodes and existing GELLO workflows.
- `tests/`: regression tests for GUI logic, planning, UR control, NDI replay,
  force/telemetry, and data schemas.

## Installation

The current development environment uses Python 3.11.

For a fresh clone:

```bash
git clone https://github.com/YuchengGu/BUS-geo.git
cd BUS-geo
git submodule init
git submodule update
```

## Visual-Guided Collection GUI
 see demo.ipynb

## NDI Relative Replay

The replay tools read marker poses from `ndi_relative_replay/ndi_pose_native.csv`
and treat the current UR TCP pose as frame 0. The default mode applies the
hand-eye conjugation from the calibrated flange-to-marker transform.

Preview generated TCP targets without commanding the robot:

```bash
python ndi_relative_replay/replay_relative_ndi.py --max-frames 80
```

Visualize the TCP trajectory in an Open3D window:

```bash
python ndi_relative_replay/preview_tcp_trajectory_gui.py --max-frames 80
```

Execute on the real robot only after checking the preview:

```bash
python ndi_relative_replay/replay_relative_ndi.py --execute
```

## Hardware Used In The Main Workflow

- Universal Robots arm, currently launched through `--robot ur`.
- Intel RealSense D405 for RGB-D surface capture.
- Ultrasound capture device, optional during GUI development.
- Force sensor, optional depending on launch arguments and hardware state.
- GELLO teleoperation device for manual positioning and episode recording.
- NDI optical tracker and marker for relative trajectory replay experiments.

## Notes

- This repository is derived from GELLO, but the active project target is
  robotic breast ultrasound data collection and path planning.
- Some scripts require connected hardware and will block or fail if the UR node,
  D405, ultrasound device, force sensor, or NDI tracker is not available.
- Real-robot replay commands should be tested in preview mode first.
