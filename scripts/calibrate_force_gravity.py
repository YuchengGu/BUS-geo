from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from visual_guided_collection_gui.force_gravity import (
    fit_force_gravity_calibration,
    save_force_gravity_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit force-sensor gravity compensation from no-contact TCP poses."
    )
    parser.add_argument("--robot-ip", default="192.168.1.15")
    parser.add_argument("--force-ip", default="192.168.1.100")
    parser.add_argument("--output", default="force_calibration/gravity_force_calib.npz")
    parser.add_argument("--num-poses", type=int, default=12)
    parser.add_argument("--samples-per-pose", type=int, default=80)
    parser.add_argument("--sample-interval-s", type=float, default=0.01)
    parser.add_argument("--raw-log", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import rtde_receive
    from gello.force_sensor_mtcp import ForceSensorMTCP

    receiver = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    force_sensor = ForceSensorMTCP(ip=args.force_ip)
    force_sensor.connect()

    tcp_poses = []
    raw_wrenches = []
    raw_records = []
    print("Gravity calibration: keep the probe in free space with no contact.")
    print("Move to different wrist orientations covering the scan range; press Enter to capture each pose.")
    for index in range(int(args.num_poses)):
        input(f"[{index + 1}/{args.num_poses}] Move to a no-contact pose, then press Enter...")
        pose = np.asarray(receiver.getActualTCPPose(), dtype=float).reshape(6)
        wrench = _read_mean_wrench(
            force_sensor,
            samples=args.samples_per_pose,
            sample_interval_s=args.sample_interval_s,
        )
        tcp_poses.append(pose)
        raw_wrenches.append(wrench)
        raw_records.append(
            {
                "index": index,
                "tcp_pose": pose.tolist(),
                "raw_wrench": wrench.tolist(),
            }
        )
        print(f"  tcp_pose = {pose.tolist()}")
        print(f"  raw_wrench_mean = {wrench.tolist()}")

    calibration = fit_force_gravity_calibration(np.vstack(tcp_poses), np.vstack(raw_wrenches))
    save_force_gravity_calibration(args.output, calibration)

    if args.raw_log:
        raw_path = Path(args.raw_log).expanduser()
    else:
        raw_path = Path(args.output).expanduser().with_suffix(".json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "robot_ip": args.robot_ip,
                "force_ip": args.force_ip,
                "output": str(Path(args.output).expanduser()),
                "force_bias_sensor": calibration.force_bias_sensor.tolist(),
                "gravity_base": calibration.gravity_base.tolist(),
                "torque_bias_sensor": calibration.torque_bias_sensor.tolist(),
                "sample_count": calibration.sample_count,
                "records": raw_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved gravity calibration:", str(Path(args.output).expanduser()))
    print("Saved raw calibration log:", str(raw_path))
    print("force_bias_sensor:", calibration.force_bias_sensor.tolist())
    print("gravity_base:", calibration.gravity_base.tolist())
    print("torque_bias_sensor:", calibration.torque_bias_sensor.tolist())


def _read_mean_wrench(force_sensor, *, samples: int, sample_interval_s: float) -> np.ndarray:
    values = []
    for _ in range(max(1, int(samples))):
        sample = force_sensor.read_values()
        if sample is not None:
            wrench = np.asarray(sample, dtype=float).reshape(-1)
            if wrench.size >= 6 and np.all(np.isfinite(wrench[:6])):
                values.append(wrench[:6].copy())
        if sample_interval_s > 0.0:
            time.sleep(float(sample_interval_s))
    if not values:
        raise RuntimeError("No valid force samples were read")
    return np.mean(np.vstack(values), axis=0)


if __name__ == "__main__":
    main()
