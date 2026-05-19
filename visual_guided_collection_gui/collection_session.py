from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass
from typing import Callable

from gello.utils.control_utils import build_time_alignment_meta

from visual_guided_collection_gui.episode_recorder import EpisodeRecorder


StatusCallback = Callable[[str], None]
SampleCallback = Callable[[dict], None]


@dataclass
class LoopConfig:
    idle_sleep_s: float = 0.001


class TeleopLoop:
    def __init__(self, *, devices, config: LoopConfig | None = None):
        self.devices = devices
        self.config = config or LoopConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_positioning(self, *, on_sample: SampleCallback | None = None, on_status: StatusCallback | None = None) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_positioning,
            kwargs={"on_sample": on_sample, "on_status": on_status},
            daemon=True,
        )
        self._thread.start()

    def start_recording(
        self,
        *,
        recorder: EpisodeRecorder,
        on_sample: SampleCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_recording,
            kwargs={"recorder": recorder, "on_sample": on_sample, "on_status": on_status},
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run_positioning(self, *, on_sample: SampleCallback | None, on_status: StatusCallback | None) -> None:
        try:
            obs = self.devices.get_obs()
            while not self._stop.is_set():
                obs, action, _obs_meta, _action_timing = self.devices.step_agent(obs)
                if on_sample is not None:
                    on_sample({"obs": obs, "action": action})
                time.sleep(self.config.idle_sleep_s)
        except Exception as exc:
            if on_status is not None:
                on_status(f"摆位循环错误: {type(exc).__name__}: {exc}")

    def _run_recording(
        self,
        *,
        recorder: EpisodeRecorder,
        on_sample: SampleCallback | None,
        on_status: StatusCallback | None,
    ) -> None:
        try:
            obs = self.devices.get_obs()
            while not self._stop.is_set():
                current_obs = obs
                obs, action, obs_meta, action_timing = self.devices.step_agent(current_obs)
                timestamp = datetime.datetime.now()
                meta = build_time_alignment_meta(
                    sample_index=recorder.sample_index,
                    episode_id=None if recorder.episode_dir is None else recorder.episode_dir.name,
                    control_loop_hz_config=getattr(self.devices.env, "control_rate_hz", None),
                    wall_time=timestamp,
                    obs_meta=obs_meta,
                    action_timing=action_timing,
                    step_timing=dict(getattr(self.devices.env, "last_step_timing", {}) or {}),
                )
                enriched_obs = recorder.save_sample(current_obs, action, meta=meta, timestamp=timestamp)
                if on_sample is not None:
                    on_sample({"obs": current_obs, "action": action, "enriched_obs": enriched_obs})
                time.sleep(self.config.idle_sleep_s)
        except Exception as exc:
            if on_status is not None:
                on_status(f"记录循环错误: {type(exc).__name__}: {exc}")
