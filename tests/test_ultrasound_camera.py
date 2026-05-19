import time
from queue import Empty, Queue

import numpy as np


class FakeCapture:
    def __init__(self):
        self.frames = Queue()
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        try:
            return True, self.frames.get(timeout=0.05)
        except Empty:
            return False, None

    def release(self):
        self.released = True

    def push(self, frame):
        self.frames.put(frame)


def wait_for_frame(camera, frame_id, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if camera.frame_id >= frame_id:
            return
        time.sleep(0.001)
    raise AssertionError(f"timed out waiting for frame {frame_id}")


def test_ultrasound_read_returns_cached_frame_without_waiting_for_next_capture(monkeypatch):
    from gello.cameras import Ultrasound as ultrasound_module

    fake_capture = FakeCapture()
    monkeypatch.setattr(ultrasound_module.cv2, "VideoCapture", lambda _index: fake_capture)

    camera = ultrasound_module.UltrasoundCamera(camera_index=4)
    try:
        frame_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        frame_bgr[:, :, 0] = 10
        frame_bgr[:, :, 1] = 20
        frame_bgr[:, :, 2] = 30
        fake_capture.push(frame_bgr)
        wait_for_frame(camera, 1)

        t0 = time.monotonic()
        rgb1, depth1 = camera.read()
        first_read_ms = (time.monotonic() - t0) * 1000.0

        t1 = time.monotonic()
        rgb2, depth2 = camera.read()
        second_read_ms = (time.monotonic() - t1) * 1000.0

        assert first_read_ms < 10.0
        assert second_read_ms < 10.0
        assert camera.last_metadata["valid"] is True
        assert camera.last_metadata["frame_new"] is False
        assert camera.last_metadata["frame_id"] == 1
        np.testing.assert_array_equal(rgb1, rgb2)
        np.testing.assert_array_equal(rgb1[0, 0], np.array([30, 20, 10], dtype=np.uint8))
        assert depth1.shape == (2, 3, 1)
        assert depth2.shape == (2, 3, 1)
    finally:
        camera.close()
        assert fake_capture.released is True
