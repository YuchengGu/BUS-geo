# import cv2
# import numpy as np
# from typing import Optional, Tuple
# from gello.cameras.camera import CameraDriver

# class UltrasoundCamera(CameraDriver):
#     def __init__(self, camera_index=4): # ⚠️ 这里的 2 替换成你查到的 video 编号
#         print(f"正在启动超声采集卡 (video{camera_index})...")
#         self.cap = cv2.VideoCapture(camera_index)
        
#         # 很多采集卡支持 1080p 或 720p，可以尝试强制设置分辨率
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

#         # 设置缓冲极小，保证实时性，防止拿到旧图像
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         if not self.cap.isOpened():
#             print(f"❌ 无法打开超声采集卡 video{camera_index}！")
#         else:
#             print("✅ 超声采集卡启动成功！")
            
#         # 准备一个全黑的深度图占位符 (超声只有二维图像，没有深度)
#         self.dummy_depth = None

#     def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
#         # 从采集卡抓取一帧图像
#         ret, frame = self.cap.read()
        
#         if not ret or frame is None:
#             # 如果没拿到，返回全黑图像防崩溃
#             return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 1), dtype=np.uint16)
        
#         # OpenCV 默认读出来是 BGR，转换为 Gello 需要的 RGB 格式
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # 缩放 (如果需要)
#         if img_size is not None:
#             frame_rgb = cv2.resize(frame_rgb, img_size)

#         # 超声没有深度，但我们必须遵守协议返回 Tuple，所以给它一个全 0 的深度图
#         if self.dummy_depth is None or self.dummy_depth.shape[:2] != frame_rgb.shape[:2]:
#             self.dummy_depth = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 1), dtype=np.uint16)

#         return frame_rgb, self.dummy_depth
    

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple
from gello.cameras.camera import CameraDriver

class UltrasoundCamera(CameraDriver):
    def __init__(self, camera_index=4): # ⚠️ 你的 video 编号
        print(f"正在启动超声采集卡 (video{camera_index})...")
        self.cap = cv2.VideoCapture(camera_index)
        
        # 强制请求 720p 分辨率 (1280x720)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"❌ 无法打开超声采集卡 video{camera_index}！")
        else:
            print("✅ 超声采集卡启动成功！")
            
        self.dummy_depth = None
        self.frame_id = 0
        self.last_frame_mono_ns = None
        self.last_rgb = None
        self._last_returned_frame_id = None
        self._last_error = "not read yet"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread = None
        self.last_metadata = {
            "valid": False,
            "frame_new": False,
            "frame_id": None,
            "cache_age_ms": None,
            "error": "not read yet",
        }
        if self.cap.isOpened():
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            read_end = time.monotonic_ns()
            if not ret or frame is None:
                with self._lock:
                    self._last_error = "cap.read failed"
                time.sleep(0.001)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self.frame_id += 1
                self.last_rgb = frame_rgb
                self.last_frame_mono_ns = read_end
                self._last_error = None

    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        read_start = time.monotonic_ns()
        with self._lock:
            frame_rgb = None if self.last_rgb is None else self.last_rgb.copy()
            frame_id = self.frame_id
            frame_mono_ns = self.last_frame_mono_ns
            error = self._last_error

        if frame_rgb is None:
            read_end = time.monotonic_ns()
            self.last_metadata = {
                "read_start_mono_ns": read_start,
                "read_end_mono_ns": read_end,
                "valid": False,
                "frame_new": False,
                "frame_id": None,
                "cache_age_ms": None,
                "error": error or "no cached frame",
            }
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 1), dtype=np.uint16)

        if img_size is not None:
            frame_rgb = cv2.resize(frame_rgb, img_size)

        if self.dummy_depth is None or self.dummy_depth.shape[:2] != frame_rgb.shape[:2]:
            self.dummy_depth = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 1), dtype=np.uint16)

        read_end = time.monotonic_ns()
        frame_new = frame_id != self._last_returned_frame_id
        self._last_returned_frame_id = frame_id
        cache_age_ms = None
        if frame_mono_ns is not None:
            cache_age_ms = (read_end - frame_mono_ns) / 1_000_000.0
        self.last_metadata = {
            "read_start_mono_ns": read_start,
            "read_end_mono_ns": read_end,
            "valid": True,
            "frame_new": frame_new,
            "frame_id": frame_id,
            "cache_age_ms": cache_age_ms,
            "error": error,
        }
        return frame_rgb, self.dummy_depth

    def close(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
