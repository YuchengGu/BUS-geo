import pyrealsense2 as rs
import numpy as np
from typing import Optional, Tuple
import time

class RealSenseD405:
    def __init__(self):
        print("正在启动 D405 相机...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        print("等待相机预热 2 秒钟...")
        time.sleep(2.0)
        
        print("正在获取第一帧画面...")
        # 初始化时，耐心等它吐出完美的双图套餐
        for _ in range(20): 
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if color_frame and depth_frame:
                    self.last_color_frame = color_frame
                    self.last_depth_frame = depth_frame
                    break
            except RuntimeError:
                continue
                
        self.last_color = np.asanyarray(color_frame.get_data())
        self.last_depth = np.asanyarray(depth_frame.get_data())
        if len(self.last_depth.shape) == 2:
            self.last_depth = self.last_depth[:, :, None]
        self.last_frame_mono_ns = time.monotonic_ns()
        self.last_frame_id = self._read_frame_attr(color_frame, "get_frame_number")
        self.last_hardware_timestamp_ms = self._read_frame_attr(color_frame, "get_timestamp")
        self.last_metadata = {
            "valid": True,
            "frame_new": True,
            "frame_id": self.last_frame_id,
            "hardware_timestamp_ms": self.last_hardware_timestamp_ms,
            "cache_age_ms": 0.0,
            "error": None,
        }
            
        print("✅ D405 准备就绪，已开启严谨防抖轮询模式！")

    @staticmethod
    def _read_frame_attr(frame, attr_name):
        try:
            return getattr(frame, attr_name)()
        except Exception:
            return None

    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        read_start = time.monotonic_ns()
        frame_new = False
        valid = True
        error = None
        try:
            # 【终极修复】：强制使用 wait_for_frames，它保证吐出来的绝对是完整的 frameset！
            # 设定 10ms 超时。因为 D405 是 30帧 (33ms一帧)，所以大部分时间它都会超时报错。
            # 这是极其正常的！正好符合我们 100Hz 的设计！
            frames = self.pipeline.wait_for_frames(10)
            
            # 只要没超时，走到这里，frames 绝对是完美的套餐！放心大胆地对齐！
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if color_frame and depth_frame:
                self.last_color_frame = color_frame
                self.last_depth_frame = depth_frame
                self.last_color = np.asanyarray(color_frame.get_data())
                
                depth = np.asanyarray(depth_frame.get_data())
                if len(depth.shape) == 2:
                    depth = depth[:, :, None]
                self.last_depth = depth
                self.last_frame_mono_ns = time.monotonic_ns()
                self.last_frame_id = self._read_frame_attr(color_frame, "get_frame_number")
                self.last_hardware_timestamp_ms = self._read_frame_attr(color_frame, "get_timestamp")
                frame_new = True

        except RuntimeError as e:
            # 拿不到新图（超时），或者数据残缺，直接静默忽略，返回老缓存！
            error = str(e)
        except Exception as e:
            # 捕获其他意外，防止系统崩溃
            valid = False
            error = str(e)

        read_end = time.monotonic_ns()
        cache_age_ms = (read_end - self.last_frame_mono_ns) / 1_000_000.0
        self.last_metadata = {
            "read_start_mono_ns": read_start,
            "read_end_mono_ns": read_end,
            "valid": valid,
            "frame_new": frame_new,
            "frame_id": self.last_frame_id,
            "hardware_timestamp_ms": self.last_hardware_timestamp_ms,
            "cache_age_ms": cache_age_ms,
            "error": error,
        }
        return self.last_color, self.last_depth

    def latest_frames(self):
        """Return the latest RealSense color/depth frame objects for SDK pointcloud generation."""
        return self.last_color_frame, self.last_depth_frame
