import numpy as np
from typing import Optional, Tuple
import time

# 引入奥比中光官方 SDK
from pyorbbecsdk import *
from gello.cameras.camera import CameraDriver

class OrbbecCamera(CameraDriver):
    def __init__(self):
        print("正在启动 Orbbec (奥比中光) 相机...")
        self.pipeline = Pipeline()
        config = Config()

        try:
            # 1. 配置彩色流：强制要求 RGB 格式，直接适配 Gello，绝不偏色！
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
            config.enable_stream(color_profile)
            
            # 2. 配置深度流：获取默认配置
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            
            # 3. 强制要求：必须彩色和深度同时准备好，才算一帧 (时间同步)
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            print(f"❌ 奥比中光流配置失败: {e}")
            return

        # 尝试开启底层硬件同步 (部分型号支持)
        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            pass # 如果不支持硬件同步就忽略

        # 启动相机
        self.pipeline.start(config)
        
        # 创建对齐过滤器：将深度图对齐到彩色图视角 (D2C，空间同步)
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        print("等待奥比中光预热 2 秒钟...")
        time.sleep(2.0)

        # 4. 获取第一帧，初始化安全缓存 (防卡死机制)
        print("正在获取奥比中光第一帧画面...")
        for _ in range(20):
            frames = self.pipeline.wait_for_frames(1000)
            if not frames:
                continue
            
            # 进行空间对齐处理
            frames = self.align_filter.process(frames)
            if not frames:
                continue
                
            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                self._update_cache(color_frame, depth_frame)
                break
                
        print("✅ Orbbec 相机准备就绪，已开启 100Hz 兼容非阻塞模式！")

    def _update_cache(self, color_frame, depth_frame):
        """将底层 Buffer 转换为 Numpy 数组并更新缓存"""
        # 提取彩色图：因为上面配置了 OBFormat.RGB，这里提取出来直接就是完美的 RGB 数组
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        self.last_color = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        
        # 提取深度图：16位毫米级数据
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_2d = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
        # 增加通道维度以匹配 Gello (H, W, 1)
        self.last_depth = depth_2d[:, :, None]
        self.last_frame_mono_ns = time.monotonic_ns()
        self.last_frame_id = self._read_frame_attr(color_frame, "get_frame_number")
        self.last_hardware_timestamp_ms = self._read_frame_attr(color_frame, "get_timestamp")

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
            # 【核心护城河】：只等 10ms！如果有新图就拿，没有就直接抛出异常，绝不卡主循环！
            frames = self.pipeline.wait_for_frames(10)
            
            if frames:
                # 空间对齐
                frames = self.align_filter.process(frames)
                if frames:
                    frames = frames.as_frame_set()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if color_frame and depth_frame:
                        self._update_cache(color_frame, depth_frame)
                        frame_new = True
        except Exception as e:
            # 捕获 10ms 超时异常。跳过更新，直接返回老照片
            error = str(e)
            
        read_end = time.monotonic_ns()
        last_frame_mono_ns = getattr(self, "last_frame_mono_ns", read_end)
        self.last_metadata = {
            "read_start_mono_ns": read_start,
            "read_end_mono_ns": read_end,
            "valid": valid,
            "frame_new": frame_new,
            "frame_id": getattr(self, "last_frame_id", None),
            "hardware_timestamp_ms": getattr(self, "last_hardware_timestamp_ms", None),
            "cache_age_ms": (read_end - last_frame_mono_ns) / 1_000_000.0,
            "error": error,
        }
        return self.last_color, self.last_depth
