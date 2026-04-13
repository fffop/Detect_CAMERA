from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.ppx],
                [0.0, self.fy, self.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class RGBDFrame:
    frame_index: int
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics

    @property
    def color_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.color_bgr, cv2.COLOR_BGR2RGB)


class AlignedRealSenseCapture:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str | None = None,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if serial:
            self._config.enable_device(serial)
        self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(self._config)
        self._align = rs.align(rs.stream.color)
        self._frame_index = 0

        device = self._profile.get_device()
        self._depth_scale = 0.001
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
            if sensor.is_depth_sensor():
                self._depth_scale = float(sensor.as_depth_sensor().get_depth_scale())

        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_stream.get_intrinsics()
        self._width = int(color_intrinsics.width)
        self._height = int(color_intrinsics.height)
        self._fps = float(color_stream.fps())
        self._intrinsics = CameraIntrinsics(
            width=self._width,
            height=self._height,
            fx=float(color_intrinsics.fx),
            fy=float(color_intrinsics.fy),
            ppx=float(color_intrinsics.ppx),
            ppy=float(color_intrinsics.ppy),
            depth_scale=self._depth_scale,
        )

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, RGBDFrame | None]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self._align.process(frames)
        except RuntimeError:
            return False, None

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return False, None

        color_bgr = np.asarray(color_frame.get_data(), dtype=np.uint8).copy(order="C")
        depth_raw = np.asarray(depth_frame.get_data(), dtype=np.uint16).copy(order="C")
        depth_m = depth_raw.astype(np.float32) * self._depth_scale

        payload = RGBDFrame(
            frame_index=self._frame_index,
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=self._intrinsics,
        )
        self._frame_index += 1
        return True, payload

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass
        self._align = None
        self._profile = None
        self._config = None
        self._pipeline = None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._frame_index)
        return 0.0
