from .CameraBase import CameraBase
import pyrealsense2 as rs
import cv2
import numpy as np


class RealsenseCamera(CameraBase):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.width = 640
        self.height = 480
        self.framerate = 30  # fps

        self.config.disable_all_streams()
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.framerate
        )

        self.pipeline.start(self.config)

    def get_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        width, height = frame.get_width(), frame.get_height()
        if width != self.width or height != self.height:
            raise ValueError("Mismatch in Frame Dimensions!")
        frame = np.asanyarray(frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def frame_width(self) -> int:
        return int(self.width)

    def frame_height(self) -> int:
        return self.height
