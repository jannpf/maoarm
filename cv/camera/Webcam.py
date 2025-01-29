from .CameraBase import CameraBase
import cv2
import numpy as np


class Webcam(CameraBase):
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        )
        self.video_capture.isOpened()

    def get_frame(self) -> np.ndarray:
        ret, frame = self.video_capture.read()
        if not ret:
            raise ValueError('Failed to capture frame.')

        return frame

    def frame_width(self) -> float:
        return self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    def frame_height(self) -> float:
        return self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
