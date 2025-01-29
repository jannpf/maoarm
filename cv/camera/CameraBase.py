import abc
import numpy as np


class CameraBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_frame(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def frame_width(self) -> int:
        pass

    @abc.abstractmethod
    def frame_height(self) -> int:
        pass
