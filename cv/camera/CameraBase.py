import abc


class CameraBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_frame(self):
        pass

    @abc.property
    def frame_width(self):
        pass

    @abc.property
    def frame_height(self):
        pass
