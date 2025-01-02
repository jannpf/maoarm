import abc

class DetectionBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        """
        Everything needed for setup / configuration
        """
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self) -> dict:
        """
        Should return a dict that contains the results as keys 
        and optionally confidence values as values.
        The result can be:
            gesture name (str) for gestures
            bounding box (tuple(x1,y1,x2,y2)) for faces
        """
        raise NotImplementedError
