from typing import Union
import abc


class DetectionBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        """
        Everything needed for setup / configuration
        """
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self, frame) -> Union[dict, list]:
        """
        For gestures, should return a dict that contains the results as keys
        and confidence values as values.
        For faces, should return a list of detected bounding boxes.

        The result must have the following structure:
            {gesture name (str): score, ...}  for gestures
            [bounding box (tuple(lx, ly, rx, ry)), ...} for faces
        """
        raise NotImplementedError
