from .DetectionBase import DetectionBase

import cv2


class CascadeFaces(DetectionBase):
    def __init__(self, filename):
        self.faceCascade = cv2.CascadeClassifier(filename)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces, _, confidence = self.faceCascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )

        result = {}
        for i, (x, y, w, h) in enumerate(faces):
            result[x, y, x+w, y+h] = confidence[i]

        return result
