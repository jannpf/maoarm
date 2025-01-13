from .DetectionBase import DetectionBase

import cv2

CONFIDENCE_THRESHOLD = 0.5


class CascadeFaces(DetectionBase):
    def __init__(self, filename):
        self.faceCascade = cv2.CascadeClassifier(filename)

    def detect(self, frame) -> list[tuple]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces, _, confidence = self.faceCascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )

        result = []
        for i, (x, y, w, h) in enumerate(faces):
            if confidence[i] > CONFIDENCE_THRESHOLD:
                box = (x, y, x + w, y + h)
                box_int = tuple(map(lambda x: int(x), box))
                result.append(box_int)

        return result
