from .DetectionBase import DetectionBase

import cv2

CONFIDENCE_THRESHOLD = 0.5


class CaffeFaces(DetectionBase):
    def __init__(self, modelpath, configpath):
        self.net = cv2.dnn.readNetFromCaffe(configpath, modelpath)

    def detect(self, frame) -> list[tuple]:
        # TODO: check if works without resizing
        # width = int(width)
        # height = int(height)
        h, w = frame.shape[:2]
        # resized_frame = cv2.resize(frame, (width, height))
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        result = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                box = box.astype("int")
                result.append(box)

        return result
