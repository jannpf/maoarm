from .DetectionBase import DetectionBase

import cv2


class CaffeFaces(DetectionBase):
    def __init__(self, modelpath, configpath):
        self.net = cv2.dnn.readNetFromCaffe(configpath, modelpath)

    def detect(self, frame) -> list[int]:
        # TODO: check if works without resizing
        # width = int(width)
        # height = int(height)
        h, w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (width, height))
        blob = cv2.dnn.blobFromImage(
            resized_frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        result = {}
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1,y1,x2,y2) = box.astype("int")
                result[(x1,y1,x2,y2)] = confidence

        return result
