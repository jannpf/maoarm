import os
import time
from multiprocessing.connection import Client

import cv2
import pyrealsense2 as rs
import numpy as np

from .face import Face
from . import detection  # CaffeFaces, MediapipeGestures


def box_size(box: tuple[int, int, int, int]) -> float:
    lx, ly, rx, ry = box
    return (rx - lx) * (ry - ly)


# video capture setup
# video_capture = cv2.VideoCapture(-1)
# video_capture.set(cv2.CAP_PROP_FOURCC,
#                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# video_capture.isOpened()

# width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

pipeline = rs.pipeline()
config = rs.config()

config.disable_all_streams()
config.enable_stream(rs.stream.color, 640,480, rs.format.rgb8, 30)

pipeline.start(config)

# IP address to communicate data with
conn = Client(("localhost", 6282))  # port in accordance with arm/control.py

# face detection setup
# TODO: enable command-line choice of detection algorithm
base_dir = os.path.dirname(os.path.abspath(__file__))
face_model_file = os.path.join(base_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")
face_config_file = os.path.join(base_dir, "models", "deploy.prototxt")
face_detector = detection.CaffeFaces(face_model_file, face_config_file)

# gesture recognition setup
modelpath = os.path.join(base_dir, "models", "gesture_recognizer.task")
gesture_recognizer = detection.MediapipeGestures(modelpath)

GESTURE_CONFIRMATION_THRESHOLD = 5
GESTURE_TIMEOUT_FRAMES = 10
gesture_buffer: dict = {}
current_confirmed_gesture: str = "None"
no_gesture_counter = 0

try:
    while True:
        # Capture frame-by-frame
        # ret, frame = video_capture.read()
        # if not ret:
        #     print('Failed to capture frame.')
        #     break

        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        width, height = frame.get_width(), frame.get_height()
        frame = np.asanyarray(frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Face Detection, boxes with high confidence
        boxes: list = face_detector.detect(frame)

        # draw bounding boxes and get largest face
        # largest box will be last
        boxes_sorted: list = sorted(boxes, key=box_size)

        # draw the bounding boxes for all faces
        for box in boxes_sorted:
            lx, ly, rx, ry = box
            # Draw a rectangle around the faces
            cv2.rectangle(frame, (lx, ly), (rx, ry), (0, 255, 0), 2)
            cv2.circle(frame, (int((lx + rx) / 2), int((ly + ry) / 2)), radius=3, color=(0, 0, 255))

        # send largest face to control process
        if boxes_sorted:
            largest_face_coords = boxes_sorted[-1]
            lflx, lfly, lfrx, lfry = largest_face_coords
            face = Face(lflx, lfly, lfrx, lfry, width, height)
        else:
            face = Face.empty()

        # recognize gestures------------------------------------------------------------------------

        gestures: dict = gesture_recognizer.detect(frame)
        gestures_sorted = sorted(gestures.items(), key=lambda x: x[1])

        if gestures_sorted:
            detected_gesture = gestures_sorted[-1][0]
            no_gesture_counter = 0  # Reset counter since a gesture is detected
        else:
            detected_gesture = None
            no_gesture_counter += 1

        # Update gesture buffer
        if detected_gesture:
            if detected_gesture in gesture_buffer:
                gesture_buffer[detected_gesture] += 1
            else:
                gesture_buffer = {detected_gesture: 1}  # Reset buffer for new gesture
        else:
            gesture_buffer = {}

        # Confirm a gesture if it exceeds the threshold
        for gesture, count in gesture_buffer.items():
            if count >= GESTURE_CONFIRMATION_THRESHOLD:
                current_confirmed_gesture = gesture
                break

        if no_gesture_counter >= GESTURE_TIMEOUT_FRAMES:
            current_confirmed_gesture = "None"

        # Display the confirmed gesture
        if current_confirmed_gesture:
            cv2.putText(
                frame,
                current_confirmed_gesture,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # send results
        print((face, current_confirmed_gesture))
        conn.send((face, current_confirmed_gesture))

        # Display the resulting frame
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    # video_capture.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
finally:
    conn.send("close")
    conn.close()
