import argparse
from multiprocessing.connection import Client
import os
import sys

import cv2

from .face import Face
from . import detection  # CaffeFaces, MediapipeGestures
from .camera import Webcam

from .detection.DetectionBase import DetectionBase
from .camera.CameraBase import CameraBase


# IP address to communicate data with
CONN = Client(("localhost", 6282))  # port in accordance with arm/control.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# gesture recognition setup
GESTURE_CONFIRMATION_THRESHOLD = 5
GESTURE_TIMEOUT_FRAMES = 10
gesture_buffer: dict = {}
current_confirmed_gesture: str = "None"  # this name will be displayed on frame
no_gesture_counter = 0


def box_size(box: tuple[int, int, int, int]) -> float:
    lx, ly, rx, ry = box
    return (rx - lx) * (ry - ly)


def parse_args() -> argparse.Namespace:
    # get command line args
    parser = argparse.ArgumentParser(description="Face Detection Script")
    parser.add_argument(
        "--camera",
        type=str,
        default="realsense",
        choices=["webcam", "realsense"],
        help="Camera source (default: realsense)",
    )
    parser.add_argument(
        "--face_detection_algorithm",
        type=str,
        default="caffe",
        choices=["caffe", "cascade"],
        help="Face detection algorithm to use (default: caffe)",
    )
    args = parser.parse_args()
    print(f"Using camera: {args.camera}")
    print(f"Face detection algorithm: {args.face_detection_algorithm}")
    return args


def initialize_camera(camera_arg: str) -> CameraBase:
    if camera_arg == "realsense":
        if sys.platform == "darwin":
            print("Realsense is not supported for MacOS, falling back to using webcam...")
            camera = Webcam.Webcam()
        else:
            from .camera import Realsense
            camera = Realsense.RealsenseCamera()
    elif camera_arg == "webcam":
        camera = Webcam.Webcam()
    else:
        raise ValueError(f"Invalid camera type: {camera_arg}")
    return camera


def face_detection_setup(face_detection_algorithm_arg: str) -> DetectionBase:
    face_detector: DetectionBase
    if face_detection_algorithm_arg == "caffe":
        face_model_file = os.path.join(
            BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel"
        )
        face_config_file = os.path.join(BASE_DIR, "models", "deploy.prototxt")
        face_detector = detection.CaffeFaces(face_model_file, face_config_file)
    elif face_detection_algorithm_arg == "cascade":
        face_model_file = os.path.join(
            BASE_DIR, "models", "cascades", "haarcascade_frontalface_default.xml"
        )
        face_detector = detection.CascadeFaces(face_model_file)
    else:
        raise ValueError(
            f"Invalid face detection algo type: {face_detection_algorithm_arg}"
        )
    return face_detector


def gesture_recognition_setup() -> DetectionBase:
    modelpath = os.path.join(BASE_DIR, "models", "gesture_recognizer.task")
    gesture_recognizer = detection.MediapipeGestures(modelpath)
    return gesture_recognizer


# initial setup -----------------------------------------------------------------------

args = parse_args()  # 2 args: camera and face_detection_algorithm
camera = initialize_camera(args.camera)
face_detector = face_detection_setup(args.face_detection_algorithm)
gesture_recognizer = gesture_recognition_setup()

frame_width: float = camera.frame_width()
frame_height: float = camera.frame_height()

try:
    while True:
        # Capture frame-by-frame
        frame = camera.get_frame()

        # detect faces ----------------------------------------------------------------

        # Face Detection, boxes with high confidence
        boxes: list = face_detector.detect(frame)  # type: ignore

        # draw bounding boxes and get largest face
        # largest box will be last
        boxes_sorted: list = sorted(boxes, key=box_size)

        # draw the bounding boxes for all faces
        for box in boxes_sorted:
            lx, ly, rx, ry = box
            # Draw a rectangle around the faces
            cv2.rectangle(frame, (lx, ly), (rx, ry), (0, 255, 0), 2)
            cv2.circle(
                frame,
                (int((lx + rx) / 2), int((ly + ry) / 2)),
                radius=3,
                color=(0, 0, 255),
            )

        # send largest face to control process
        if boxes_sorted:
            largest_face_coords = boxes_sorted[-1]
            lflx, lfly, lfrx, lfry = largest_face_coords
            face = Face(int(lflx), int(lfly), int(lfrx), int(lfry), int(frame_width), int(frame_height))
        else:
            face = Face.empty()

        # recognize gestures -----------------------------------------------------------------------

        gestures: dict = gesture_recognizer.detect(frame)  # type: ignore
        gestures_sorted: list = sorted(gestures.items(), key=lambda x: x[1])

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
                # Reset buffer for new gesture
                gesture_buffer = {detected_gesture: 1}
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

        # send results and wrap up ----------------------------------------------------

        print((face, current_confirmed_gesture))
        CONN.send((face, current_confirmed_gesture))

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
    CONN.send("close")
    CONN.close()
