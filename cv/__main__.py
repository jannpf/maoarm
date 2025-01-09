import os
import cv2
from multiprocessing.connection import Client
from .face import Face
from . import detection  # CaffeFaces, MediapipeGestures


def box_size(box: list[int]) -> float:
    lx, ly, rx, ry = box
    return (rx - lx) * (ry - ly)


# video capture setup
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FOURCC,
                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
video_capture.isOpened()

width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# IP address to communicate data with
conn = Client(('localhost', 6282))  # port in accordance with arm/control.py

# face detection setup
# TODO: enable command-line choice of detection algorithm
base_dir = os.path.dirname(os.path.abspath(__file__))
face_model_file = os.path.join(
    base_dir, 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
face_config_file = os.path.join(base_dir, 'models', 'deploy.prototxt')
face_detector = detection.CaffeFaces(face_model_file, face_config_file)

# gesture recognition setup
modelpath = os.path.join(base_dir, 'models', 'gesture_recognizer.task')
gesture_recognizer = detection.MediapipeGestures(modelpath)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print('Failed to capture frame.')
            break

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
            cv2.circle(frame, (int((lx+rx)/2), int((ly+ry)/2)),
                       radius=3, color=(0, 0, 255))

        # send largest face to control process
        if boxes_sorted:
            largest_face_coords = boxes_sorted[-1]
            lflx, lfly, lfrx, lfry = largest_face_coords
            face = Face(lflx, lfly, lfrx, lfry, width, height)
        else:
            face = Face.empty()

        # recognize gestures
        gestures = gesture_recognizer.detect(frame)
        # sort by score
        gestures_sorted = sorted(gestures.items(), key=lambda x: x[1])

        if gestures_sorted:
            gesture = gestures_sorted[-1][0]
            cv2.putText(frame, gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            gesture = None

        # send results
        print((face, gesture))
        conn.send((face, gesture))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
finally:
    conn.send('close')
    conn.close()
