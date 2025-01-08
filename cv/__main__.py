import os
import cv2
from multiprocessing.connection import Client
from . import detection
# CaffeFaces, MediapipeGestures


def to_center_coord(x1, y1, x2, y2, width, height):
    """
    Convert bounding box coordinates to centered coordinates.
    Returns:
        tuple[int, int, int, int, int, int]: A tuple containing:
            - lx: Center x-coordinate with respect to the frame's center.
            - ly: Center y-coordinate with respect to the frame's center (inverted).
            - lw: Width of the bounding box.
            - lh: Height of the bounding box.
            - width: The input width of the frame.
            - height: The input height of the frame.
    """

    (lx, ly, lw, lh) = (x1, y1, x2-x1, y2-y1)

    # convert the coor so that 0,0 is in the center
    (lx, ly) = (lx - width/2, ly - height/2)

    # return the middle coordinate of the face
    (lx, ly) = ((lx+int(lw/2), ly+int(lh/2)))

    return (lx, -ly, lw, lh, width, height)


def box_size(x1, y1, x2, y2):
    return (x2-x1)*(y2-y1)


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

        # Face Detection
        detections = face_detector.detect(frame, width, height)

        # draw bounding boxes and get largest face
        faces_sorted = sorted(detections.items(),
                              key=lambda r: box_size(*r[0]))

        # draw the bounding boxes for all faces
        for box, confidence in faces_sorted:
            x1, y1, x2, y2 = box
            if confidence > 0.5:
                # Draw a rectangle around the faces
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)),
                           radius=3, color=(0, 0, 255))

        # send largest face to control process
        if faces_sorted:
            face = to_center_coord(*faces_sorted[-1][0], width, height)
        else:
            face = (None, None, None, None, width, height)

        # recognize gestures
        gestures = gesture_recognizer.detect(frame)
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
