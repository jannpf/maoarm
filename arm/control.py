"""
Robotic arm control module integrating face and gesture detection data.
Listens for detection data, processes it, and controls arm movement and LED status.
"""

import os
import time
import json
import queue
import threading
import logging
from collections import deque
from multiprocessing.connection import Listener

from cv.face import Face
from arm import AngleControl, PID, Cat

WINDOW_SIZE = 5
IPC_PORT = 6282
ARM_ADDRESS = '192.168.4.1'
MVMT_UPDATE_TIME = 0.015  # how often to check for current coord in seconds
MOOD_UPDATE_TIME = 1  # in seconds
CHARACTER_FILE = 'spot.json'

data_queue = queue.Queue(maxsize=100)

# ensures safe (one thread at a time) access to shared data
face_lock = threading.Lock()
gesture_lock = threading.Lock()

current_face = Face.empty()
current_gesture = None

logging.basicConfig(handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def mood_control():
    """
    Controls the cat behaviour, including random mood drift and
    mood changes based on interaction through gestures
    """
    # cat characteristics
    cat_profile = json.load(open(CHARACTER_FILE, 'r'))
    character_gaussians = cat_profile["gaussians"]
    gesture_impact = cat_profile["gesture_impact"]

    cat = Cat(
        gaussians=character_gaussians,
        valence=0.0,
        arousal=0.0,
        proposal_sigma=0.2,
        plot=True,
        maxtracelen=1000,
    )

    last_gesture = None

    while True:
        print("iterating mood.")
        cat.mood_iteration()

        # check for gestures
        with gesture_lock:
            detected_gesture = current_gesture

        # only recognize one gesture at a time
        if detected_gesture != last_gesture:
            last_gesture = detected_gesture
            if detected_gesture in gesture_impact:
                v_offset, a_offset = gesture_impact[detected_gesture]
                cat.override_mood(cat.valence+v_offset, cat.arousal+a_offset)

        time.sleep(MOOD_UPDATE_TIME)


def listen():
    """
    Listens at localhost/{IPC_PORT} to get data from CV face/gesture
    detection algorithms. Stores received messages in a thread-safe queue.
    """

    listener = Listener(('localhost', IPC_PORT))
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)
    while True:
        msg = conn.recv()  # face, gesture from CV algo
        if msg == 'close':
            conn.close()
            break
        try:
            data_queue.put(msg)
        except queue.Full:
            print("Queue is full! Dropping...")
    listener.close()


def control_movement():
    """
    Monitors the global current_face coordinates and
    controls the robotic arm movement using move_control().
    Turns the LED on or off based on whether a face is detected.
    """

    c = AngleControl(ARM_ADDRESS)
    c.to_initial_position()
    pid = PID(control=c)

    while True:
        with face_lock:  # data shared with process()
            x, y, frame_width, frame_height = (
                current_face.x or 0,
                current_face.y or 0,
                current_face.frame_width,
                current_face.frame_height
            )
        if not current_face.is_detected():
            c.stop()
            c.led_off()
        elif c.elbow_breach() or c.base_breach():
            c.stop()
            c.to_initial_position()
            c.led_off()
        else:
            print(f"Moving to {x},{y}; ({frame_width}x{frame_height})")
            c.led_on(40)
            pid.move_control(x, y, frame_width, frame_height)
        time.sleep(MVMT_UPDATE_TIME)


def process():
    """
    Gets current face bounding box coordinates and gestures from queue,
    served by listener(). Updates current_face global variable.
    """

    def face_coord_ratio_lower_than_threshold(
        previous_face: Face, current_face: Face, threshold: float = 1.3
    ) -> bool:
        """
        Helper function. Designed for the purpose of ignoring face update
        when coordinates change too dramatically.

        Returns True if face did not move/change more
        than by the factor of threshold.
        """
        current_x = current_face.x
        current_y = current_face.y
        previous_x = previous_face.x or 1e-5  # avoid div by zero
        previous_y = previous_face.y or 1e-5
        ratio_x = 1 + abs((current_x - previous_x) /
                          previous_x)  # type: ignore
        ratio_y = 1 + abs((current_y - previous_y) /
                          previous_y)  # type: ignore
        ratio = max(ratio_x, ratio_y)
        # Empirically 1.3 is the best threshold
        return ratio < threshold

    window = deque(maxlen=WINDOW_SIZE)
    global current_face
    global current_gesture
    face: Face
    gesture: str

    while True:
        try:
            face, gesture = data_queue.get(timeout=1)
            window.append(face)

            # only runs face updates when 2 consecutive frames have a face
            # TODO: this is empirical, maybe find more robust logic
            if (
                len(window) == WINDOW_SIZE
                and face.is_detected()  # current face/frame
                and window[-2].is_detected()  # previous face/frame
            ):
                # only runs face updates when faces in 2 consecutive frames
                # don't differ too much
                # TODO: this is empirical, maybe find more robust logic
                if face_coord_ratio_lower_than_threshold(window[-1], window[-2]):
                    print(f"{face}, queue len: {data_queue.qsize()}")
                    with face_lock:  # data shared with control_movement()
                        current_face = face
            else:  # case face not detected 2 frames in a row
                with face_lock:
                    current_face = Face.empty()

            if gesture:
                with gesture_lock:
                    current_gesture = gesture
                print(f"Gesture: {gesture}")

        except queue.Empty:
            continue


def main():
    try:
        os.remove("interim_values.json")
    except FileNotFoundError:
        pass

    input_thread = threading.Thread(target=listen, daemon=True)
    input_thread.start()
    control_thread = threading.Thread(target=control_movement, daemon=True)
    control_thread.start()
    processing_thread = threading.Thread(target=process, daemon=True)
    processing_thread.start()
    mood_control()


if __name__ == "__main__":
    main()
