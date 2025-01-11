"""
Robotic arm control module integrating face and gesture detection data.
Listens for detection data, processes it, and controls arm movement and LED status.
"""

import time
import queue
import threading
import logging
from collections import deque
from multiprocessing.connection import Listener

from ArmControl import AngleControl


WINDOW_SIZE = 5
IPC_PORT = 6282
ARM_ADDRESS = '192.168.4.1'
MVMT_UPDATE_TIME = 0.025  # how often to check for current coord

data_queue = queue.Queue(maxsize=100)
# ensures safe (one thread at a time) access to shared data
face_lock = threading.Lock()
current_face = (0, 0, 0, 0)

logging.basicConfig(handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def listen():
    """
    Listens at localhost/{IPC_PORT} to get data from CV face/gesture
    detection algorithms. Stores received messages in a thread-safe queue.
    """

    listener = Listener(('localhost', IPC_PORT))
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)
    while True:
        msg = conn.recv()
        if msg == 'close':
            conn.close()
            break
        try:
            data_queue.put(msg)
        except queue.Full:
            print("Queue is full! Dropping...")
    listener.close()


def move_control(control: AngleControl, target_x, target_y, width, height):
    """
    Moves the robotic arm in the direction of the specified coordinates
    (e.g. the central point of the detected face).
    Adjusts movement speed proportionally based on distance from the target
    to ensure smooth control.

    Args:
        control (AngleControl): The control interface for the robotic arm.
        target_x: Target x-coordinate relative to the frame's center.
        target_y: Target y-coordinate relative to the frame's center.
        width: Width of the frame/image.
        height: Height of the frame/image.
    """

    spdx = int(target_x / (width/2) * 12) + 8
    spdy = int(target_y / (height/2) * 16) + 4

    if target_x > 10:
        control.base_cw(spdx)
    elif target_x < -10:
        control.base_ccw(spdx)
    else:
        control.base_stop()
    if target_y > 5:
        control.elbow_up(spdy)
    elif target_y < -5:
        control.elbow_down(spdy)
    else:
        control.elbow_stop()


def control_movement():
    """
    Monitors the global current_face coordinates and
    controls the robotic arm movement using move_control().
    Turns the LED on or off based on whether a face is detected.
    """

    c = AngleControl(ARM_ADDRESS)
    c.to_initial_position()

    while True:
        with face_lock:  # data shared with process()
            x, y, width, height = current_face
        if x == 0 and y == 0:
            c.stop()
            c.led_off()
        elif c.elbow_breach() or c.base_breach():
            c.stop()
            c.to_initial_position()
            c.led_off()
        else:
            print(f"Moving to {x},{y} ({width}, {height})")
            c.led_on(80)
            move_control(c, x, y, width, height)
        time.sleep(MVMT_UPDATE_TIME)


def process():
    """
    Gets current face bounding box coordinates and gestures from queue,
    served by listener(). Updates current_face global variable.
    """

    window = deque(maxlen=WINDOW_SIZE)
    global current_face
    while True:
        try:
            face, gesture = data_queue.get(timeout=1)
            window.append(face)
            if len(window) == WINDOW_SIZE and all(None not in x[:4] for x in window):
                print(f"Face at: {face}, queue len: {data_queue.qsize()}")
                (x, y, _, _, width, height) = face
                with face_lock:  # data shared with control_movement()
                    current_face = (x, y, width, height)
            else:
                with face_lock:
                    current_face = (0, 0, 0, 0)
            if gesture:
                # TODO: add current_gesture global
                print(f"Gesture: {gesture}")
        except queue.Empty:
            continue

def main():
    input_thread = threading.Thread(target=listen, daemon=True)
    input_thread.start()
    control_thread = threading.Thread(target=control_movement, daemon=True)
    control_thread.start()
    process()


if __name__ == "__main__":
    main()
