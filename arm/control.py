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
from typing import Optional

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


class PID:
    def _init_(self, control, setpoint=0, min_output=-10, max_output=10):
        """
        kp, ki, kd: PID gains.
        setpoint:   Desired target (usually 0 for "centered").
        min_output, max_output: limits for the controller output.
        """
        self.kpx = 12
        self.kpy = 16
        # self.ki = ki
        # self.kd = kd
        self.control = control
        # self.setpoint = setpoint
        # self.min_output = min_output
        # self.max_output = max_output

        # Internal variables
        self.integral = 0
        self.last_error = 0
        self.last_time = None

    def move_control(
        self,
        target_x,
        target_y,
        width,
        height,
        last_face: Optional[tuple],
    ):
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
            last_face (tuple): (x, y, w, h) for face position at prev time step.
        """
        error_x = target_x / (width / 2)
        error_y = target_y / (height / 2)

        # Proportional term
        p_x = self.kpx * error_x
        p_y = self.kpy * error_y

        spdx = int(p_x) + 8
        spdy = int(p_y) + 4

        if target_x > 10:
            self.control.base_cw(spdx)
        elif target_x < -10:
            self.control.base_ccw(spdx)
        else:
            self.control.base_stop()
        if target_y > 5:
            self.control.elbow_up(spdy)
        elif target_y < -5:
            self.control.elbow_down(spdy)
        else:
            self.control.elbow_stop()


def control_movement():
    """
    Monitors the global current_face coordinates and
    controls the robotic arm movement using move_control().
    Turns the LED on or off based on whether a face is detected.
    """

    c = AngleControl(ARM_ADDRESS)
    c.to_initial_position()
    last_face = None
    pid = PID(control=c)

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
            pid.move_control(x, y, width, height, last_face)
        last_face = x, y, width, height
        time.sleep(MVMT_UPDATE_TIME)


def process():
    """
    Gets current face bounding box coordinates and gestures from queue,
    served by listener(). Updates current_face global variable.
    """

    window = deque(maxlen=WINDOW_SIZE)
    global current_face
    # TODO: last_error, last_timestamp, current_timestamp
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
