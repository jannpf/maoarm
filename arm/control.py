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
MVMT_UPDATE_TIME = 0.015  # how often to check for current coord

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
    def __init__(self, control):
        """
        kp, ki, kd: PID gains.
        setpoint:   Desired target (usually 0 for "centered").
        min_output, max_output: limits for the controller output.
        """
        self.kpx = 12
        self.kpy = 16
        self.kix = 0.1
        self.kiy = 0.1
        self.kdx = 0.1
        self.kdy = 0.1
        self.control = control

        self.min_output_x = 4
        self.max_output_x = 20
        self.min_output_y = 2
        self.max_output_y = 20

        # Internal variables
        self.error_sum_x = 0
        self.error_sum_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.dt = MVMT_UPDATE_TIME

    def move_control(
        self,
        target_x,
        target_y,
        width,
        height,
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
        error_x = abs(target_x / (width / 2))
        error_y = abs(target_y / (height / 2))

        # Proportional term
        p_x = self.kpx * error_x
        p_y = self.kpy * error_y

        # Integral term
        self.error_sum_x += error_x * self.dt
        self.error_sum_x += error_x * self.dt
        i_x = self.error_sum_x * self.kix
        i_y = self.error_sum_y * self.kiy

        # Derivative term
        d_x = self.kdx * (error_x - self.last_error_x) / self.dt
        d_y = self.kdy * (error_y - self.last_error_y) / self.dt
        self.last_error_x = error_x
        self.last_error_y = error_x

        # PID output
        control_x = p_x + i_x + d_x
        control_y = p_y + i_y + d_y
        print(f"{control_x=}")
        print(f"{control_y=}")

        # np.clip from min to max
        spdx = min(max(int(control_x), self.min_output_x), self.max_output_x)
        spdy = min(max(int(control_y), self.min_output_y), self.max_output_y)

        if target_x > width / 20:
            self.control.base_cw(spdx)
        elif target_x < - width / 20:
            self.control.base_ccw(spdx)
        else:
            self.control.base_stop()
        if target_y > height / 10:
            self.control.elbow_up(spdy)
        elif target_y < - height / 10:
            self.control.elbow_down(spdy)
        else:
            self.control.elbow_stop()
        # reset shoulder as it tends to move
        self.control.shoulder_to(0, spd=2, acc=2)


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
            pid.move_control(x, y, width, height)
        time.sleep(MVMT_UPDATE_TIME)


def process():
    """
    Gets current face bounding box coordinates and gestures from queue,
    served by listener(). Updates current_face global variable.
    """

    def get_face_coord_ratio(previous_frame, current_frame):
        """
        Helper function. Designed for the purpose of ignoring face update
        when coordinates change too dramatically
        """
        current_x = current_frame[0]
        current_y = current_frame[1]
        previous_x = previous_frame[0]
        previous_y = previous_frame[1]
        ratio_x = abs((current_x - previous_x) / previous_x)
        ratio_y = abs((current_y - previous_y) / previous_y)
        ratio = max(ratio_x, ratio_y)
        return ratio

    window = deque(maxlen=WINDOW_SIZE)
    global current_face

    while True:
        try:
            face, gesture = data_queue.get(timeout=1)
            window.append(face)

            if (
                len(window) == WINDOW_SIZE
                and all(x is not None for x in window[-1])  # current face
                and all(x is not None for x in window[-2])  # previous face
            ):
                ratio = get_face_coord_ratio(window[-1], window[-2])
                # 1.3 is empirical
                if ratio < 1.3:  # if face did not move/change too quickly
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
