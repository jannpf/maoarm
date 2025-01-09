"""
Robotic arm control module integrating face and gesture detection data.
Listens for detection data, processes it, and controls arm movement and LED status.
"""

import time
import queue
import threading
import logging
from dataclasses import dataclass, field
from collections import deque
from multiprocessing.connection import Listener

from ArmControl import AngleControl


WINDOW_SIZE = 5
IPC_PORT = 6282
ARM_ADDRESS = '192.168.4.1'
MVMT_UPDATE_TIME = 0.015  # how often to check for current coord

data_queue = queue.Queue(maxsize=100)

# ensures safe (one thread at a time) access to shared data
face_lock = threading.Lock()

"""
Face has (x, y, w, h) interface in accordance with CV algo, where:
    x: (float): x coord of the centre of face's bounding box.
        Lies within [-w/2: w/2] interval, w/2 is on the right
    y: (float): y coord of the centre of face's bounding box
        Lies within [-h/2: h/2] interval, h/2 is up (!)
    w: (float): width of the frame (not the width of the bounding box)
    h: (float): height of the frame (not the height of the bounding box)
"""
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
        msg = conn.recv()  # face and gesture from CV algo
        if msg == 'close':
            conn.close()
            break
        try:
            data_queue.put(msg)
        except queue.Full:
            print("Queue is full! Dropping...")
    listener.close()


@dataclass
class PID:
    """
    A PID controller for smooth robotic arm control.

    Attributes:
        control (AngleControl): Interface for controlling the robotic arm's movement.
        dt (float): Time step between control updates.
        kp, ki, kd (float): PID gains.
        min_output, max_output (int): limits for the speed controller output.
    """

    control: AngleControl
    dt: float

    kpx: float = 80.0
    kpy: float = 16.0
    kix: float = 0.1
    kiy: float = 0.1
    kdx: float = 0.5
    kdy: float = 0.1

    min_output_x: int = 10
    max_output_x: int = 30
    min_output_y: int = 2
    max_output_y: int = 20

    # Internal variables
    error_sum_x: float = field(default=0.0, init=False)
    error_sum_y: float = field(default=0.0, init=False)
    last_error_x: float = field(default=0.0, init=False)
    last_error_y: float = field(default=0.0, init=False)

    def move_control(
        self,
        target_x: float,
        target_y: float,
        width: float,
        height: float,
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
    pid = PID(control=c, dt=MVMT_UPDATE_TIME)

    while True:
        with face_lock:  # data shared with process()
            x, y, width, height = current_face
        if x == 0 and y == 0:  # case no face detected
            c.stop()
            c.led_off()
        elif c.elbow_breach() or c.base_breach():
            c.stop()
            c.to_initial_position()
            c.led_off()
        else:
            print(f"Moving to {x},{y} ({width}, {height})")
            c.led_on(40)
            pid.move_control(x, y, width, height)
        time.sleep(MVMT_UPDATE_TIME)


def process():
    """
    Gets current face bounding box coordinates and gestures from queue,
    served by listener(). Updates current_face global variable.
    """

    def face_coord_ratio_lower_than_threshold(
        previous_face: tuple, current_face: tuple, threshold: float = 1.3
    ) -> bool:
        """
        Helper function. Designed for the purpose of ignoring face update
        when coordinates change too dramatically.

        Returns True if face did not move/change more
        than by the factor of threshold.
        """
        current_x = current_face[0]
        current_y = current_face[1]
        previous_x = previous_face[0]
        previous_y = previous_face[1]
        ratio_x = 1 + abs((current_x - previous_x) / previous_x)
        ratio_y = 1 + abs((current_y - previous_y) / previous_y)
        ratio = max(ratio_x, ratio_y)
        # Empirically 1.3 is the best threshold
        return ratio < threshold

    window = deque(maxlen=WINDOW_SIZE)
    global current_face

    while True:
        try:
            face, gesture = data_queue.get(timeout=1)
            window.append(face)

            # only runs face updates when 2 consecutive frames have a face
            # TODO: this is empirical, maybe find more robust logic
            if (
                len(window) == WINDOW_SIZE
                and all(x is not None for x in window[-1])  # current face/frame
                and all(x is not None for x in window[-2])  # previous face/frame
            ):
                # only runs face updates when faces in 2 consecutive frames
                # don't differ too much
                # TODO: this is empirical, maybe find more robust logic
                if face_coord_ratio_lower_than_threshold(window[-1], window[-2]):
                    print(f"Face at: {face}, queue len: {data_queue.qsize()}")
                    (x, y, _, _, width, height) = face
                    with face_lock:  # data shared with control_movement()
                        current_face = (x, y, width, height)
            else:  # case face not detected 2 frames in a row
                with face_lock:
                    current_face = (0, 0, 0, 0)

            if gesture:
                # TODO: add current_gesture global?
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
