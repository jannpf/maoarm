import time
import queue
import threading
import logging
from collections import deque
from multiprocessing.connection import Listener

import numpy as np
from ArmControl import AngleControl


WINDOW_SIZE = 5
IPC_PORT = 6282
ARM_ADDRESS = '192.168.4.1'

data_queue = queue.Queue(maxsize=100)
face_lock = threading.Lock()
current_face = (0, 0, 0, 0)

logging.basicConfig(handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def listen():
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
    c = AngleControl(ARM_ADDRESS)
    c.reset()

    while True:
        with face_lock:
            x, y, width, height = current_face
        if x == 0 and y == 0:
            c.stop()
            c.led_off()
        else:
            print(f"Moving to {x},{y} ({width}, {height})")
            c.led_on(80)
            move_control(c, x, y, width, height)
        time.sleep(0.05)


def process():
    window = deque(maxlen=WINDOW_SIZE)
    global current_face
    while True:
        try:
            data = data_queue.get(timeout=1)
            window.append(data)
            if len(window) == WINDOW_SIZE and all(x[:4] != (0, 0, 0, 0) for x in window):
                print(f"Face at: {data}, queue len: {data_queue.qsize()}")
                (x, y, _, _, width, height) = data
                with face_lock:
                    current_face = (x, y, width, height)
            else:
                with face_lock:
                    current_face = (0, 0, 0, 0)
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
