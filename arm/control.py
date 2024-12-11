from multiprocessing.connection import Listener
from collections import deque
import threading
import queue

data_queue = queue.Queue(maxsize=100)
WINDOW_SIZE = 5

def listen():
    listener = Listener(('localhost', 6282))
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
            print("Queue is full! Dropping data new entry...")
    listener.close()


def process():
    window = deque(maxlen=WINDOW_SIZE)
    while True:
        try:
            data = data_queue.get(timeout=1)
            window.append(data)
            if len(window) == WINDOW_SIZE and all(x != (0,0,0,0) for x in window):
                print(f"Face at: {data}, queue len: {data_queue.qsize()}")
        except queue.Empty:
            continue

def main():
    input_thread = threading.Thread(target=listen, daemon=True)
    input_thread.start()
    process()
    # control_thread = threading.Thread(target=process, daemon=True)
    # control_thread.start()

if __name__ == "__main__":
    main()
