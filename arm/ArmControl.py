"""
Python wrapper around http commands for controlling robotic arm:
https://www.waveshare.com/wiki/RoArm-M2-S_Robotic_Arm_Control
"""

import requests
import json
import time
import csv


class BasicControl:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.session: requests.Session = requests.session()
        self.timestamps = []
        self.response_times = []
        self.log_file = "response_log.csv"
        # Initialize the log file with headers
        with open(self.log_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "response_time"])  # CSV headers

    def run_and_get_response(self, command: str) -> str:
        url = f"http://{self.ip_address}/js?json={command}"
        current_time = time.time()

        try:
            response = self.session.get(url, timeout=(1, 3))
            response_time = time.time() - current_time  # Fixed bug
            response_text = response.text
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            response_time = -1  # Mark failed requests with -1

        # Store data
        self.timestamps.append(current_time)
        self.response_times.append(response_time)

        # Write to file
        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([current_time, response_time])

        return response_text if response_time != -1 else "ERROR"

    def current_position(self) -> dict:
        command = {"T": 105}
        coord_str = self.run_and_get_response(json.dumps(command))
        return json.loads(coord_str)

    def torque_limit():
        raise NotImplementedError()

    def reset(self):
        command = {"T": 100}
        return self.run_and_get_response(json.dumps(command))

    def led_on(self, brightness=255):
        command = {"T": 114, "led": brightness}
        return self.run_and_get_response(json.dumps(command))

    def led_off(self):
        command = {"T": 114, "led": 0}
        return self.run_and_get_response(json.dumps(command))


class AngleControl(BasicControl):
    def __init__(self, ip_address):
        super().__init__(ip_address)

    def to_initial_position(self, spd=150, acc=100):
        command = {
            "T": 102,
            "base": 0,
            "shoulder": -0.5,
            "elbow": 1.87,
            "hand": 3,
            "spd": spd,
            "acc": acc
        }
        self.run_and_get_response(json.dumps(command))

    def to_absolute(self, base, shoulder, elbow, hand, spd=150, acc=100):
        command = {
            "T": 102,
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "hand": hand,
            "spd": spd,
            "acc": acc
        }
        self.run_and_get_response(json.dumps(command))

    def hand_to(self, rad, spd, acc):
        command = {"T": 101, "joint": 4, "rad": rad, "spd": spd, "acc": acc}
        self.run_and_get_response(json.dumps(command))

    def hand_down(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 4, "cmd": 2, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def hand_up(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 4, "cmd": 1, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def hand_stop(self):
        command = {"T": 123, "m": 0, "axis": 4, "cmd": 0}
        self.run_and_get_response(json.dumps(command))

    def elbow_to(self, rad, spd, acc):
        command = {"T": 101, "joint": 3, "rad": rad, "spd": spd, "acc": acc}
        self.run_and_get_response(json.dumps(command))

    def elbow_down(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 3, "cmd": 1, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def elbow_up(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 3, "cmd": 2, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def elbow_stop(self):
        command = {"T": 123, "m": 0, "axis": 3, "cmd": 0}
        self.run_and_get_response(json.dumps(command))

    def elbow_breach(self, coords: dict = {}) -> bool:
        if not coords:
            coords = self.current_position()
        if coords["e"] < 0.28 or coords["b"] > 2.50:
            return True
        return False

    def shoulder_to(self, rad, spd, acc):
        command = {"T": 101, "joint": 2, "rad": rad, "spd": spd, "acc": acc}
        self.run_and_get_response(json.dumps(command))

    def shoulder_down(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 2, "cmd": 1, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def shoulder_up(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 2, "cmd": 2, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def shoulder_stop(self):
        command = {"T": 123, "m": 0, "axis": 2, "cmd": 0}
        self.run_and_get_response(json.dumps(command))

    def base_to(self, rad, spd, acc):
        command = {"T": 101, "joint": 1, "rad": rad, "spd": spd, "acc": acc}
        self.run_and_get_response(json.dumps(command))

    def base_cw(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 1, "cmd": 2, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def base_ccw(self, speed: int = 5):
        command = {"T": 123, "m": 0, "axis": 1, "cmd": 1, "spd": speed}
        self.run_and_get_response(json.dumps(command))

    def base_stop(self):
        command = {"T": 123, "m": 0, "axis": 1, "cmd": 0}
        self.run_and_get_response(json.dumps(command))

    def base_breach(self, coords: dict = {}) -> bool:
        if not coords:
            coords = self.current_position()
        if coords["b"] < -3.14 or coords["b"] > 3.14:
            return True
        return False

    def stop(self):
        command = {"T": 123, "m": 0, "cmd": 0}
        self.run_and_get_response(json.dumps(command))


import matplotlib.pyplot as plt

def visualize_from_csv(log_file="response_log.csv"):
    """Loads timestamp and response time data from a CSV file and visualizes it."""
    
    timestamps = []
    response_times = []

    # Load data from CSV
    try:
        with open(log_file, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            for row in reader:
                timestamps.append(float(row[0]))
                response_times.append(float(row[1]))

    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        return
    except ValueError:
        print("Error: Invalid data format in CSV.")
        return

    if not timestamps:
        print("No data to visualize.")
        return

    plt.figure(figsize=(12, 5))

    # 1. Histogram of timestamps (requests per second)
    plt.subplot(1, 2, 1)
    bins = int(timestamps[-1] - timestamps[0]) or 1  # Ensure at least 1 bin
    plt.hist(timestamps, bins=bins, edgecolor="black")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Request count")
    plt.title("Request Frequency per Second")

    # 2. Line plot of response times
    plt.subplot(1, 2, 2)
    plt.plot(timestamps, response_times, marker="o", linestyle="-", color="r", label="Response Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Response Time (s)")
    plt.title("Response Time Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()
