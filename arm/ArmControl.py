"""
Python wrapper around http commands for controlling robotic arm:
https://www.waveshare.com/wiki/RoArm-M2-S_Robotic_Arm_Control
"""

import requests
import json


class BasicControl:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def run_and_get_response(self, command: str) -> str:
        url = "http://" + self.ip_address + "/js?json=" + command
        response = requests.get(url)
        content = response.text
        return content

    def current_position(self):
        command = {"T": 105}
        return self.run_and_get_response(json.dumps(command))

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
            "shoulder": 0,
            "elbow": 1.57,
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

    def stop(self):
        self.base_stop()
        self.shoulder_stop()
        self.elbow_stop()
        self.hand_stop()
