"""
Simple demonstration of movement possibilities.
This script moves the "hand" of the robotic arm up and down.
Looks like pacman.
"""

# import argparse
import requests
from time import sleep


IP_ADDR = "192.168.4.1"


def main():
    # parser = argparse.ArgumentParser(description='Http JSON Communication')
    # parser.add_argument('ip', type=str, help='IP address: 192.168.4.1')
    # args = parser.parse_args()
    # ip_addr = args.ip

    switch = True
    move_hand = HandMovement()

    try:
        while True:
            # command = input("input your json cmd: ")
            if switch:
                move_hand.down_slow()
            else:
                move_hand.up_slow()
            switch = not switch
            sleep(5)
    except KeyboardInterrupt:
        move_hand.stop()
        reset_coord()


def run_and_get_response(command: str) -> str:
    url = "http://" + IP_ADDR + "/js?json=" + command
    response = requests.get(url)
    content = response.text
    print(content)
    return content


def reset_coord():
    command = '{"T":102,"base":0,"shoulder":0,"elbow":1.57,"hand":3,spd":20,acc":30}'  # initial coord
    run_and_get_response(command)


class HandMovement:
    def __init__(self):
        pass

    def down_slow(self):
        command = '{"T":123,"m":0,"axis":4,"cmd":2,"spd":5}'  # down
        run_and_get_response(command)

    def up_slow(self):
        command = '{"T":123,"m":0,"axis":4,"cmd":1,"spd":5}'  # up
        run_and_get_response(command)

    def stop(self):
        command = '{"T":123,"m":0,"axis":4,"cmd":0,"spd":5}'  # stop
        run_and_get_response(command)


if __name__ == "__main__":
    main()
