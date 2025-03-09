# arm

A python package for controlling the movement of the arm. Overview of the main modules:
1. `control.py`
    * Is the main controller
    * Sets up an interface to listen to updates from face/gesture detection algorithm
    * Adjusts the arm position/behaviour in accordance with the updates
2. `ArmControl.py`
    * Contains basic movement commands
    * Is a python wrapper over HTTP interface of the arm
3. `pid.py`
    * Defines a PID controller for movement
    * Optionally (enabled by default) writes current PID error terms to a file
    * Also contains corresponding visualization code in a separate function

## Some notes on movement

More here: 
* [Python communication with the arm](https://www.waveshare.com/wiki/RoArm-M2-S_Python_HTTP_Request_Communication)
* [json commands](https://www.waveshare.com/wiki/RoArm-M2-S_Robotic_Arm_Control)

## Quickstart for experimenting with movement
1. Assemble and turn on the robotic arm.
2. Connect to Wi-Fi `RoArm-M2` (password is 12345678).
3. Run 192.168.4.1 in browser to access UI, or, alternatively, execute step 4.
4. Start the server: `python3 scripts/basic.py 192.168.4.1` and enter json commands to control the arm.

Examples of json commands are provided in further sections. 

There is also a `scripts/pacman.py` script available for a simple demonstration on how to implement movement commands.

## Commands
### Basic commands 
Notes:
* `T` is the command code
* T: 101 - single joint control (except wrist)
* T: 102 - all joints control
* T: 105 - get coordinate

```py
# somewhat optimal for base rotation (b)
{"T":101,"joint":1,"rad":0,"spd":200,"acc":30}

# somewhat optimal for elbow rotation (e)
{"T":101,"joint":3,"rad":2.50,"spd":200,"acc":30}

# all joints
{"T":102,"base":0,"shoulder":0,"elbow":0.7,"hand":3,spd":200,acc":30}  # hand and shoulder should be fixed

# base position
{"T":102,"base":0,"shoulder":0,"elbow":1.57,"hand":3,spd":200,acc":30}  # hand and shoulder should be fixed

# get coordinate
{"T":105}
```

### Advanced commands (continuous control)
Notes:
* T: 123 - continuous control, with angles ("m": 1)
* Speed 10 is optimal
* Speed should be 0-20

```py
# base movement
{"T":123,"m":0,"axis":1,"cmd":2,"spd":1} # clockwise
{"T":123,"m":0,"axis":1,"cmd":1,"spd":1} # counterclockwise
{"T":123,"m":0,"axis":1,"cmd":0,"spd":1} # stop

# elbow movement
{"T":123,"m":0,"axis":3,"cmd":2,"spd":1} # up
{"T":123,"m":0,"axis":3,"cmd":1,"spd":1} # down
{"T":123,"m":0,"axis":3,"cmd":0,"spd":1} # stop

# wrist movement
{"T":123,"m":0,"axis":4,"cmd":2,"spd":1} # down
{"T":123,"m":0,"axis":3,"cmd":1,"spd":1} # up
{"T":123,"m":0,"axis":3,"cmd":0,"spd":1} # stop
```

### Misc commands
* T: 112 - set torque limits (useful to make the arm less harmful!)
