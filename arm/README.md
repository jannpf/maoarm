# arm

A collection of python scripts for controlling the arm movement.

## Some notes on movement

More here: 
* [Python communication](https://www.waveshare.com/wiki/RoArm-M2-S_Python_HTTP_Request_Communication)
* [JSON commands](https://www.waveshare.com/wiki/RoArm-M2-S_Robotic_Arm_Control)

### Basic commands 
Notes:
* `T` is the command code
* T: 101 - single joint control (except wrist)
* T: 102 - all joints control
* T: 105 - get coordinate

```json
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

```json
# base movement
{"T":123,"m":0,"axis":1,"cmd":2,"spd":1} # clockwise
{"T":123,"m":0,"axis":1,"cmd":1,"spd":1} # counterclockwise
{"T":123,"m":0,"axis":1,"cmd":0,"spd":1} # stop

# elbow movement
{"T":123,"m":0,"axis":3,"cmd":2,"spd":1} # up
{"T":123,"m":0,"axis":3,"cmd":1,"spd":1} # down
{"T":123,"m":0,"axis":3,"cmd":0,"spd":1} # stop

# wrist movement
{"T":123,"m":0,"axis":4,"cmd":2,"spd":1} # up
{"T":123,"m":0,"axis":3,"cmd":1,"spd":1} # down
{"T":123,"m":0,"axis":3,"cmd":0,"spd":1} # stop
```

### Misc
* T: 112 - set torque limits (useful to make the arm less harmful!)
