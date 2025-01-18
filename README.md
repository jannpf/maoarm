# maoarm

**Was wollt ihr denn?**  
  
Controlling the movement of the robotic arm RoArm-M2-S. In a cat-like way (whatever that is, we'll figure out).

## Basic usage

1. Assemble and turn on the robotic arm.
2. Connect to Wi-Fi `RoArm-M2` (password is 12345678).

In 2 separate terminal windows, run the following scripts:

```sh
python3 -m arm.control
python3 -m cv
```

## Architecture overview
[Link to editable pic](https://excalidraw.com/#json=lx-_RDzIbAT0w5aPFi1q0,5ac6kPpwJuyASrhV8_aF5Q)

![image](.assets/arch.png)

## Mood Impact of Gestures
![image](.assets/mood_impact_gestures.png)
