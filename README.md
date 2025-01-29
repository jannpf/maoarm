# maoarm

**Was wollt ihr denn?**  
  
This repo contains code for controlling the movement of the robotic arm RoArm-M2-S based on the obtained computer vision data. In a cat-like way (whatever that is, we'll figure out).

The idea of this project is to attach a camera to the robotic arm in order to recognize faces and gestures, and use this information to update the movement mode of the arm. The arm is originally designed to move the camera in the direction of the detected face, but also supports different movement modes, based on the current mood of the cat character. Yes, there is an underlying character defining the movement style :)

**Current features**:
* Cross-platform compatibility
* PID-like movement control
* Face and gesture recognition
* Support of 2 face detection algorithms
* Recognition of 7 gestures (including middle finger :sunglasses:)
* 4 movement modes (relaxed, excited, depressed, angry)
* Cat mood system to control movement modes:
    * Automatic random updates of the current mood based on Markov chain Monte Carlo (MCMC) approach
    * Mood updates are influenced by detected gestures

## Prerequisites
#### Hardware:
* RoArm-M2-S
* A camera or a web camera
* A computer (e.g. a laptop or a Raspberry Pi)

#### Software
* python3 (works best with 3.12) 

#### Download and configure
Clone repository and install dependencies:
```
python3 -m pip install -r requirements.txt
```

## Basic usage

1. Assemble and turn on the robotic arm.
2. Connect to Wi-Fi `RoArm-M2` (password is 12345678).
3. Place a webcam near robotic arm's LED and connect it to the computer.

In 2 separate terminal windows, run the following scripts:

```sh
python3 -m arm.control
python3 -m cv
```

`python3 -m cv` supports different options for camera and face detection algorithm. Run `python3 -m cv --help` for details.

## Architecture overview
[Link to editable pic](https://excalidraw.com/#json=YMj9FkYP01Mxj0f9aDgZH,iCZbDP2nFBiHpHw-pKpiCQ)

![image](.assets/arch.png)

## Mood Impact of Gestures
![image](.assets/mood_impact_gestures.png)
