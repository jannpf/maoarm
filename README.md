# 🖐️ Dynamic Hand Wave Detection
## 📖 Description
This file detects hand wave gestures using MediaPipe and OpenCV. It tracks hand landmarks and analyzes movement to detect “waving” gestures with a configurable threshold.
## 📦 Requirements
| Package   | Version                       |
|-----------|-------------------------------|
| Python    |3.7 to 3.11 (recommended: 3.11)|
| OpenCV    |Latest                         |
| MediaPipe |latest                         |

## ⚙️ Configuration
You can adjust these in the MediapipeWaves class constructor:
- wave_frames_window = 30             # Number of frames to check for wave
- wave_movement_threshold = 0.05      # Minimum movement needed for a wave
- max_history_length = 50              # How much history to keep

## 🚀 How to Run
Run the wave_detection script.
This will open the webcam feed and start tracking hand movements.
