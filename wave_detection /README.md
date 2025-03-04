# Overview
This Python script detects dynamic waving gestures using a webcam feed. It leverages MediaPipe Hands to detect hand landmarks and tracks the horizontal (X-axis) movement of the wrist over a short time window. If the wrist moves back and forth across a defined threshold, it classifies the motion as a wave gesture.
## Dependencies:
- opencv-python (cv2)
- mediapipe