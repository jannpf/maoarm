from .DetectionBase import DetectionBase

import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2


class MediapipeGestures(DetectionBase):
    def __init__(self, modelpath):
        # Configure Gesture Recognizer
        self.base_options = python.BaseOptions(model_asset_path=modelpath)
        self.gesture_options = vision.GestureRecognizerOptions(
            base_options=self.base_options)
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(
            self.gesture_options)

        # Initialize MediaPipe Hands for landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def calculate_angle(a, b, c) -> float:
        """
        Calculate the angle between three points a, b, c
        where b is the vertex.
        """
        ab = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
        magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if magnitude_ab * magnitude_bc == 0:
            return 0
        angle = math.acos(dot_product / (magnitude_ab * magnitude_bc))
        return math.degrees(angle)

    def detect_middle_finger(self, hand_landmarks) -> bool:
        """
        Improved middle finger detection using PIP as a reference.
        """
        # Landmarks
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]

        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]

        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Angle check for the middle finger: MCP -> PIP -> TIP
        middle_angle = self.calculate_angle(middle_mcp, middle_pip, middle_tip)

        # Conditions for middle finger raised
        middle_raised = middle_tip.y < middle_pip.y  # Middle finger tip is above PIP
        middle_straight = middle_angle > 160  # Middle finger is straight

        # Other fingers should be folded
        index_folded = index_tip.y > index_pip.y
        ring_folded = ring_tip.y > ring_pip.y
        pinky_folded = pinky_tip.y > pinky_pip.y

        return middle_raised and middle_straight and index_folded and ring_folded and pinky_folded

    def detect(self, frame) -> dict:
        # Gesture Recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognition_result = self.gesture_recognizer.recognize(mp_image)

        result = {}
        for g in recognition_result.gestures:
            detected_category = g[0]
            result[detected_category.category_name] = detected_category.score

        # Detect custom gestures (e.g., middle finger)
        landmarks = self.hands.process(rgb_frame)
        if landmarks.multi_hand_landmarks:
            for hand_landmarks in landmarks.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)
                if self.detect_middle_finger(hand_landmarks):
                    result["Middle Finger"] = 1.0

        return result
