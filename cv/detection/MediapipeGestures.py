from .DetectionBase import DetectionBase

import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


class MediapipeGestures(DetectionBase):
    def __init__(self, modelpath, wave_frames_window=30, wave_movement_threshold=0.05, max_history_length=50):

        # Configure Gesture Recognizer
        self.base_options = python.BaseOptions(model_asset_path=modelpath)
        self.gesture_options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(self.gesture_options)

        # Initialize MediaPipe Hands for landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # History tracking for wave detection
        self.wave_history = []
        self.max_history_length = 30  # Keep last 30 frames
        self.wave_threshold = 4  # Required direction changes to confirm a wave
        self.detected_wave = False  # Flag to prevent immediate overwrite
        self.wave_persistence = 10  # Keep wave active for 10 frames
        self.wave_counter = 0  # Counter to track persistence duration
        self.min_movement_distance = 0.02  # Minimum x-movement required to count as a valid direction change

    @staticmethod
    def calculate_angle(a, b, c) -> float:
        """
        Calculate the angle between three points a, b, c
        where b is the vertex.
        """
        ab = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
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
    
    def detect_wave(self) -> bool:
        """
        Detect a wave if the "Open_Palm" gesture appears at least 5 times in the last 30 frames
        with at least 4 direction changes in temporal order and significant movement.
        """
        if len(self.wave_history) < 5:
            return False

        # Count direction changes with movement threshold
        direction_changes = 0
        for i in range(1, len(self.wave_history) - 1):
            prev_x = self.wave_history[i - 1]
            current_x = self.wave_history[i]
            next_x = self.wave_history[i + 1]

            if abs(current_x - prev_x) > self.min_movement_distance and abs(next_x - current_x) > self.min_movement_distance:
                if (prev_x < current_x and next_x < current_x) or (prev_x > current_x and next_x > current_x):
                    direction_changes += 1

        return direction_changes >= self.wave_threshold

    def detect(self, frame) -> dict:
        """
        Detects gestures (including wave detection) in the given frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognition_result = self.gesture_recognizer.recognize(mp_image)
        result = {}
        detected_open_palm = False

        for g in recognition_result.gestures:
            detected_category = g[0]
            result[detected_category.category_name] = detected_category.score

            if detected_category.category_name == "Open_Palm":
                detected_open_palm = True

        # Track "Open_Palm" positions for wave detection
        if detected_open_palm:
            hand_landmarks = self.hands.process(rgb_frame)
            if hand_landmarks.multi_hand_landmarks:
                for hand_landmarks in hand_landmarks.multi_hand_landmarks:
                    palm_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                    self.wave_history.append(palm_x)
                    if len(self.wave_history) > self.max_history_length:
                        self.wave_history.pop(0)

        # Detect custom gestures (e.g., middle finger)
        landmarks = self.hands.process(rgb_frame)
        if landmarks.multi_hand_landmarks:
            for hand_landmarks in landmarks.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.detect_middle_finger(hand_landmarks):
                    result["middle_finger"] = 1.0

        # Check for wave gesture and maintain persistence for 10 frames
        if self.detect_wave():
            result["Wave"] = 1.0
            self.detected_wave = True  # Set flag to persist wave detection
            self.wave_counter = self.wave_persistence  # Reset counter
            self.wave_history.clear()
        elif self.detected_wave and self.wave_counter > 0:
            result["Wave"] = 1.0  # Keep wave detection active
            self.wave_counter -= 1  # Decrease counter
        else:
            self.detected_wave = False  # Reset flag after persistence ends

        return result
