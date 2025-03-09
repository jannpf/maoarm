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

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.wave_frames_window = wave_frames_window
        self.wave_movement_threshold = wave_movement_threshold
        self.max_history_length = max_history_length
        self.right_hand_x_history = []
        self.left_hand_x_history = []
        self.missing_hand_frames = {"right": 0, "left": 0}

        # History tracking for wave detection
        self.wave_history = []  # Stores last 30 frame detections
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
    
    def detect_wave(self, hand_x_history) -> bool:
        """
        Detects the wave motion based on the hand's x-coordinate movements.
        """
        if len(hand_x_history) < self.wave_frames_window:
            return False

        recent_x = hand_x_history[-self.wave_frames_window:]
        differences = [recent_x[i] - recent_x[i - 1] for i in range(1, len(recent_x))]

        # Track direction changes and movements
        cumulative_movement = 0
        movement_count = 0
        last_direction = 0

        for diff in differences:
            direction = 1 if diff > 0 else -1
            if direction == last_direction or last_direction == 0:
                cumulative_movement += diff
            else:
                if abs(cumulative_movement) >= self.wave_movement_threshold:
                    movement_count += 1
                cumulative_movement = diff
            last_direction = direction

        if abs(cumulative_movement) >= self.wave_movement_threshold:
            movement_count += 1

        return movement_count >= 3

    def detect(self, frame) -> dict:
        """
        Detects gestures (including wave detection) in the given frame.
        """
        result = {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Process gestures using the recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognition_result = self.gesture_recognizer.recognize(mp_image)

        for g in recognition_result.gestures:
            detected_category = g[0]
            result[detected_category.category_name] = detected_category.score

        # Detect custom gestures (e.g., middle finger and wave)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.detect_middle_finger(hand_landmarks):
                    result["middle_finger"] = 1.0

                # Wave detection for left and right hands
                hand_label = results.multi_handedness[0].classification[0].label
                if hand_label == 'Right' and hand_landmarks:
                    wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                    self.right_hand_x_history.append(wrist_x)
                    if self.detect_wave(self.right_hand_x_history):
                        result["right_wave"] = 1.0
                elif hand_label == 'Left' and hand_landmarks:
                    wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                    self.left_hand_x_history.append(wrist_x)
                    if self.detect_wave(self.left_hand_x_history):
                        result["left_wave"] = 1.0

        return result
