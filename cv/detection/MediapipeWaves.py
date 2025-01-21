from .DetectionBase import DetectionBase
import cv2
import mediapipe as mp

import logging

logging.basicConfig(level=logging.DEBUG) 
logger = logging.getLogger(__name__)


class MediapipeWaves:
    def __init__(self, wave_frames_window=30, wave_movement_threshold=0.01, max_history_length=200):

        self.wave_frames_window = wave_frames_window
        self.wave_movement_threshold = wave_movement_threshold
        self.max_history_length = max_history_length  # Limit for history length
        self.right_hand_x_history = []
        self.left_hand_x_history = []
        self.missing_hand_frames = {"right": 0, "left": 0}  # Track frames with no hand

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame) -> dict:
        logger.debug("Testing MediaPipe Hands directly in detect()")

        logger.debug(f"Frame dimensions: {frame.shape}")
        logger.debug(f"Frame type: {type(frame)}")
        logger.debug(f"Frame dtype: {frame.dtype}")

        cv2.imshow("Debug Frame", frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        logger.debug(f"MediaPipe Hands results: {results}")
        logger.debug(f"Results object details: {vars(results) if results else 'No results found.'}")

        if results.multi_hand_landmarks:
            logger.debug(f"Hand landmarks detected: {results.multi_hand_landmarks}")
        else:
            logger.debug("No hand landmarks detected.")

        return {}

    # def detect(self, frame) -> dict:

    #     logger.debug("Processing frame in MediapipeWaves.detect()")

    #     result = {}
    #     logger.debug(f"Original frame shape: {frame.shape}")
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     logger.debug(f"Converted RGB frame shape: {rgb_frame.shape}")

    #     results = self.hands.process(rgb_frame)
    #     logger.debug(f"MediaPipe processing results: {results}")
    #     if results.multi_hand_landmarks:
    #         logger.debug(f"Hand landmarks detected: {results.multi_hand_landmarks}")
    #     else:
    #         logger.debug("No hand landmarks detected.")

    #     left_hand_landmarks = None
    #     right_hand_landmarks = None
    #     if results.multi_hand_landmarks and results.multi_handedness:
    #         for idx, hand_info in enumerate(results.multi_handedness):
    #             label = hand_info.classification[0].label
    #             hand_landmarks = results.multi_hand_landmarks[idx]
    #             lm_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    #             if label == 'Right':
    #                 right_hand_landmarks = lm_list
    #             else:
    #                 left_hand_landmarks = lm_list

    #             self.mp_draw.draw_landmarks(
    #                 frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
    #             )

    #     # Handle right hand
    #     if right_hand_landmarks:
    #         logger.debug(f"Right hand detected with landmarks: {right_hand_landmarks}")
    #         self.missing_hand_frames["right"] = 0  # Reset missing frames counter
    #         wrist_x = right_hand_landmarks[0][0]

    #         self._track_hand_x(self.right_hand_x_history, wrist_x)
    #         logger.debug("Calling detect_wave for the right hand.")
    #         if self.detect_wave(self.right_hand_x_history):
    #             result["right_wave"] = 1.0
    #     # else:
    #     #     self._handle_missing_hand("right")
    #     else:
    #         logger.debug("No right hand detected.")

    #     # Handle left hand
    #     if left_hand_landmarks:
    #         logger.debug(f"Left hand detected with landmarks: {left_hand_landmarks}")
    #         self.missing_hand_frames["left"] = 0  # Reset missing frames counter
    #         wrist_x = left_hand_landmarks[0][0]
    #         self._track_hand_x(self.left_hand_x_history, wrist_x)
    #         logger.debug("Calling detect_wave for the left hand.")
    #         if self.detect_wave(self.left_hand_x_history):
    #             result["left_wave"] = 1.0
    #     # else:
    #     #     self._handle_missing_hand("left")
    #     else:
    #         logger.debug("No left hand detected.")
    #     return result

    def _handle_missing_hand(self, hand: str):
        """
        Handles cases where a hand is not detected.
        Resets history if the hand is missing for multiple frames.
        """
        self.missing_hand_frames[hand] += 1
        if hand == "right":
            if self.missing_hand_frames["right"] > self.wave_frames_window:
                self.right_hand_x_history = []  # Clear history
        elif hand == "left":
            if self.missing_hand_frames["left"] > self.wave_frames_window:
                self.left_hand_x_history = []  # Clear history

    def detect_wave(self, hand_x_history) -> bool:

        logger.debug(f"Starting wave detection. History length: {len(hand_x_history)}")

        if len(hand_x_history) < self.wave_frames_window:
            logger.debug("Insufficient history for wave detection.")
            return False

        recent_x = hand_x_history[-self.wave_frames_window:]
        logger.debug(f"Recent X values: {recent_x}")

        differences = [recent_x[i] - recent_x[i - 1] for i in range(1, len(recent_x))]
        logger.debug(f"Differences between consecutive X values: {differences}")

        # Variables to track cumulative movement and direction
        cumulative_movement = 0
        last_direction = 0
        movement_count = 0

        for index, diff in enumerate(differences):
            direction = 1 if diff > 0 else -1  # Determine current direction
            logger.debug(f"Index {index}: Difference {diff}, Direction {direction}")

            if direction == last_direction or last_direction == 0:
                # Accumulate movement in the same direction
                cumulative_movement += diff
            else:
                # Direction changed, check if the last movement exceeded the threshold
                if abs(cumulative_movement) >= self.wave_movement_threshold:
                    movement_count += 1
                    logger.debug(f"Movement {movement_count} detected with amplitude {abs(cumulative_movement)}")
                
                # Reset for the new direction
                cumulative_movement = diff

            # Update the last direction
            last_direction = direction
            logger.debug(f"Updated cumulative movement: {cumulative_movement}, Last direction: {last_direction}")

        # Check the final movement
        if abs(cumulative_movement) >= self.wave_movement_threshold:
            movement_count += 1
            logger.debug(f"Final movement {movement_count} detected with amplitude {abs(cumulative_movement)}")

        logger.debug(f"Total movements: {movement_count}, Movement threshold: {self.wave_movement_threshold}")

        # A wave is detected if there are at least 3 movements with direction changes
        if movement_count >= 3:
            logger.debug("Wave detected!")
            return True

        logger.debug("No wave detected.")
        return False

    def _track_hand_x(self, history_list, x_normalized):
        history_list.append(x_normalized)
        if len(history_list) > self.max_history_length:  # Limit history size
            history_list.pop(0)