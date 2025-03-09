import cv2
import mediapipe as mp
import sys

class MediapipeWaves:
    def __init__(self, hands, wave_frames_window=30, wave_movement_threshold=0.05, max_history_length=50):
        self.hands = hands
        self.wave_frames_window = wave_frames_window
        self.wave_movement_threshold = wave_movement_threshold
        self.max_history_length = max_history_length
        self.right_hand_x_history = []
        self.left_hand_x_history = []
        self.missing_hand_frames = {"right": 0, "left": 0}  

        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame) -> dict:
        result = {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        left_hand_landmarks = None
        right_hand_landmarks = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_info in enumerate(results.multi_handedness):
                label = hand_info.classification[0].label
                hand_landmarks = results.multi_hand_landmarks[idx]
                lm_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                if label == 'Right':
                    right_hand_landmarks = lm_list
                else:
                    left_hand_landmarks = lm_list

                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Process hand movement
        if right_hand_landmarks:
            self.missing_hand_frames["right"] = 0
            wrist_x = right_hand_landmarks[0][0]
            self._track_hand_x(self.right_hand_x_history, wrist_x)
            if self.detect_wave(self.right_hand_x_history):
                result["right_wave"] = 1.0
        else:
            self._handle_missing_hand("right")

        if left_hand_landmarks:
            self.missing_hand_frames["left"] = 0
            wrist_x = left_hand_landmarks[0][0]
            self._track_hand_x(self.left_hand_x_history, wrist_x)
            if self.detect_wave(self.left_hand_x_history):
                result["left_wave"] = 1.0
        else:
            self._handle_missing_hand("left")

        return result

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
        if len(hand_x_history) < self.wave_frames_window:
            print("Insufficient history for wave detection.")
            return False

        recent_x = hand_x_history[-self.wave_frames_window:]
        differences = [recent_x[i] - recent_x[i - 1] for i in range(1, len(recent_x))]

        # Variables to track cumulative movement and direction
        cumulative_movement = 0
        last_direction = 0
        movement_count = 0

        for diff in differences:
            direction = 1 if diff > 0 else -1  # Determine current direction

            if direction == last_direction or last_direction == 0:
                # Accumulate movement in the same direction
                cumulative_movement += diff
            else:
                # Direction changed, check if the last movement exceeded the threshold
                if abs(cumulative_movement) >= self.wave_movement_threshold:
                    movement_count += 1
                    print(f"Movement {movement_count} detected with amplitude {abs(cumulative_movement)}")
                
                # Reset for the new direction
                cumulative_movement = diff

            # Update the last direction
            last_direction = direction

        # Check the final movement
        if abs(cumulative_movement) >= self.wave_movement_threshold:
            movement_count += 1
            print(f"Final movement {movement_count} detected with amplitude {abs(cumulative_movement)}")

        print(f"Total movements: {movement_count}, Threshold: {self.wave_movement_threshold}")

        # A wave is detected if there are at least 3 movements with direction changes
        if movement_count >= 3:
            print("Wave detected!")
            return True

        return False

    def _track_hand_x(self, history_list, x_normalized):
        history_list.append(x_normalized)

        print(f"Updated history: {history_list[-10:]}")  # Print last 10 values

        if len(history_list) > self.max_history_length:  # Limit history size
            history_list.pop(0)



def main():
    wave_detector = MediapipeWaves()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            wave_results = wave_detector.detect(frame)

            if "right_wave" in wave_results:
                cv2.putText(frame, "Right Hand Waving", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if "left_wave" in wave_results:
                cv2.putText(frame, "Left Hand Waving", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Wave Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()