import cv2
import mediapipe as mp

class MediapipeWaves:
    def __init__(self,wave_frames_window=30,wave_movement_threshold=0.05, max_history_length=50):
        self.wave_frames_window=wave_frames_window
        self.wave_movement_threshold = wave_movement_threshold
        self.max_history_length = max_history_length
        self.right_hand_x_history = []
        self.left_hand_x_history = []
        self.missing_hand_frames = {"right": 0, "left": 0}
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        # Detect hands
        left_hand_landmarks = None
        right_hand_landmarks = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_info in enumerate(results.multi_handedness):
                # "Left" or "Right"
                label = hand_info.classification[0].label
                hand_landmarks = results.multi_hand_landmarks[idx]
                lm_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                if label == 'Right':
                   right_hand_landmarks = lm_list
                else:
                   left_hand_landmarks = lm_list
                   # Draw landmarks
                self.mp_draw.draw_landmarks(
                       frame,
                       hand_landmarks,
                       self.mp_hands.HAND_CONNECTIONS
                )
        # Wave Detection
        if right_hand_landmarks:
            self.missing_hand_frames["right"] = 0  # Reset missing frames counter
            wrist_x = right_hand_landmarks[0][0]
            print(f"Right wrist x: {wrist_x}")
            self._track_hand_x(self.right_hand_x_history, wrist_x)
            if self.detect_wave(self.right_hand_x_history):
                print("Right hand wave detected!")
                cv2.putText(frame, "Right Hand Waving", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
                self._handle_missing_hand("right")
        if left_hand_landmarks:
            self.missing_hand_frames["left"] = 0  # Reset missing frames counter
            wrist_x = left_hand_landmarks[0][0]
            print(f"Left wrist x: {wrist_x}")
            self._track_hand_x(self.left_hand_x_history, wrist_x)
            if self.detect_wave(self.left_hand_x_history):
                print("Left hand wave detected!")
                cv2.putText(frame, "Left Hand Waving", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self._handle_missing_hand("left")

        return frame

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


    def detect_wave(self, hand_x_history):
        if len(hand_x_history) < self.wave_frames_window:
            print("Insufficient history for wave detection.")
            return False
        recent_x = hand_x_history[-self.wave_frames_window:]
        differences = [recent_x[i] - recent_x[i - 1] for i in range(1, len(recent_x))]
        if (max(recent_x) - min(recent_x)) > self.wave_movement_threshold:
            return True
        return False

    def _track_hand_x(self, history_list, x_normalized):
        history_list.append(x_normalized)
        if len(history_list) > self.max_history_length:  # Limit history size
            history_list.pop(0)

def main():
    wave_detector = MediapipeWaves()
    # Open a webcam feed
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
              frame = wave_detector.process_frame(frame)
              cv2.imshow("Dynamic Gesture Detection", frame)
              # Break the loop if 'q' is pressed
              if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

