


import cv2
import mediapipe as mp


class dynamic_gesture_detection:

    def __init__(
        self,
        wave_frames_window=10,
        wave_movement_threshold=0.04,
    ):
        
        # Number of frames to store for detecting wave
        self.wave_frames_window = wave_frames_window
        # X-axis movement threshold (normalized) for wave detection
        self.wave_movement_threshold = wave_movement_threshold

        # keeping a rolling x-position history for each hand:
        self.right_hand_x_history = []
        self.left_hand_x_history = []

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
       
        # Flip horizontally for a selfie-view
        frame = cv2.flip(frame, 1)

        # Converting the BGR image to RGB for MediaPipe
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
            wrist_x = right_hand_landmarks[0][0]
            self._track_hand_x(self.right_hand_x_history, wrist_x)
            if self.detect_wave(self.right_hand_x_history):
                cv2.putText(frame, "Right Hand Waving", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if left_hand_landmarks:
            wrist_x = left_hand_landmarks[0][0]
            self._track_hand_x(self.left_hand_x_history, wrist_x)
            if self.detect_wave(self.left_hand_x_history):
                cv2.putText(frame, "Left Hand Waving", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def detect_wave(self, hand_x_history):
        if len(hand_x_history) < self.wave_frames_window:
            return False
        recent_x = hand_x_history[-self.wave_frames_window:]
        if (max(recent_x) - min(recent_x)) > self.wave_movement_threshold:
            return True
        return False


    def _track_hand_x(self, history_list, x_normalized):
        history_list.append(x_normalized)
        if len(history_list) > 50:
            history_list.pop(0)
            
            
    
    def main():
        gesture_detector = dynamic_gesture_detection()

        # Open a webcam feed
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
           print("Error: Could not open webcam.")
           return  

        try:
            while True:
                 ret, frame = cap.read()
                 if not ret:
                    print("Error: Could not read frame from webcam.")
                    break

                 frame = gesture_detector.process_frame(frame)
                 cv2.imshow("Dynamic Gesture Detection", frame)

                 # Break the loop if 'q' is pressed
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
             cap.release()
             cv2.destroyAllWindows()


    if __name__ == "__main__":
        main()
        
        