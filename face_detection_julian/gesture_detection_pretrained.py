import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(os.path.dirname(__file__), 'models', 'gesture_recognizer.task')

# Configure Gesture Recognizer
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Initialize MediaPipe Hands for landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(0)

def detect_middle_finger(hand_landmarks):
    """
    Custom logic for detecting the middle finger gesture.
    """
    # Get landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Threshold for "raised finger"
    threshold = 0.05

    # Check middle finger raised
    middle_raised = middle_mcp.y - middle_tip.y > threshold

    # Check other fingers folded
    #thumb_folded = thumb_tip.y > thumb_mcp.y
    index_folded = index_tip.y > index_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y

    # Detect middle finger gesture
    if middle_raised and index_folded and ring_folded and pinky_folded:
        return "Middle Finger"
    return None

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize(mp_image)

    gesture_name = None
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        gesture_name = top_gesture.category_name
        confidence = top_gesture.score
        print(f"Gesture: {gesture_name}, Confidence: {confidence:.2f}")


    custom_gesture = None
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            custom_gesture = detect_middle_finger(hand_landmarks)

    # Display gesture on video feed
    if custom_gesture:
        gesture_name = custom_gesture

    if gesture_name:
        cv2.putText(frame, gesture_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
