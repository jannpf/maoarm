import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def detect_gesture(hand_landmarks):
    """
    Detects gestures based on hand landmarks.
    """
    # Get landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Calculate distance between thumb tip and index tip (for OK gesture)
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

    # Thumbs-Up Detection
    if (
        thumb_tip.y < index_mcp.y  # Thumb raised
        and index_tip.y > index_mcp.y  # Index finger down
        and middle_tip.y > middle_mcp.y  # Middle finger down
        and ring_tip.y > ring_mcp.y  # Ring finger down
        and pinky_tip.y > pinky_mcp.y  # Pinky finger down
    ):
        return "Thumbs Up"

    # Thumbs-Down Detection
    elif (
        thumb_tip.y > index_mcp.y  # Thumb down
        and index_tip.y > index_mcp.y  # Index finger relaxed
        and middle_tip.y > middle_mcp.y  # Middle finger relaxed
        and ring_tip.y > ring_mcp.y  # Ring finger relaxed
        and pinky_tip.y > pinky_mcp.y  # Pinky finger relaxed
    ):
        return "Thumbs Down"

    # OK Gesture Detection
    elif (
        distance < 0.05  # Thumb and index finger are close
        and middle_tip.y < middle_mcp.y  # Middle finger stretched
        and ring_tip.y < ring_mcp.y  # Ring finger stretched
        and pinky_tip.y < pinky_mcp.y  # Pinky finger stretched
    ):
        return "OK Gesture"
    return None

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks and detect gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(hand_landmarks)
            if gesture:
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(gesture)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
