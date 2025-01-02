import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def detect_gesture(hand_landmarks):
    """
    Detects gestures based on hand landmarks.
    """

   

     # Get landmarks
    # Thumb and other fingers' landmark positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # Use MCP for thumb

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]


     # Threshold for "raised finger" (tolerance for slight elevation)
    #threshold = -0.15  # Tune this based on the camera resolution

     # Check relative positions
    thumb_index_distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

    thumb_index_distance = index_tip.y - index_mcp.y
    print(f"Index tip mcp Distance: {thumb_index_distance}")

    # High-Five Detection: All fingers extended and spread
    if (
        thumb_tip.y < thumb_mcp.y  # Thumb extended upward (relative to MCP)
        and index_tip.y < index_mcp.y  # Index finger extended
        and middle_tip.y < middle_mcp.y  # Middle finger extended
        and ring_tip.y < ring_mcp.y  # Ring finger extended
        and pinky_tip.y < pinky_mcp.y  # Pinky finger extended
        and thumb_tip.x < index_tip.x - 0.05  # Thumb spread out to the side
    ):
        return "High-Five ✋"

     # Threshold for defining "finger folded"
    threshold = 0.02  # Smaller values make finger folding strict

    # Middle Finger Detection
    middle_raised = middle_tip.y < middle_mcp.y - threshold  # Middle finger raised
    index_folded = index_tip.y > index_mcp.y  # Index finger folded
    ring_folded = ring_tip.y > ring_mcp.y  # Ring finger folded
    pinky_folded = pinky_tip.y > pinky_mcp.y  # Pinky finger folded
    thumb_folded = thumb_tip.y > thumb_mcp.y  # Thumb can be relaxed (inside or outside)

    if (
        middle_raised
        and index_folded
        and ring_folded
        and pinky_folded
        and thumb_folded  # Thumb inside or relaxed
    ):
        return "☹️"  # Middle finger gesture
    
    # OK Gesture Detection (relaxed for extended fingers)
    if (
        thumb_index_distance < 0.035  # Thumb and index finger close together
        #and thumb_tip.y < thumb_ip.y  # Thumb is bent slightly
        #and index_tip.y < index_mcp.y  # Index finger is up
        and middle_tip.y > middle_mcp.y + threshold  # Middle finger extended slightly
        and ring_tip.y > ring_mcp.y + threshold  # Ring finger extended slightly
        and pinky_tip.y > pinky_mcp.y + threshold  # Pinky finger extended slightly
    ):
        return "OK Gesture"
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
        and index_tip.y > index_mcp.y  # Index finger down
        and middle_tip.y > middle_mcp.y  # Middle finger down
        and ring_tip.y > ring_mcp.y  # Ring finger down
        and pinky_tip.y > pinky_mcp.y  # Pinky finger down
    ):
        return "Thumbs Down"
    
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
                # Use PIL for Unicode-friendly text rendering
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                #font = ImageFont.truetype("arial.ttf", 32)  
                font = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 32)

                # Draw gesture on the frame
                draw.text((50, 50), gesture, font=font, fill=(0, 255, 0))

                # Convert back to OpenCV format
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                print(gesture)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
