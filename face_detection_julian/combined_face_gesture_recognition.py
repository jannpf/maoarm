import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_dir = os.path.dirname(os.path.abspath(__file__))
face_model_file = os.path.join(base_dir, "models", "Res10_300x300_ssd_iter_140000.caffemodel")
face_config_file = os.path.join(base_dir, "models", "deploy.prototxt")

face_net = cv2.dnn.readNetFromCaffe(face_config_file, face_model_file)

gesture_model_path = os.path.join(base_dir, 'models', 'gesture_recognizer.task')

# Configure Gesture Recognizer
base_options = python.BaseOptions(model_asset_path=gesture_model_path)
gesture_options = vision.GestureRecognizerOptions(base_options=base_options)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

# Initialize MediaPipe Hands for landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(0)

import math

def calculate_angle(a, b, c):
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

def detect_middle_finger(hand_landmarks):
    """
    Improved middle finger detection using PIP as a reference.
    """
    # Landmarks
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # Angle check for the middle finger: MCP -> PIP -> TIP
    middle_angle = calculate_angle(middle_mcp, middle_pip, middle_tip)

    # Conditions for middle finger raised
    middle_raised = middle_tip.y < middle_pip.y  # Middle finger tip is above PIP
    middle_straight = middle_angle > 160  # Middle finger is straight

    # Other fingers should be folded
    index_folded = index_tip.y > index_pip.y
    ring_folded = ring_tip.y > ring_pip.y
    pinky_folded = pinky_tip.y > pinky_pip.y

    if middle_raised and middle_straight and index_folded and ring_folded and pinky_folded:
        return "Middle Finger"
    return None


while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Face Detection
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (800, 600))
    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    largest_face = None
    largest_area = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x1, y1) = box.astype("int")
            area = (x1 - x) * (y1 - y)
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, x1, y1)

    if largest_face:
        x, y, x1, y1 = largest_face
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
        print(f"Largest face coordinates: Top Left ({x}, {y}), Bottom Right ({x1}, {y1})")

    # Gesture Recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    recognition_result = gesture_recognizer.recognize(mp_image)

    gesture_name = None
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        gesture_name = top_gesture.category_name
        confidence = top_gesture.score
        print(f"Gesture: {gesture_name}, Confidence: {confidence:.2f}")

    # Detect custom gestures (e.g., middle finger)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            custom_gesture = detect_middle_finger(hand_landmarks)
            if custom_gesture:
                gesture_name = custom_gesture

    if gesture_name:
        cv2.putText(frame, gesture_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face and Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
