import cv2
from modules.detection import detect_face_and_pose
from modules.recognition import recognize_gesture, recognize_body_emotion
from modules.models_loader import load_models, load_encodings_from_file
import warnings


warnings.filterwarnings('ignore')

# Initialize models
gesture_recognizer, body_language_model, emotion_detection_model = load_models()

# Base position for the text to display on screen
start_x = 10  # x-coordinate (left margin)
start_y = 30  # y-coordinate for the first line
line_spacing = 40  # Spacing between lines
font_scale = 1.2  # Increase this value for larger text

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip the frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face, pose, and draw landmarks
    frame = detect_face_and_pose(frame, rgb_frame)
    
    # Recognize gesture, body pose, and emotion
    recognize_gesture(rgb_frame, frame, gesture_recognizer, start_x, start_y, line_spacing, font_scale)
    recognize_body_emotion(rgb_frame, frame, body_language_model, emotion_detection_model, start_x, start_y, line_spacing, font_scale)

    # frame = gesture_detector.process_frame(frame)

    # Display the frame
    cv2.imshow("Face, Gesture, Emotion, and Pose Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()