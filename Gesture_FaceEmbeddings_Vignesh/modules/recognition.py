import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import face_recognition
import pickle
import os
from modules.models_loader import load_encodings_from_file




# if ret:
#     imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#     faceCurFrame = face_recognition.face_locations(imgS)
#     storedEncodings = face_recognition.face_encodings(imgS, faceCurFrame)

# Function to save encodings to a pickle file
def save_encodings_to_file(encodings, folder_path='models', file_name='stored_face_encodings.pkl'):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'wb') as file:
        pickle.dump(encodings, file)
    print(f"Encodings saved to {file_name}")

# Initialize the variable to store the encoding
storedEncodings = load_encodings_from_file()

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize models
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def recognize_gesture(rgb_frame, frame, gesture_recognizer, start_x, start_y, line_spacing, font_scale):
    
    # Recognize gestures using the gesture recognizer model.
    rgb_frame = cv2.flip(rgb_frame, 1)  # Flip the frame horizontally
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    try:
        recognition_result = gesture_recognizer.recognize(mp_image)
        if recognition_result.gestures:
            hand_name = recognition_result.handedness[0][0].category_name
            top_gesture = recognition_result.gestures[0][0]
            gesture_name = top_gesture.category_name
            confidence = top_gesture.score
            if gesture_name == "ILoveYou":
                # Resize and convert frame
                imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

                # Detect and encode faces in the current frame
                faceCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

                if encodeCurFrame:  # Ensure at least one face encoding is found
                    new_encoding = encodeCurFrame[0] 
                    # Check if this encoding matches any in the stored_encodings
                    matches = face_recognition.compare_faces(storedEncodings, new_encoding, tolerance=0.6)

                    if not any(matches):  # If no matches found, store the new encoding
                        storedEncodings.append(new_encoding)
                        print("New face encoding stored!")
                        save_encodings_to_file(storedEncodings, folder_path='models')
                    else:
                        print("Face already recorded.")

        else:
            hand_name, gesture_name = "None", "None"
            # pass

        cv2.putText(frame, f"Hand : {hand_name} {gesture_name}", (start_x, start_y + 4 * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    except:
        pass

def recognize_body_emotion(rgb_frame, frame, body_language_model, emotion_detection_model, start_x, start_y, line_spacing, font_scale):
    
    # Recognize body pose and emotion using the pre-trained models.
    try:
        # Extract Pose landmarks
        holistic_results = holistic.process(rgb_frame)
        pose = holistic_results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                  for landmark in pose]).flatten())

        # Extract Face landmarks
        face = holistic_results.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                  for landmark in face]).flatten())

        # Make Body Pose Prediction
        X = pd.DataFrame([pose_row])
        body_language_class = body_language_model.predict(X)[0]
        body_language_prob = body_language_model.predict_proba(X)[0]
        cv2.putText(frame, f"Pose : {body_language_class}", (start_x, start_y + 2 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        # Make Emotion Prediction
        Y = pd.DataFrame([face_row])
        emotion_detection_class = emotion_detection_model.predict(Y)[0]
        emotion_detection_prob = emotion_detection_model.predict_proba(Y)[0]
        cv2.putText(frame, f"Emotion : {emotion_detection_class}", (start_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    except:
        pass