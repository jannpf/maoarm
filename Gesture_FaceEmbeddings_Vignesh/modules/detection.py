import mediapipe as mp
import cv2
import face_recognition
from modules.models_loader import load_encodings_from_file
import numpy as np 

# Initialize MediaPipe components
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize models
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Custom drawing styles
face_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
pose_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)
hand_landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)
connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)



def detect_face_and_pose(frame, rgb_frame):
    #detect offerder face encoding:
    offender_face_encodings = load_encodings_from_file()
    # Detect face and pose landmarks and draw them on the frame.
    
    # Face Detection
    face_result = face_detection.process(rgb_frame)
    if face_result.detections:
        for detection in face_result.detections:
            mp_drawing.draw_detection(frame, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            confidence_score = int(detection.score[0] * 100)

            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

            # Detect and encode faces in the current frame
            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
            offender_detected = False
            if encodeCurFrame:  # Ensure at least one face encoding is found
                new_encoding = encodeCurFrame[0] 
                # Check if this encoding matches any in the stored_encodings
                matches = face_recognition.compare_faces(offender_face_encodings, new_encoding, tolerance=0.6)

                if not any(matches):  # If no matches found, store the new encoding
                    # print("New face encoding stored!")
                    offender_detected = False                    
                else:
                    offender_detected = True
                    # print("offender Detected.")
            
            formatted_confidence = f'{confidence_score:.2f}'  # Two decimal places

            # Dynamic text
            label_text = f'{formatted_confidence}% {"Offender" if offender_detected else "New Face"}'
            cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No face detected.")

    # Pose and Hand Detection
    holistic_results = holistic.process(rgb_frame)
    if holistic_results.face_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=face_landmark_style)
    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=pose_landmark_style, connection_drawing_spec=connection_style)
    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=hand_landmark_style, connection_drawing_spec=connection_style)
    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=hand_landmark_style, connection_drawing_spec=connection_style)

    return frame
