import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dropbox
import json


def create_config(file_path="config.json"):
    config = {
        "dropbox": {
            "access_token": "YDROPBOX_ACCESS_TOKEN"
        }
    }

    try:
        with open(file_path, 'w') as file:
            json.dump(config, file, indent=4)
        print(f"Configuration file '{file_path}' created successfully.")
    except Exception as e:
        print(f"Failed to create configuration file: {e}")


def load_config(file_path="config.json"):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file '{file_path}' not found. Creating a new one.")
        create_config(file_path)
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


# Variables for face embeddings
known_faces = []
known_labels = []

# Dropbox configuration
# it should run just at the first time
# config=create_config(file_path="config.json")
load_cfg = load_config(file_path="config.json")
DROPBOX_ACCESS_TOKEN = load_cfg['dropbox']['access_token']
dropbox_client = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


def save_to_dropbox(file_path, dropbox_path):
    with open(file_path, 'rb') as f:
        dropbox_client.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)


################################################### Function to calculate the embedding of a face##################################################################

def calculate_embedding(face_img):
    resized_face = cv2.resize(face_img, (100, 100))
    return resized_face.flatten()


# This function shoulde be replaced with the main code for face detecting
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def process_faces(frame, faces, output_dir):
    global known_faces, known_labels

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        embedding = calculate_embedding(face_img)

        # Check if the face is already known
        found = False
        if known_faces:
            similarities = cosine_similarity([embedding], known_faces)
            max_similarity = np.max(similarities)
            if max_similarity > 0.8:
                found = True
                label_index = np.argmax(similarities)
                label = known_labels[label_index]
                print(f"This face is known: {label}")
                cv2.putText(frame, f"Known Face: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not found:
            # save the face as a new known face
            new_label = f"Face_{len(known_faces) + 1}"
            known_faces.append(embedding)
            known_labels.append(new_label)
            face_filename = os.path.join(output_dir, f"{new_label}.jpg")
            cv2.imwrite(face_filename, face_img)

            # Save the image to Dropbox
            dropbox_path = f"/detected_faces/{new_label}.jpg"
            save_to_dropbox(face_filename, dropbox_path)

            cv2.putText(frame, "New Face Saved", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def main():
    # Load Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # save the detected face images
    output_dir = "detected_faces"
    os.makedirs(output_dir, exist_ok=True)

    # video capture from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Detect faces in the frame
        faces = detect_faces(frame, face_cascade)

        # Process each detected face
        process_faces(frame, faces, output_dir)

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        #  if 'q' is pressed it will break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()