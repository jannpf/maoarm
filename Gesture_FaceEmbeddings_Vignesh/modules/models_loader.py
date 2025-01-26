import os
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_models():
    # Load all models: gesture recognizer, body language, and emotion detection.
    
    # Load Gesture Recognizer
    model_path = os.path.join('models', 'gesture_recognizer.task')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gesture recognizer model not found at {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

    # Load Body Language Model
    with open(os.path.join('models', 'body_language.pkl'), 'rb') as f:
        body_language_model = pickle.load(f)

    # Load Emotion Detection Model
    with open(os.path.join('models', 'emotion_detection.pkl'), 'rb') as f:
        emotion_detection_model = pickle.load(f)

    return gesture_recognizer, body_language_model, emotion_detection_model

# Function to load encodings from the specified folder and file
def load_encodings_from_file(folder_path='models', file_name='stored_face_encodings.pkl'):
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, 'rb') as file:
            encodings = pickle.load(file)
        # print(f"Encodings loaded from {file_path}")
        return encodings
    except FileNotFoundError:
        print(f"No existing file found at {file_path}. Starting fresh.")
        return []  # Return an empty list if the file doesn't exist
