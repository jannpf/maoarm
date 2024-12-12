import cv2
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(base_dir, "models", "Res10_300x300_ssd_iter_140000.caffemodel")
config_file = os.path.join(base_dir, "models", "deploy.prototxt")

net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Initialize webcam
# video_capture = cv2.VideoCapture("C:/Users/Julian/OneDrive/_Julian/Julian Master Leuphana/Semester 3/AI Lab/face_video.mp4")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize frame to a manageable size for display
    frame = cv2.resize(frame, (800, 600))

    # Prepare the frame for the DNN model
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    largest_face = None
    largest_area = 0

    # Loop over detections and find the largest face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x1, y1) = box.astype("int")
            area = (x1 - x) * (y1 - y)
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, x1, y1)

    # If a largest face is found, draw the bounding box and print coordinates
    if largest_face:
        x, y, x1, y1 = largest_face
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
        print(f"Largest face coordinates: Top Left ({x}, {y}), Bottom Right ({x1}, {y1})")

    # Display the resulting frame
    cv2.namedWindow("Video - Face Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Video - Face Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
