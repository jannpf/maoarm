import sys
import os

# Dynamically add the YOLOv7 repository to the Python path
yolov7_path = os.path.abspath("yolov7")
sys.path.append(yolov7_path)

import torch
import cv2
from models.experimental import attempt_load  # Correct path to YOLOv7 function
from utils.general import non_max_suppression, scale_coords

# Load the YOLOv7-tiny model
model_path = "models/yolov7-tiny.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(model_path, map_location=device).eval()

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Preprocess frame
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        results = model(img)[0]
        results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.45)

    # Draw bounding boxes
    for det in results:
        if len(det):
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f"Face detected: ({x1}, {y1}) to ({x2}, {y2})")

    # Show frame
    cv2.imshow("YOLOv7 - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
