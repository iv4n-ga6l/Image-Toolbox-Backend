# app/services/object_detection/yolov5_detector.py
from .base_detector import BaseObjectDetector
from app.utils.image_utils import get_base_path
import os
import torch
import cv2
import numpy as np

class YOLOv5Detector(BaseObjectDetector):
    def __init__(self):
        # Check if local model file exists
        local_model_path = os.path.join(get_base_path(), 'yolov5s.pt')

        if os.path.exists(local_model_path):
            print(f"Loading YOLOv5 model from local file: {local_model_path}")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=local_model_path, force_reload=True)
        else:
            print("Local YOLOv5 model not found. Downloading from torch hub...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def detect_objects(self, image):
        # Convert image to RGB (YOLOv5 expects RGB images)
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()
        
        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:  # Confidence threshold
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert the image back to bytes
        _, buffer = cv2.imencode('.jpg', image_rgb)
        return buffer.tobytes()

    def count_objects(self, image):
        # Convert image to RGB (YOLOv5 expects RGB images)
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()
        
        # Count objects
        object_counts = {}
        for detection in detections:
            _, _, _, _, conf, cls = detection
            if conf > 0.5:  # Confidence threshold
                label = self.model.names[int(cls)]
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

        return object_counts