# app/services/object_detection/yolov8_detector.py
from .base_detector import BaseObjectDetector
from app.utils.image_utils import get_base_path
import os
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv8Detector(BaseObjectDetector):
    def __init__(self):
        # Check if local model file exists
        local_model_path = os.path.join(get_base_path(), 'yolov8n.pt')

        if os.path.exists(local_model_path):
            print(f"Loading YOLOv8 model from local file: {local_model_path}")
            self.model = YOLO(local_model_path)
        else:
            print("Local YOLOv8 model not found. Downloading from Ultralytics...")
            self.model = YOLO('yolov8n.pt')

    def detect_objects(self, image):
        # Convert image to RGB (YOLOv8 expects RGB images)
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Convert the annotated frame back to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        return buffer.tobytes()

    def count_objects(self, image):
        # Convert image to RGB (YOLOv8 expects RGB images)
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Count objects
        object_counts = {}
        for result in results:
            for c in result.boxes.cls:
                class_name = self.model.names[int(c)]
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1

        return object_counts