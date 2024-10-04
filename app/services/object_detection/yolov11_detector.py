# app/services/object_detection/yolov11_detector.py
from .base_detector import BaseObjectDetector
from app.utils.image_utils import get_base_path
import os
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv11Detector(BaseObjectDetector):
    def __init__(self):
        # Check if local model file exists
        local_model_path = os.path.join(get_base_path(), 'yolo11n.pt')

        if os.path.exists(local_model_path):
            print(f"Loading YOLO11 model from local file: {local_model_path}")
            self.model = YOLO(local_model_path)
        else:
            print("Local YOLO11 model not found. Downloading from Ultralytics...")
            self.model = YOLO('yolo11n.pt')

    def detect_objects(self, image):
        # Convert image to RGB (YOLOv8 expects RGB images)
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Create a copy of the original image to draw on
        annotated_frame = image_rgb.copy()
        
        # Loop through the detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label
                label = f'{self.model.names[cls]} {conf:.2f}'
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw filled rectangle for text background
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Put text on the image
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Convert the annotated frame back to BGR for OpenCV encoding
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Convert the annotated frame back to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame_bgr)
        return buffer.tobytes()

    def count_objects(self, image):
        # Convert image to RGB (YOLOv8 expects RGB images)
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Count objects
        object_counts = {}
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
        
        return object_counts