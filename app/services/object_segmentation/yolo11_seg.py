# app/services/object_segmentation/yolov11_seg.py
from app.utils.image_utils import get_base_path
import os
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv11Seg:
    def __init__(self):
        # Check if local model file exists
        local_model_path = os.path.join(get_base_path(), 'yolo11n-seg.pt')

        if os.path.exists(local_model_path):
            print(f"Loading YOLO11 Seg model from local file: {local_model_path}")
            self.model = YOLO(local_model_path)
        else:
            print("Local YOLO11 Seg model not found. Downloading from Ultralytics...")
            self.model = YOLO('yolo11n-seg.pt')

    def segment_objects(self, image):
        # Convert image to RGB
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = self.model(image_rgb)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Convert the annotated frame back to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        return buffer.tobytes()