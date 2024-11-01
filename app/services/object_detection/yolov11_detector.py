# app/services/object_detection/yolov11_detector.py
from .base_detector import BaseObjectDetector
from app.utils.image_utils import get_base_path
import os
import cv2
import numpy as np
from colorsys import hsv_to_rgb
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
        
        self.class_colors = {}
    
    def _get_class_color(self, class_name):
        """Generate and store a unique color for each class."""
        if class_name not in self.class_colors:
            # Generate colors using HSV color space for better distinction
            hue = len(self.class_colors) * 0.1 % 1.0  # Spread colors across hue spectrum
            # Convert HSV to RGB (using 100% saturation and value)
            rgb = hsv_to_rgb(hue, 1.0, 1.0)
            # Convert to BGR (OpenCV format) and scale to 0-255
            self.class_colors[class_name] = tuple(int(c * 255) for c in reversed(rgb))
        return self.class_colors[class_name]

    def detect_objects(self, image, confidence_threshold=0.25):
        """
        Detect objects in an image using YOLOv11 with class-specific colors.
        
        Args:
            image: Input image file object
            confidence_threshold (float): Confidence threshold for detections (0.0 to 1.0)
            
        Returns:
            tuple: (annotated image bytes, list of detections)
        """
        # Convert image to RGB
        image_rgb = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = self.model(image_rgb, conf=confidence_threshold)
        result = results[0]
        
        annotated_frame = image_rgb.copy()
        detections = []
        
        # Process each detection
        for box in result.boxes:
            confidence = float(box.conf)
            class_id = int(box.cls)
            class_name = result.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class-specific color
            color = self._get_class_color(class_name)
            
            detection_info = {
                'class': class_name,
                'confidence': confidence,
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                },
                'color': color 
            }
            detections.append(detection_info)
            
            # Draw bounding box with class-specific color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f'{class_name}: {confidence:.2f}'
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background with same class color
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text (white or black depending on background color brightness)
            # Calculate brightness using weighted RGB values
            brightness = sum(c * w for c, w in zip(color, [0.299, 0.587, 0.114]))
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
        
        # Convert annotated frame to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        
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