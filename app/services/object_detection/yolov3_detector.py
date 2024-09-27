# app/services/object_detection/yolov3_detector.py
from .base_detector import BaseObjectDetector
from app.utils.image_utils import get_base_path
import os
import cv2
import numpy as np

class YOLOv3Detector(BaseObjectDetector):
    def __init__(self):
        # Load YOLOv3 model and configuration
        weights_path = os.path.join(get_base_path(), 'yolov3.weights')
        cfg_path = os.path.join(get_base_path(), 'yolov3.cfg')
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        
        # Load class names
        classes_path = os.path.join(get_base_path(), 'coco.names')
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image):
        # Load the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        # Preprocess the image
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Detection information
        class_ids = []
        confidences = []
        boxes = []

        # Analyze network outputs
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Bounding box coordinates (upper left corner)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Remove redundant detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the results on the image
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert OpenCV image to bytes
        _, img_encoded = cv2.imencode('.jpg', img)
        
        return img_encoded.tobytes()

    def count_objects(self, image):
        # Load the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        # Preprocess the image
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Detection information
        object_counts = {}

        # Analyze the network's outputs
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = self.classes[class_id]
                    object_counts[label] = object_counts.get(label, 0) + 1

        return object_counts