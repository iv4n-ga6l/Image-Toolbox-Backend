from .yolov3_detector import YOLOv3Detector
from .yolov5_detector import YOLOv5Detector
from .yolov7_detector import YOLOv7Detector
from .yolov8_detector import YOLOv8Detector
from .yolov10_detector import YOLOv10Detector
from .yolov11_detector import YOLOv11Detector

def get_object_detector(model_name):
    if model_name == 'yolov3':
        return YOLOv3Detector()
    elif model_name == 'yolov5':
        return YOLOv5Detector()
    elif model_name == 'yolov7':
        return YOLOv7Detector()
    elif model_name == 'yolov8':
        return YOLOv8Detector()
    elif model_name == 'yolov10':
        return YOLOv10Detector()
    elif model_name == 'yolov11':
        return YOLOv11Detector()
    else:
        raise ValueError(f"Unsupported model: {model_name}")