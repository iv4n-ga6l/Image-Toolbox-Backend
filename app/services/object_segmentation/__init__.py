from .yolov8_seg import YOLOv8Seg
from .yolo11_seg import YOLOv11Seg

def get_object_seg(model_name):
    if model_name == 'yolov8_seg':
        return YOLOv8Seg()
    if model_name == 'yolo11_seg':
        return YOLOv11Seg()
    else:
        raise ValueError(f"Unsupported model: {model_name}")