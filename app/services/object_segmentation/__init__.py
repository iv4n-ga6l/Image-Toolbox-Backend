from .yolov8_seg import YOLOv8Seg

def get_object_seg(model_name):
    if model_name == 'yolov8_seg':
        return YOLOv8Seg()
    else:
        raise ValueError(f"Unsupported model: {model_name}")