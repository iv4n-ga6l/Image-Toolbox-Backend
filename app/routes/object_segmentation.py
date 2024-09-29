from flask import Blueprint, request, jsonify, send_file
from app.services.object_segmentation import get_object_seg
import io

object_segmentation = Blueprint('object_segmentation', __name__)

@object_segmentation.route('/segment_objects', methods=['POST'])
def segment_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    model_name = request.args.get('model', default='yolov8_seg')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            detector = get_object_seg(model_name)
            processed_image = detector.segment_objects(file)
            return send_file(io.BytesIO(processed_image), mimetype='image/jpeg'), 200
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400