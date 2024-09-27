from flask import Blueprint, request, jsonify, send_file
from app.services.object_detection import get_object_detector
import io

object_detection = Blueprint('object_detection', __name__)

@object_detection.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    model_name = request.args.get('model', default='yolov3')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            detector = get_object_detector(model_name)
            processed_image = detector.detect_objects(file)
            return send_file(io.BytesIO(processed_image), mimetype='image/jpeg'), 200
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

@object_detection.route('/count_objects', methods=['POST'])
def count_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    model_name = request.args.get('model', default='yolov3')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            detector = get_object_detector(model_name)
            object_counts = detector.count_objects(file)
            return jsonify(object_counts), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400