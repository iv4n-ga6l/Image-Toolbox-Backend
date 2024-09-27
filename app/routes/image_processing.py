import os
from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image
import pytesseract
from app.utils.image_utils import adjust_brightness, process_frame, load_graph_opt_model, get_base_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

image_processing = Blueprint('image_processing', __name__)

@image_processing.route('/resize_image', methods=['POST'])
def resize_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            width = request.args.get('width', default=300, type=int)
            height = request.args.get('height', default=300, type=int)

            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            resized_img = cv2.resize(img, (width, height))
            
            _, img_encoded = cv2.imencode('.jpg', resized_img)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

@image_processing.route('/apply_filter', methods=['POST'])
def apply_filter():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            filter_type = request.args.get('filter', default='blur', type=str)

            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            if filter_type == 'blur':
                filtered_img = cv2.blur(img, (5, 5))
            elif filter_type == 'sharpen':
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                filtered_img = cv2.filter2D(img, -1, kernel)
            elif filter_type == 'grayscale':
                filtered_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                return jsonify({"error": "Invalid filter type. Available options: blur, sharpen, grayscale"}), 400
            
            _, img_encoded = cv2.imencode('.jpg', filtered_img)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400
    
@image_processing.route('/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            image = Image.open(io.BytesIO(file.read()))
            
            extracted_text = pytesseract.image_to_string(image)
            
            return jsonify({"text": extracted_text}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

@image_processing.route('/enhance_image', methods=['POST'])
def enhance_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            brightness_factor = request.args.get('brightness', default=1.0, type=float)
            contrast_factor = request.args.get('contrast', default=1.0, type=float)
            
            enhanced_image = adjust_brightness(img, brightness_factor)
            
            enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast_factor, beta=0)
            
            _, img_encoded = cv2.imencode('.jpg', enhanced_image)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

@image_processing.route('/compress_image', methods=['POST'])
def compress_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            quality = request.args.get('quality', default=80, type=int)
            
            _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

@image_processing.route('/compare_images', methods=['POST'])
def compare_images():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both files are required"}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "Both files must be selected"}), 400
    
    if (file1 and file1.filename.lower().endswith(('.jpg', '.jpeg', '.png')) and 
        file2 and file2.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
        try:
            img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
            similarity_score = np.max(result)

            return jsonify({"similarity_score": float(similarity_score)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

@image_processing.route('/detect_open_poses', methods=['POST'])
def detect_open_poses():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            graph_opt_path = os.path.join(get_base_path(), 'open_pose.pb.pb')
            net = load_graph_opt_model(graph_opt_path)
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

            POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
            
            frame = process_frame(net, frame, 368, 368, BODY_PARTS, POSE_PAIRS, 0.2)
            _, img_encoded = cv2.imencode('.jpg', frame)
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400