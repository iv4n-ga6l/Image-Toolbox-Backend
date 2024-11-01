import os
from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image
import pytesseract
from app.utils.image_utils import adjust_brightness, process_frame, load_graph_opt_model, get_base_path

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

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
            is_aspect_ratio_locked = request.args.get('isAspectRatioLocked', default='false', type=str).lower() == 'true'
            
            # Read the original image
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Get original image dimensions
            orig_height, orig_width = img.shape[:2]
            original_aspect_ratio = orig_width / orig_height
            
            if is_aspect_ratio_locked:
                # Calculate resize keeping aspect ratio
                if width / height > original_aspect_ratio:
                    # Width is proportionally larger
                    new_width = int(height * original_aspect_ratio)
                    new_height = height
                else:
                    # Height is proportionally larger
                    new_width = width
                    new_height = int(width / original_aspect_ratio)
                
                # Resize the image
                resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Create a blank canvas of the original requested size
                canvas = np.zeros((height, width, 3), dtype=np.uint8) + 255  # White background
                
                # Calculate position to center the resized image
                start_x = (width - new_width) // 2
                start_y = (height - new_height) // 2
                
                # Place the resized image on the canvas
                canvas[start_y:start_y+new_height, start_x:start_x+new_width] = resized_img
                
                resized_img = canvas
            else:
                # Standard resize without maintaining aspect ratio
                resized_img = cv2.resize(img, (width, height))
            
            # Encode the image
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
            elif filter_type == 'sepia':
                # Create sepia tone filter
                kernel = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
                filtered_img = cv2.transform(img, kernel)
                filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
            elif filter_type == 'edge_enhance':
                # Edge enhancement kernel
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                filtered_img = cv2.filter2D(img, -1, kernel)
            elif filter_type == 'emboss':
                # Emboss kernel
                kernel = np.array([[-2, -1, 0],
                                   [-1,  1, 1],
                                   [ 0,  1, 2]])
                filtered_img = cv2.filter2D(img, -1, kernel)
                # Add 128 to center the pixel values around gray
                filtered_img = cv2.convertScaleAbs(filtered_img, alpha=1, beta=128)
            else:
                return jsonify({
                    "error": "Invalid filter type. Available options: blur, sharpen, grayscale, sepia, edge_enhance, emboss"
                }), 400
            
            # Ensure the image is in the correct format for encoding
            if len(filtered_img.shape) == 2:  # Grayscale
                filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
            
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
            graph_opt_path = os.path.join(get_base_path(), 'open_pose.pb')
            net = load_graph_opt_model(graph_opt_path)
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            showSekeleton = request.args.get('showSekeleton', type=bool)
            showJointConfidence = request.args.get('showJointConfidence', type=bool)
            confidence_threshold = request.args.get('confidence_threshold', type=float)

            BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

            POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
            
            frame = process_frame(
                net=net,
                frame=frame,
                inWidth=368,
                inHeight=368,
                BODY_PARTS=BODY_PARTS,
                POSE_PAIRS=POSE_PAIRS,
                confidenceThreshold=confidence_threshold,
                showSkeleton=showSekeleton,
                showJointConfidence=showJointConfidence
            )
            _, img_encoded = cv2.imencode('.jpg', frame)
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400