from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import urllib.request
import numpy as np
import io
import os
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# https://pypi.org/project/pytesseract/
# https://tesseract-ocr.github.io/tessdoc/Downloads.html
# https://sourceforge.net/projects/tesseract-ocr-alt/files/

# pip install flask-cors

# generate requirements.txt
# pip freeze > requirements.txt

# pip install waitress


app = Flask(__name__)
CORS(app)


def download_file(url):
    with urllib.request.urlopen(url) as response:
        return response.read()

# detect objects in provided image
def detect_objects(image):
    # Charger le modèle YOLOv3 pré-entraîné
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()

    # Load YOLOv3 weights and configuration from URL
    # weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    # cfg_url = "https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg"
    
    # # Download weights and configuration files
    # weights_data = download_file(weights_url)
    # cfg_data = download_file(cfg_url)
    
    # # Load weights and configuration from memory
    # net = cv2.dnn.readNetFromDarknet(io.BytesIO(weights_data), io.BytesIO(cfg_data))
    # layer_names = net.getLayerNames()
    
    # Obtenir les noms des couches de sortie
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Charger des classes
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]

    # Charger l'image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width, channels = img.shape

    # Prétraitement des images
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informations de détection
    class_ids = []
    confidences = []
    boxes = []

    # Analyser les sorties du réseau
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Coordonnées du cadre de délimitation
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordonnées du cadre de délimitation (coin supérieur gauche)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Supprimer les détections redondantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dessiner les résultats sur l'image
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convertir l'image OpenCV en bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    
    return img_encoded.tobytes()

@app.route('/detect_objects', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            processed_image = detect_objects(file)
            return send_file(io.BytesIO(processed_image), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400


# resize the provided image
@app.route('/resize_image', methods=['POST'])
def resize_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Parse dimensions from request (default to 300x300 if not provided)
            width = request.args.get('width', default=300, type=int)
            height = request.args.get('height', default=300, type=int)

            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Resize image
            resized_img = cv2.resize(img, (width, height))
            
            # Convert the resized image to bytes
            _, img_encoded = cv2.imencode('.jpg', resized_img)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400


# apply filter to provided image
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Parse filter type from request
            filter_type = request.args.get('filter', default='blur', type=str)

            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Apply filter based on filter type
            if filter_type == 'blur':
                filtered_img = cv2.blur(img, (5, 5))  # Example: applying blur filter
            elif filter_type == 'sharpen':
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                filtered_img = cv2.filter2D(img, -1, kernel)  # Example: applying sharpen filter
            elif filter_type == 'grayscale':
                filtered_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Example: converting to grayscale
            else:
                return jsonify({"error": "Invalid filter type. Available options: blur, sharpen, grayscale"}), 400
            
            # Convert the filtered image to bytes
            _, img_encoded = cv2.imencode('.jpg', filtered_img)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400
    

# extract text from provided image
@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Read image using PIL (Python Imaging Library)
            image = Image.open(io.BytesIO(file.read()))
            
            # Perform OCR using pytesseract
            extracted_text = pytesseract.image_to_string(image)
            
            return jsonify({"text": extracted_text}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400



def adjust_brightness(image, factor):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Adjust brightness by multiplying value channel by the brightness factor
    v = cv2.multiply(v, factor)
    
    # Clip the values to ensure they remain within the valid range
    v = np.clip(v, 0, 255)
    
    # Merge the channels back together and convert back to BGR color space
    enhanced_image = cv2.merge((h, s, v))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_HSV2BGR)
    
    return enhanced_image

@app.route('/enhance_image', methods=['POST'])
def enhance_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Parse enhancement parameters from request
            brightness_factor = request.args.get('brightness', default=1.0, type=float)
            contrast_factor = request.args.get('contrast', default=1.0, type=float)
            
            # Apply brightness adjustment
            enhanced_image = adjust_brightness(img, brightness_factor)
            
            # Apply contrast adjustment
            enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast_factor, beta=0)
            
            # Convert the enhanced image to bytes
            _, img_encoded = cv2.imencode('.jpg', enhanced_image)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400




@app.route('/compress_image', methods=['POST'])
def compress_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Parse compression quality from request (default to 80 if not provided)
            quality = request.args.get('quality', default=80, type=int)
            
            # Convert image to JPEG format with specified quality
            _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400




def detect_objects2(image):
    # Load the pre-trained YOLOv3 model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    
    # Get the names of the output layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load classes
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]

    # Load the image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width, channels = img.shape

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detection information
    object_counts = {}

    # Analyze the network's outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                object_counts[label] = object_counts.get(label, 0) + 1

    return object_counts

@app.route('/count_objects', methods=['POST'])
def count_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            object_counts = detect_objects2(file)
            return jsonify(object_counts), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400



def compare_images(image1, image2):
    # Load the images
    img1 = cv2.imdecode(np.fromstring(image1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.fromstring(image2.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSIM) between the two images
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    ssim_score = np.max(result)

    return float(ssim_score)

@app.route('/compare_images', methods=['POST'])
def compare_images_endpoint():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both files are required"}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "Both files must be selected"}), 400
    
    if file1 and file1.filename.endswith(('.jpg', '.jpeg', '.png')) and \
       file2 and file2.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            similarity_score = compare_images(file1, file2)
            return jsonify({"similarity_score": similarity_score}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400



def load_graph_opt_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    net = cv2.dnn.readNetFromTensorflow(model_path)
    return net

def process_frame(net, frame, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, thr):
    frameHeight, frameWidth = frame.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return frame

@app.route('/detect_open_poses', methods=['POST'])
def detect_open_poses():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            net = load_graph_opt_model('graph_opt.pb')
            frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

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





# if __name__ == '__main__':
#     app.run(debug=True)
