from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
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

# detect objects in provided image
def detect_objects(image):
    # Charger le modèle YOLOv3 pré-entraîné
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    
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



# if __name__ == '__main__':
#     app.run(debug=True)
