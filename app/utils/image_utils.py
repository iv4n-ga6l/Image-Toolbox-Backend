import os
import cv2
import numpy as np
import urllib3

# Get the absolute path to the models folder
def get_base_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

# Download the file from url
def download_file(url):
    with urllib3.request.urlopen(url) as response:
        return response.read()

# Adjust brigthness for image enhancing
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

# Check similarity between two images
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

# Load graph opt model
def load_graph_opt_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    net = cv2.dnn.readNetFromTensorflow(model_path)
    return net

# Process frame for poses detection
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
