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
def process_frame(net, frame, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, confidenceThreshold=0.1, showSkeleton=True, showJointConfidence=False):
    """
    Process frame for pose detection with configurable visualization options.
    
    Args:
        net: Neural network model
        frame: Input image frame
        inWidth: Input width for the network
        inHeight: Input height for the network
        BODY_PARTS: Dictionary of body parts and their corresponding indices
        POSE_PAIRS: List of pairs defining connections between body parts
        confidenceThreshold: Minimum confidence threshold for detecting joints (default: 0.1)
        showSkeleton: Whether to draw the skeleton lines between joints (default: True)
        showJointConfidence: Whether to display confidence scores near joints (default: False)
    
    Returns:
        Processed frame with visualizations
    """
    frameHeight, frameWidth = frame.shape[:2]
    
    # Prepare input blob and run inference
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), 
                                      (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    # Dictionary to store points and their confidence values
    points = []
    confidences = []
    
    # Detect keypoints
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        
        if conf > confidenceThreshold:
            points.append((int(x), int(y)))
            confidences.append(conf)
        else:
            points.append(None)
            confidences.append(0)
    
    # Draw skeleton and joints
    if showSkeleton:
        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                # Draw skeleton lines
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                
                # Draw joints
                cv2.circle(frame, points[idFrom], 4, (0, 0, 255), thickness=cv2.FILLED)
                cv2.circle(frame, points[idTo], 4, (0, 0, 255), thickness=cv2.FILLED)
                
                # Show confidence values if enabled
                if showJointConfidence:
                    # Display confidence for start point
                    conf_text = f"{confidences[idFrom]:.2f}"
                    cv2.putText(frame, conf_text, 
                              (points[idFrom][0] + 10, points[idFrom][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, conf_text, 
                              (points[idFrom][0] + 10, points[idFrom][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display confidence for end point
                    conf_text = f"{confidences[idTo]:.2f}"
                    cv2.putText(frame, conf_text, 
                              (points[idTo][0] + 10, points[idTo][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, conf_text, 
                              (points[idTo][0] + 10, points[idTo][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
