# --- Start of New, Corrected final.py ---

import cv2
import pickle
import numpy as np
import os
from glob import glob
from tensorflow.keras.models import load_model # Use modern import

# --- FIXED: Load our new .keras model ---
model = load_model('cnn_model.keras')

# --- NEW: Load the gesture map we created ---
with open("gesture_map.pkl", "rb") as f:
    gesture_map = pickle.load(f)
# Create a reverse map for easy lookup: {0: 'A', 1: 'B', ...}
reverse_gesture_map = {v: k for k, v in gesture_map.items()}

def get_hand_hist():
    """Loads the saved hand histogram data."""
    if not os.path.exists("hist"):
        print("Error: 'hist' file not found. Please run set_hand_histogram.py first.")
        exit()
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_image_size():
    """
    Dynamically finds the first image in the gestures folder to get its dimensions.
    """
    image_files = glob("gestures/*/*.jpg")
    if not image_files:
        print("Error: No images found in 'gestures' directory. Cannot determine image size.")
        return 0, 0
    img = cv2.imread(image_files[0], 0)
    if img is None:
        print(f"Error: Could not read image at {image_files[0]}")
        return 0, 0
    return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
    """Reshapes the image for the model."""
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    """Gets a prediction from the model."""
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = np.argmax(pred_probab)
    return max(pred_probab), pred_class

# --- FIXED: Replaced the database function with our new map-based function ---
def get_pred_text_from_map(pred_class):
    """
    Converts the predicted class number to its gesture name using our map.
    """
    return reverse_gesture_map.get(pred_class, "Unknown")

def recognize():
    """
    Main function to run the live sign language interpreter.
    """
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("ERROR: Cannot open camera.")
        return

    hist = get_hand_hist()
    
    # Region of Interest
    x, y, w, h = 300, 100, 300, 300
    
    # Text display variables
    text = ""
    pred_text = ""
    count_same_frame = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break
            
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        
        # Hand detection and processing
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        hand_thresh = thresh[y:y+h, x:x+w]
        contours, _ = cv2.findContours(hand_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        old_pred_text = pred_text
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                # --- Get prediction from contour ---
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = hand_thresh[y1:y1+h1, x1:x1+w1]
                
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0,0,0))
                
                pred_probab, pred_class = keras_predict(model, save_img)
                
                # Only update text if confidence is high enough
                if pred_probab * 100 > 1: # Confidence threshold
                    pred_text = get_pred_text_from_map(pred_class)
                # --- End prediction ---

        # Display UI
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Sign Language Interpreter", (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))
        cv2.putText(blackboard, "Prediction:", (30, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(blackboard, pred_text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        res = np.hstack((img, blackboard))
        cv2.imshow("Sign Language Interpreter", res)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()


# --- Main execution ---
if __name__ == '__main__':
    # A "warm-up" prediction to initialize TensorFlow
    if image_x > 0:
        keras_predict(model, np.zeros((image_x, image_y), dtype=np.uint8))
        recognize()

# --- End of New, Corrected final.py ---