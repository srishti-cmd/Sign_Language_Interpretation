import cv2
import numpy as np
import pickle

def build_squares(img):
    # These coordinates are corrected to fit within a 640x480 window
    x, y, w, h = 180, 100, 20, 20
    d = 10 # Distance between squares

    imgCrop = None
    crop = None
    
    # A 10x10 grid is a good number of sample points
    for i in range(10):
        for j in range(10):
            if np.any(imgCrop == None):
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x += w + d
            
        if np.any(crop == None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop)) 
            
        imgCrop = None
        x = 180  # Reset x to the initial start position for the next row
        y += h + d
        
    return crop

# --- Start of Replacement Code ---

def get_hand_hist():
    print("Starting histogram capture process...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("FATAL ERROR: Cannot open any camera. Exiting.")
        return

    hist = None
    flagPressedS = False

    print("Entering main loop. Press 'c' to capture, 's' to save and exit.")
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("ERROR: Failed to grab frame. Exiting loop.")
            break
        
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        
        # We only need to build the squares if we haven't saved yet
        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.imshow("Set hand histogram", img)
        
        # waitKey(1) is crucial. It waits 1ms for a key press.
        keypress = cv2.waitKey(1) & 0xFF

        if keypress != 255: # 255 is the value for 'no key pressed'
            # Try to decode the key for printing, default to its number
            key_char = chr(keypress) if 32 <= keypress <= 126 else str(keypress)
            print(f"--- Key Detected! ASCII value: {keypress}, Character: '{key_char}' ---")

        if keypress == ord('c'):
            print("'c' key pressed. Attempting to capture histogram...")
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                print(">>> SUCCESS: Histogram captured successfully in memory!")
            else:
                print(">>> ERROR: Could not capture histogram because imgCrop was empty. Is your hand in the squares?")

        elif keypress == ord('s'):
            print("'s' key pressed. Exiting loop to save...")
            flagPressedS = True 
            break
            
        elif keypress == 27: # 27 is the ASCII for the Escape key
            print("'Esc' key pressed. Exiting without saving...")
            hist = None # Ensure we don't save if user presses Esc
            break

    # --- End of Loop ---
    print("Loop exited.")
    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        print("Saving histogram to file...")
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
        print(">>> FINAL RESULT: Histogram saved successfully.")
    else:
        print(">>> FINAL RESULT: No histogram was captured to save.")


# Make sure to call the function
get_hand_hist()

# --- End of Replacement Code ---