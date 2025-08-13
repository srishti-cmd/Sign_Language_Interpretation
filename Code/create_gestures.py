# --- Start of New create_gestures.py ---

import cv2
import numpy as np
import pickle
import os
import sqlite3

# Global settings    
image_x, image_y = 50, 50

def get_hand_hist():
    """Loads the saved hand histogram data."""
    if not os.path.exists("hist"):
        print("Error: hist file not found. Please run set_hand_histogram.py first.")
        return None
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    """Initializes the gestures folder and the SQLite database if they don't exist."""
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
        conn.close()

def create_folder(folder_name):
    """Creates a folder for a new gesture."""
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    """Stores the gesture ID and name into the database."""
    conn = sqlite3.connect("gesture_db.db")
    # Use parameterized queries to prevent SQL injection
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)"
    try:
        conn.execute(cmd, (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ").lower()
        if choice == 'y':
            cmd = "UPDATE gesture SET g_name = ? WHERE g_id = ?"
            conn.execute(cmd, (g_name, g_id))
        else:
            print("Doing nothing...")
            return
    conn.commit()
    conn.close()

# --- Replace your old main() function with this one ---

# --- Start of Replacement Code ---

# def main():
#     """Main function with heavy debugging to trace execution."""
#     print("--- 1. Script Main Function Started ---")
    
#     # Initialize database and folders
#     init_create_folder_database()
#     print("--- 2. Database and Folders Initialized ---")
    
#     # Load the hand histogram
#     print("--- 3. Attempting to load 'hist' file... ---")
#     hist = get_hand_hist()
#     if hist is None:
#         # This part is not the problem, but we'll leave the check
#         print("--- EXIT: 'hist' file not found. ---")
#         return
#     print("--- 4. SUCCESS: 'hist' file loaded successfully. ---")

#     # Open the camera
#     print("--- 5. Attempting to open camera at index 0... ---")
#     cam = cv2.VideoCapture(0)
#     if not cam.isOpened():
#         print("--- INFO: Camera 0 failed. Trying camera at index 1... ---")
#         cam = cv2.VideoCapture(1)
    
#     if not cam.isOpened():
#         print("--- FATAL EXIT: Could not open any camera. ---")
#         return
#     print("--- 6. SUCCESS: Camera opened successfully. ---")
    
#     # ROI for hand detection
#     x, y, w, h = 300, 100, 300, 300
    
#     # Frame counter to slow down capture
#     frame_counter = 0
#     capture_speed = 5
    
#     # Main loop variables
#     is_capturing = False
#     pic_no = 0
#     current_g_id = None
    
#     print("--- 7. Entering main camera loop... (A window should appear now) ---")
#     print("Press 'n' for new gesture, 'c' to start/stop capture, 'ESC' to exit.")

#     while True:
#         ret, img = cam.read()
#         if not ret:
#             print("--- ERROR in loop: Failed to grab frame. ---")
#             break
            
#         frame_counter += 1
#         img = cv2.flip(img, 1)
#         # ... (rest of the image processing logic is likely fine) ...
#         # The code below will only run if the loop is successfully entered
#         imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
#         disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#         cv2.filter2D(dst, -1, disc, dst)
#         blur = cv2.GaussianBlur(dst, (11, 11), 0)
#         blur = cv2.medianBlur(blur, 15)
#         thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         if current_g_id is not None:
#              cv2.putText(img, f"Gesture: {current_g_id} | Count: {pic_no}", (30, 50), font, 1, (255, 255, 255), 2)
#         else:
#             cv2.putText(img, "Press 'n' for a new gesture", (30, 50), font, 1, (255, 255, 255), 2)
#         if is_capturing:
#             cv2.putText(img, "CAPTURING", (30, 80), font, 1, (0, 0, 255), 2)

#         hand_thresh = thresh[y:y+h, x:x+w]
#         contours, _ = cv2.findContours(hand_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#         if len(contours) > 0 and is_capturing and (frame_counter % capture_speed == 0):
#             contour = max(contours, key=cv2.contourArea)
#             if cv2.contourArea(contour) > 1000:
#                 x1, y1, w1, h1 = cv2.boundingRect(contour)
#                 pic_no += 1
#                 save_img = hand_thresh[y1:y1+h1, x1:x1+w1]
#                 if w1 > h1:
#                     save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
#                 elif h1 > w1:
#                     save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0,0,0))
#                 save_img = cv2.resize(save_img, (image_x, image_y))
#                 cv2.imwrite(f"gestures/{current_g_id}/{pic_no}.jpg", save_img)

#         cv2.imshow("Capture Gestures", img)
#         cv2.imshow("Hand Mask", thresh)

#         keypress = cv2.waitKey(1) & 0xFF
#         if keypress == 27: break
#         elif keypress == ord('n'):
#             g_id = input("Enter gesture no. (ID): ")
#             g_name = input("Enter gesture name/text: ")
#             current_g_id = int(g_id)
#             store_in_db(current_g_id, g_name)
#             create_folder(f"gestures/{current_g_id}")
#             is_capturing = False
#             pic_no = 0
#         elif keypress == ord('c'):
#             if current_g_id is not None:
#                 is_capturing = not is_capturing
#                 if not is_capturing:
#                     print(f"Capture stopped. Total images for gesture {current_g_id}: {pic_no}")
#             else:
#                 print("Please create a new gesture first by pressing 'n'.")

#     # Cleanup
#     print("--- 8. Cleaning up and closing... ---")
#     cam.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()

# --- Replace your old main() function with this one ---

def main():
    """Main function to run the gesture capture process."""
    init_create_folder_database()
    
    hist = get_hand_hist()
    if hist is None:
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # ROI for hand detection
    x, y, w, h = 300, 100, 300, 300
    
    # Frame counter to slow down capture
    frame_counter = 0
    capture_speed = 5
    
    # Main loop variables
    is_capturing = False
    pic_no = 0
    
    # --- MODIFICATION START ---
    # We will now track the gesture name for folder creation
    current_g_name = None
    # --- MODIFICATION END ---
    
    print("Camera started. Press 'n' for new gesture, 'c' to start/stop capture, 'ESC' to exit.")

    while True:
        ret, img = cam.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break
            
        frame_counter += 1
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- MODIFICATION START ---
        # Display the gesture NAME on screen instead of the ID
        if current_g_name is not None:
             cv2.putText(img, f"Gesture: {current_g_name} | Count: {pic_no}", (30, 50), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(img, "Press 'n' for a new gesture", (30, 50), font, 1, (255, 255, 255), 2)
        # --- MODIFICATION END ---
            
        if is_capturing:
            cv2.putText(img, "CAPTURING", (30, 80), font, 1, (0, 0, 255), 2)

        hand_thresh = thresh[y:y+h, x:x+w]
        contours, _ = cv2.findContours(hand_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0 and is_capturing and (frame_counter % capture_speed == 0):
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 1000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = hand_thresh[y1:y1+h1, x1:x1+w1]
                
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0,0,0))
                
                save_img = cv2.resize(save_img, (image_x, image_y))
                
                # --- MODIFICATION START ---
                # Save the image using the gesture NAME for the folder
                cv2.imwrite(f"gestures/{current_g_name}/{pic_no}.jpg", save_img)
                # --- MODIFICATION END ---

        cv2.imshow("Capture Gestures", img)
        cv2.imshow("Hand Mask", thresh)

        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == 27: # ESC key
            break
            
        elif keypress == ord('n'): # 'n' for new gesture
            g_id = input("Enter gesture no. (ID): ")
            g_name = input("Enter gesture name/text: ")
            
            # --- MODIFICATION START ---
            # Store the current gesture name for use in folder paths
            current_g_name = g_name
            # --- MODIFICATION END ---
            
            store_in_db(int(g_id), g_name)
            
            # --- MODIFICATION START ---
            # Create the folder using the gesture NAME
            create_folder(f"gestures/{current_g_name}")
            # --- MODIFICATION END ---
            
            is_capturing = False
            pic_no = 0
            
        elif keypress == ord('c'): # 'c' to toggle capture
            if current_g_name is not None:
                is_capturing = not is_capturing
                if not is_capturing:
                    print(f"Capture stopped. Total images for gesture '{current_g_name}': {pic_no}")
            else:
                print("Please create a new gesture first by pressing 'n'.")

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()