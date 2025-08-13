# --- Start of New, Corrected load_images.py ---

import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import pickle
import os

def load_and_prepare_data():
    """
    Loads images from named folders, creates numerical labels,
    and saves the prepared training, validation, and test sets.
    """
    print("--- Loading images and creating labels ---")
    gest_folder = "gestures"
    images_labels = []

    # --- NEW: Automatically create a mapping from folder names to integer labels ---
    try:
        # Get a sorted list of gesture names (folder names)
        gesture_names = sorted([name for name in os.listdir(gest_folder) if os.path.isdir(os.path.join(gest_folder, name))])
    except FileNotFoundError:
        print(f"Error: The '{gest_folder}' directory was not found. Please create your dataset first.")
        return None

    # Create the map: {'A': 0, 'B': 1, 'hello': 2, ...}
    label_map = {name: i for i, name in enumerate(gesture_names)}
    
    print(f"Found {len(gesture_names)} gestures. Created the following mapping:")
    print(label_map)

    # Save this crucial map to a file for later use in the prediction script
    with open("gesture_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
    print("Gesture map saved to 'gesture_map.pkl'")
    # --- END NEW ---

    # Loop through all images using glob
    image_paths = glob(os.path.join(gest_folder, "*", "*.jpg"))

    for image_path in image_paths:
        try:
            # Extract the folder name (e.g., 'A') from the path
            folder_name = os.path.basename(os.path.dirname(image_path))
            
            # Look up the integer label from our map
            label = label_map[folder_name]
            
            # Read the image in grayscale
            img = cv2.imread(image_path, 0)

            if img is not None:
                images_labels.append((np.array(img, dtype=np.uint8), label))
            else:
                print(f"Warning: Could not read image {image_path}. Skipping.")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    return images_labels

# --- Main script execution ---
if __name__ == '__main__':
    # Load the data using our new function
    images_labels = load_and_prepare_data()

    if images_labels:
        # This part remains the same, but it's now working with the correctly labeled data
        images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
        images, labels = zip(*images_labels)
        
        print("\n--- Splitting and Saving Datasets ---")
        print(f"Total images loaded: {len(images_labels)}")

        # Splitting logic: 83.3% train, 8.3% test, 8.3% validation
        train_split_idx = int(5/6 * len(images))
        test_split_idx = int(11/12 * len(images))

        # Training set
        train_images = images[:train_split_idx]
        train_labels = labels[:train_split_idx]
        print(f"Training set size: {len(train_images)} images")
        with open("train_images.pkl", "wb") as f: pickle.dump(train_images, f)
        with open("train_labels.pkl", "wb") as f: pickle.dump(train_labels, f)

        # Testing set
        test_images = images[train_split_idx:test_split_idx]
        test_labels = labels[train_split_idx:test_split_idx]
        print(f"Testing set size: {len(test_images)} images")
        with open("test_images.pkl", "wb") as f: pickle.dump(test_images, f)
        with open("test_labels.pkl", "wb") as f: pickle.dump(test_labels, f)

        # Validation set
        val_images = images[test_split_idx:]
        val_labels = labels[test_split_idx:]
        print(f"Validation set size: {len(val_images)} images")
        with open("val_images.pkl", "wb") as f: pickle.dump(val_images, f)
        with open("val_labels.pkl", "wb") as f: pickle.dump(val_labels, f)
        
        print("\n--- All datasets saved successfully! ---")

# --- End of New, Corrected load_images.py ---