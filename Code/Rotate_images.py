# --- Start of New, Corrected Rotate_images.py ---

import cv2
import os

def flip_images():
    """
    Finds all images in the 'gestures' directory, flips them horizontally,
    and saves them with a 'flipped_' prefix.
    This version is robust and handles any number of images per folder.
    """
    gest_folder = "gestures"
    print("--- Starting Image Augmentation (Flipping) ---")

    try:
        # Get the list of all gesture folders (e.g., '0', 'A', 'hello')
        gesture_folders = os.listdir(gest_folder)
    except FileNotFoundError:
        print(f"Error: The '{gest_folder}' directory was not found. Please run create_gestures.py first.")
        return

    for g_folder_name in gesture_folders:
        folder_path = os.path.join(gest_folder, g_folder_name)

        # Make sure we're only processing directories, not other files
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: '{g_folder_name}'")
        
        try:
            # Get a list of all items in the current gesture folder
            images_in_folder = os.listdir(folder_path)
        except FileNotFoundError:
            # This case is unlikely if the parent loop worked, but it's safe to have
            print(f"  - Warning: Could not find folder {folder_path}. Skipping.")
            continue

        for image_name in images_in_folder:
            # IMPORTANT: We only want to flip original images, not ones we have already flipped.
            if image_name.startswith('flipped_'):
                continue
            
            # Make sure we are only processing image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Construct the full path to the image to be read
            original_image_path = os.path.join(folder_path, image_name)
            
            # Read the image in grayscale (mode 0) as in the original script
            img = cv2.imread(original_image_path, 0)

            # CRUCIAL: Check if the image was read successfully before processing
            if img is not None:
                # Flip the image horizontally (1 = horizontal, 0 = vertical)
                flipped_img = cv2.flip(img, 1)

                # Create a new name for the flipped image
                # e.g., '1.jpg' becomes 'flipped_1.jpg'
                new_image_name = f"flipped_{image_name}"
                new_image_path = os.path.join(folder_path, new_image_name)

                # Save the new flipped image
                cv2.imwrite(new_image_path, flipped_img)
            else:
                print(f"  - Warning: Could not read image {original_image_path}. Skipping.")

    print("\n--- Image flipping complete! ---")

# --- Main execution ---
if __name__ == '__main__':
    flip_images()

# --- End of New, Corrected Rotate_images.py ---