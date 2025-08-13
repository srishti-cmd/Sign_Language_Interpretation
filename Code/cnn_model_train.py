# --- Start of New, Corrected cnn_model_train.py ---

import numpy as np
import pickle
import cv2
import os
from glob import glob
from tensorflow import keras # Modern way to import Keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical # Use to_categorical from keras.utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Set TensorFlow log level to suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    """
    Dynamically finds the first image in the gestures folder to get its dimensions.
    """
    # Use glob to find any image file
    image_files = glob("gestures/*/*.jpg")
    if not image_files:
        print("Error: No images found in the 'gestures' directory. Cannot determine image size.")
        return 0, 0
    # Read the first image found
    img = cv2.imread(image_files[0], 0)
    if img is None:
        print(f"Error: Could not read the image at {image_files[0]}")
        return 0, 0
    return img.shape

def get_num_of_classes():
    """
    Dynamically counts the number of gesture subdirectories.
    """
    return len(os.listdir('gestures'))

# Set image dimensions and number of classes globally
image_x, image_y = get_image_size()
num_of_classes = get_num_of_classes()

# --- Replace your old cnn_model() function with this one ---

def cnn_model():
    """
    Defines the CNN model architecture using modern Keras standards.
    """
    # --- FIXED: Use the modern keras.Input layer to define the input shape ---
    model = Sequential([
        keras.Input(shape=(image_x, image_y, 1)),
        Conv2D(16, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_of_classes, activation='softmax')
    ])

    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # --- FIXED: Change the filename to end with .keras ---
    filepath = "cnn_model.keras"
    
    # Use 'val_accuracy' which is the modern standard for the metric name
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

# --- Replace your old train() function with this one ---

def train():
    """
    Loads the prepared data and trains the CNN model.
    """
    # Load data from the correct .pkl file names
    with open("train_images.pkl", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels.pkl", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images.pkl", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels.pkl", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    # Reshape data for the CNN
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    # --- FIXED LINES ---
    # One-hot encode the labels using the correct variable name 'num_of_classes'
    train_labels = to_categorical(train_labels, num_of_classes)
    val_labels = to_categorical(val_labels, num_of_classes)
    # --- END OF FIX ---

    print("Starting model training...")
    
    model, callbacks_list = cnn_model()
    model.summary() # Print a summary of the model architecture
    
    # Start the training process
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
    
    # Evaluate the final model on the validation data
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\nCNN Final Validation Error: {100 - scores[1] * 100:.2f}%")

# --- Main execution ---
if __name__ == '__main__':
    if image_x > 0: # Only train if image size was successfully determined
        train()
    K.clear_session() # Clear TensorFlow session memory

# --- End of New, Corrected cnn_model_train.py ---