import os
import json
import numpy as np
import cv2
import shutil
import tensorflow as tf
from venation_extractor import extract_veins

# =====================================================================
# Auto-Sort: Classifies unsorted leaf images using the CNN model.
# Pipeline: Raw image -> Vein extraction -> CNN -> Classify
# Requires: pip install tensorflow opencv-python rembg
# =====================================================================

MODEL_PATH = "leaf_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
INPUT_FOLDER = "mixed_images"
OUTPUT_FOLDER = "sorted"
TEMP_VEIN_DIR = "./temp_veins"
IMAGE_SIZE = (128, 128)

# Load model
print("Loading CNN model...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model '{MODEL_PATH}' not found. Run train_model.py first.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Classes: {class_names}")

# Create output folders
print("Creating output folders...")
for c in class_names:
    os.makedirs(os.path.join(OUTPUT_FOLDER, c.lower()), exist_ok=True)
os.makedirs(TEMP_VEIN_DIR, exist_ok=True)

if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)
    print(f"Created '{INPUT_FOLDER}'. Place your unsorted leaf images inside and run again.")
    exit()

image_files = [f for f in os.listdir(INPUT_FOLDER)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"No images found in '{INPUT_FOLDER}'. Add leaf images and try again.")
    exit()

print(f"Processing {len(image_files)} images...\n")

for file in image_files:
    img_path = os.path.join(INPUT_FOLDER, file)
    vein_path = os.path.join(TEMP_VEIN_DIR, f"veins_{file}")

    extract_veins(img_path, vein_path)

    vein_img = cv2.imread(vein_path, cv2.IMREAD_GRAYSCALE)
    if vein_img is None:
        print(f"  -> Skipping {file} (vein extraction failed)")
        continue

    vein_img = cv2.resize(vein_img, IMAGE_SIZE)
    img_array = np.expand_dims(vein_img, axis=(0, -1)).astype(np.float32)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    prediction = class_names[predicted_idx].lower()

    destination = os.path.join(OUTPUT_FOLDER, prediction, file)
    shutil.copy2(img_path, destination)

    print(f"  {file} -> {prediction} ({predictions[predicted_idx]*100:.1f}%)")

shutil.rmtree(TEMP_VEIN_DIR, ignore_errors=True)

print(f"\nDone! {len(image_files)} images sorted into '{OUTPUT_FOLDER}/'.")