import sys
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from venation_extractor import extract_veins

# =====================================================================
# Leaf Venation Classifier - CNN Single Image Prediction
# Pipeline: Raw image -> Vein extraction -> CNN -> Classification
# Usage: python classifier.py <image_path>
# Requires: pip install tensorflow opencv-python rembg
# =====================================================================

MODEL_PATH = "leaf_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (128, 128)
TEMP_VEIN = "./temp_classify_vein.png"

if len(sys.argv) < 2:
    print("Usage: python classifier.py <image_path>")
    print("Example: python classifier.py my_leaf.jpg")
    exit()

# Join all args to support paths with spaces
image_path = " ".join(sys.argv[1:])

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model '{MODEL_PATH}' not found. Run train_model.py first.")
    exit()

# Step 1: Extract venation map
print(f"Classifying: {os.path.basename(image_path)}")
extract_veins(image_path, TEMP_VEIN)

# Step 2: Load vein image
vein_img = cv2.imread(TEMP_VEIN, cv2.IMREAD_GRAYSCALE)
if vein_img is None:
    print("Error: Venation extraction failed.")
    exit()

vein_img = cv2.resize(vein_img, IMAGE_SIZE)

# Reshape for CNN: (1, 128, 128, 1)
img_array = np.expand_dims(vein_img, axis=(0, -1)).astype(np.float32)

# Step 3: Load model and predict
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

predictions = model.predict(img_array, verbose=0)[0]
predicted_idx = np.argmax(predictions)
prediction = class_names[predicted_idx]

# Clean up temp file
if os.path.exists(TEMP_VEIN):
    os.remove(TEMP_VEIN)

print(f"\n{'=' * 45}")
print(f"  Result:  {prediction.upper()} venation")
print(f"{'=' * 45}")
print(f"  Confidence:")
for i, cls in enumerate(class_names):
    prob = predictions[i]
    bar = "█" * int(prob * 20)
    print(f"    {cls:>10}: {prob*100:5.1f}%  {bar}")
print(f"{'=' * 45}")