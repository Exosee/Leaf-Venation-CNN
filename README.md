# Leaf Venation Pattern Classification

A computer vision project that classifies leaf venation patterns into three categories -- Parallel, Pinnate, and Palmate -- using image processing and deep learning.

The system extracts venation (vein) networks from leaf images using background removal and adaptive thresholding, then classifies the extracted patterns using a Convolutional Neural Network (CNN).


## Venation Types

| Type | Description | Example |
|------|-------------|---------|
| Parallel | Veins run side by side along the length of the leaf (e.g., grass, bamboo, banana) | Long, narrow leaves with horizontal lines |
| Pinnate | Secondary veins branch off a single central midrib in a fishbone pattern (e.g., mango, rose, neem) | Oval leaves with a prominent central vein |
| Palmate | Multiple main veins radiate outward from a single point at the leaf base (e.g., maple, papaya, castor) | Wide/lobed leaves with veins spreading like fingers |


## Project Structure

```
Leaf Venation/
|
|-- venation_extractor.py    # Extracts vein networks from leaf images
|-- train_model.py           # Trains the CNN classifier on venation maps
|-- classifier.py            # Classifies a single leaf image
|-- auto_sort.py             # Batch-sorts a folder of unsorted leaf images
|-- app.py                   # Streamlit web GUI
|
|-- sorted_leaves/           # Manually sorted training images
|   |-- Parallel/            # 200 parallel-veined leaf images
|   |-- Pinnate/             # 200 pinnate-veined leaf images
|   |-- Palmate/             # 213 palmate-veined leaf images
|
|-- venation_maps/           # Extracted venation maps (generated)
|   |-- Parallel/
|   |-- Pinnate/
|   |-- Palmate/
|
|-- leaf_cnn_model.keras     # Trained CNN model (generated)
|-- class_names.json         # Class label mapping (generated)
|-- raw_whatsapp_images/     # Unsorted raw images for testing
```


## How It Works

### 1. Venation Extraction (venation_extractor.py)

The extraction pipeline processes each leaf image through the following steps:

1. Background removal using the rembg library
2. Grayscale conversion and contrast enhancement (CLAHE)
3. Gaussian blur to reduce surface texture noise
4. Adaptive thresholding to isolate vein structures
5. Mask erosion to remove leaf edge artifacts
6. Contour filtering to eliminate small noise fragments
7. Morphological closing to bridge broken vein segments

The output is a binary (black and white) image showing only the vein network.

### 2. Model Training (train_model.py)

A CNN is trained on 613 venation map images with the following architecture:

- 4 convolutional blocks (32, 64, 128, 256 filters) with batch normalization
- Global average pooling
- Dense layers (256, 128) with dropout for regularization
- Softmax output layer for 3-class classification

Training features include:
- Built-in data augmentation (flips, rotation, zoom, translation, contrast)
- Gaussian noise injection for robustness to imperfect vein extraction
- Early stopping and learning rate reduction callbacks
- 80/20 train/validation split

### 3. Classification Pipeline

When classifying a new leaf image, the system:

1. Extracts the venation map using venation_extractor.py
2. Resizes to 128x128 grayscale
3. Feeds the vein map through the trained CNN
4. Returns the predicted venation type with confidence scores


## Requirements

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- rembg (for background removal)
- TensorFlow 2.x
- scikit-learn (for evaluation metrics)
- Streamlit (for web GUI)

Install all dependencies:

```
pip install opencv-python numpy rembg tensorflow scikit-learn streamlit
```


## Usage

### Step 1: Sort Your Leaf Images

Manually sort leaf images into the following folder structure:

```
sorted_leaves/
|-- Parallel/    (place parallel-veined leaf images here)
|-- Pinnate/     (place pinnate-veined leaf images here)
|-- Palmate/     (place palmate-veined leaf images here)
```

Aim for at least 200 images per class for best results.

### Step 2: Extract Venation Maps

```
python venation_extractor.py
```

This processes all images in sorted_leaves/ and saves venation maps to venation_maps/.

### Step 3: Train the Model

```
python train_model.py
```

Training takes a few minutes. The script will display accuracy metrics and save the model as leaf_cnn_model.keras.

### Step 4: Classify Leaf Images

Classify a single image:

```
python classifier.py path/to/leaf_image.jpg
```

Output:

```
=============================================
  Result:  PALMATE venation
=============================================
  Confidence:
       Palmate:  98.4%  ===================
      Parallel:   0.2%
       Pinnate:   1.4%
=============================================
```

### Step 5: Batch Sort (Optional)

To automatically sort a folder of unsorted leaf images:

1. Place images in a mixed_images/ folder
2. Run:

```
python auto_sort.py
```

3. Results are saved in sorted/ with subfolders for each class.

### Step 6: Web GUI (Optional)

Launch the Streamlit web interface for an interactive experience:

```
python -m streamlit run app.py
```

This opens a browser-based GUI at http://localhost:8501 where you can:

- Upload a leaf image and instantly see the classification result
- View the original image alongside the extracted venation map
- See confidence scores for all three classes with visual progress bars
- Read about each venation type with common examples


## Tips for Better Accuracy

- Use clean, well-lit photos of individual leaves
- White or plain backgrounds produce the best venation maps
- Include a variety of leaf species in each class during training
- More training images (especially real-world photos) improve generalization
- Ensure correct labels when sorting training images -- mislabeled data hurts accuracy significantly
