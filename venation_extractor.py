import cv2
import numpy as np
import os
import glob
from rembg import remove

# =====================================================================
# Advanced Leaf Venation Extraction with Noise Filtering & Gap Bridging
# Requires: pip install opencv-python numpy rembg
# =====================================================================

def extract_veins(image_path, output_path):
    """
    Processes a leaf image, removes background, extracts venation, 
    filters out "salt and pepper" noise, and bridges gap in the veins.
    """
    print(f"Processing: {os.path.basename(image_path)}")
    
    # 1. Read the image and remove background
    with open(image_path, 'rb') as i:
        input_data = i.read()
        
    try:
        # Remove background (returns image with alpha channel)
        bg_removed_data = remove(input_data)
        nparr = np.frombuffer(bg_removed_data, np.uint8)
        img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"  -> Error removing background: {e}")
        return

    if img_rgba is None:
        print(f"  -> Error: Could not decode image.")
        return

    # Extract BGR channels and the Alpha channel (the leaf mask)
    img_bgr = img_rgba[:, :, :3]
    alpha_mask = img_rgba[:, :, 3]

    # 2. Grayscale & Contrast Enhancement
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(gray)

    # 3. MODERATE BLURRING (Smooth noise but preserve vein detail)
    blurred = cv2.GaussianBlur(enhanced_contrast, (7, 7), 0)

    # 4. Adaptive Thresholding
    # Smaller block size (15) and lower constant (3) to capture finer veins
    binary_veins = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 3  
    )

    # 5. MASKING: Apply background mask to remove noise outside the leaf
    # Gentle erosion to preserve thin leaf lobes and edges
    mask_kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(alpha_mask, mask_kernel, iterations=1)
    masked_veins = cv2.bitwise_and(binary_veins, binary_veins, mask=eroded_mask)

    # 6. CONTOUR FILTERING (The ultimate fix for the white specks)
    # Find all continuous white shapes in the image
    contours, _ = cv2.findContours(masked_veins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a pure black canvas to draw the clean veins on
    final_veins = np.zeros_like(masked_veins)
    
    # Only keep shapes (contours) that are larger than a certain area.
    # Veins will have a large area; the "salt and pepper" noise will have a tiny area.
    MIN_AREA = 15 
    
    for contour in contours:
        if cv2.contourArea(contour) > MIN_AREA:
            # If it's big enough, draw it as a filled white shape on our black canvas
            cv2.drawContours(final_veins, [contour], -1, 255, thickness=cv2.FILLED)

    # 7. BRIDGING GAPS (Morphological Closing)
    # Re-connects broken vein segments that were separated during thresholding
    # A 5x5 kernel will bridge gaps up to ~5 pixels wide
    close_kernel = np.ones((5, 5), np.uint8)
    connected_veins = cv2.morphologyEx(final_veins, cv2.MORPH_CLOSE, close_kernel)

    # 8. Save the final cleaned venation network
    cv2.imwrite(output_path, connected_veins)
    print(f"  -> Saved clean venation map!")


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_DIR = "./sorted_leaves"
    OUTPUT_DIR = "./venation_maps"
    CLASSES = ["Parallel", "Pinnate", "Palmate"]

    # Check that the sorted_leaves folder exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: '{INPUT_DIR}' not found. Please sort your leaves first.")
        exit()

    total_processed = 0

    for class_name in CLASSES:
        class_input = os.path.join(INPUT_DIR, class_name)
        class_output = os.path.join(OUTPUT_DIR, class_name)

        if not os.path.exists(class_input):
            print(f"Warning: '{class_input}' not found, skipping.")
            continue

        os.makedirs(class_output, exist_ok=True)

        # Find all JPEG/PNG files in this class folder
        image_files = glob.glob(os.path.join(class_input, "*.[jJ][pP]*[gG]"))
        image_files.extend(glob.glob(os.path.join(class_input, "*.[pP][nN][gG]")))

        if not image_files:
            print(f"No images found in '{class_input}', skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {len(image_files)} images from '{class_name}'...")
        print(f"{'='*60}")

        for img_path in image_files:
            filename = os.path.basename(img_path)
            output_name = f"veins_{filename}"
            out_path = os.path.join(class_output, output_name)

            extract_veins(img_path, out_path)
            total_processed += 1

    print(f"\nDone! Processed {total_processed} images across {len(CLASSES)} classes.")
    print(f"Venation maps saved to '{OUTPUT_DIR}/'.")