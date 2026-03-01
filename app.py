import streamlit as st
import numpy as np
import cv2
import os
import json
import tempfile
import tensorflow as tf
from venation_extractor import extract_veins

# =====================================================================
# Leaf Venation Classifier - Streamlit GUI
# Run: streamlit run app.py
# =====================================================================

# --- Page config ---
st.set_page_config(
    page_title="Leaf Venation Classifier",
    page_icon="🍃",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E7D32;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1B5E20 0%, #388E3C 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .result-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .info-card {
        background: #1a3a1a;
        border-left: 4px solid #4CAF50;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #E8F5E9 !important;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .info-card strong {
        color: #A5D6A7 !important;
        font-size: 1.15rem;
    }
    .info-card em {
        color: #81C784 !important;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="main-title">🍃 Leaf Venation Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf image to identify its venation pattern</p>', unsafe_allow_html=True)

# --- Load model ---
MODEL_PATH = "leaf_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (128, 128)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

if model is None:
    st.error("Model not found. Please run `python train_model.py` first.")
    st.stop()

# --- Venation type info ---
VENATION_INFO = {
    "Parallel": {
        "icon": "🌿",
        "description": "Veins run side by side along the length of the leaf.",
        "examples": "Grass, Bamboo, Banana, Rice, Wheat, Corn"
    },
    "Pinnate": {
        "icon": "🌱",
        "description": "Secondary veins branch off a central midrib in a fishbone pattern.",
        "examples": "Mango, Rose, Neem, Oak, Cherry, Guava"
    },
    "Palmate": {
        "icon": "🍁",
        "description": "Multiple main veins radiate outward from a single point at the leaf base.",
        "examples": "Maple, Papaya, Castor, Grape, Cotton"
    }
}

# --- File upload ---
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_img is None:
        st.error("Could not read the image. Please try another file.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Uploaded Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Extract venation
    with st.spinner("Extracting venation pattern..."):
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_input:
            tmp_input.write(file_bytes.tobytes())
            tmp_input_path = tmp_input.name

        # Save temp input properly
        cv2.imwrite(tmp_input_path, original_img)

        tmp_output_path = tmp_input_path.replace(".jpg", "_veins.jpg")

        extract_veins(tmp_input_path, tmp_output_path)

    # Load venation map
    vein_img = cv2.imread(tmp_output_path, cv2.IMREAD_GRAYSCALE)

    if vein_img is None:
        st.error("Venation extraction failed. Please try a clearer image.")
        # Cleanup
        os.unlink(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.unlink(tmp_output_path)
        st.stop()

    with col2:
        st.markdown("#### Venation Map")
        st.image(vein_img, use_container_width=True)

    # Classify
    with st.spinner("Classifying..."):
        vein_resized = cv2.resize(vein_img, IMAGE_SIZE)
        img_array = np.expand_dims(vein_resized, axis=(0, -1)).astype(np.float32)
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        prediction = class_names[predicted_idx]
        confidence = predictions[predicted_idx]

    # Cleanup temp files
    os.unlink(tmp_input_path)
    if os.path.exists(tmp_output_path):
        os.unlink(tmp_output_path)

    # --- Display results ---
    st.markdown("---")

    info = VENATION_INFO.get(prediction, {})

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Detected Venation Pattern</div>
        <div class="result-value">{info.get('icon', '')} {prediction.upper()}</div>
        <div class="result-label">{confidence*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    st.markdown("#### Confidence Breakdown")

    for i, cls in enumerate(class_names):
        prob = predictions[i]
        cls_info = VENATION_INFO.get(cls, {})
        icon = cls_info.get("icon", "")

        col_label, col_bar, col_pct = st.columns([2, 6, 1])
        with col_label:
            st.markdown(f"{icon} **{cls}**")
        with col_bar:
            st.progress(float(prob))
        with col_pct:
            st.markdown(f"**{prob*100:.1f}%**")

    # Venation type info
    st.markdown("---")
    st.markdown(f"""
    <div class="info-card">
        <strong>{info.get('icon', '')} About {prediction} Venation</strong><br>
        {info.get('description', '')}<br>
        <em>Common examples: {info.get('examples', '')}</em>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show info cards when no image is uploaded
    st.markdown("#### Venation Types")

    cols = st.columns(3)
    for i, (vtype, info) in enumerate(VENATION_INFO.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="info-card">
                <strong>{info['icon']} {vtype}</strong><br>
                {info['description']}<br>
                <em>{info['examples']}</em>
            </div>
            """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#999;font-size:0.85rem;">'
    'Leaf Venation Classifier | Uses CNN on extracted venation maps'
    '</p>',
    unsafe_allow_html=True
)
