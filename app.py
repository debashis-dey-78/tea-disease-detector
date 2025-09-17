# =====================================================================================
# Library imports
# =====================================================================================
import numpy as np
import streamlit as st
import cv2
import os
import requests  # For downloading from Hugging Face

# Import the TFLite interpreter
try:
    import tflite_runtime.interpreter as tflite  # lightweight package
except ImportError:
    import tensorflow.lite as tflite            # fallback if tflite_runtime unavailable


# =====================================================================================
# Page Configuration (must be first Streamlit command)
# =====================================================================================
st.set_page_config(layout="wide", page_title="Two-Stage Leaf Analysis")


# =====================================================================================
# Model Paths and URLs (Hugging Face)
# =====================================================================================
MODEL_PATH_LNL = "leaf_detection.tflite"
URL_LNL = "https://huggingface.co/kd8811/TeaLeafNet/resolve/main/leaf_detection.tflite"

MODEL_PATH_DISEASE = "disease_classification.tflite"
URL_DISEASE = "https://huggingface.co/kd8811/TeaLeafNet/resolve/main/disease_classification.tflite"


# =====================================================================================
# Utility: download model from Hugging Face
# =====================================================================================
def download_from_hf(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        st.error(f"❌ Error downloading {output_path}: {e}")


# =====================================================================================
# Load models (cached)
# =====================================================================================
@st.cache_resource
def load_all_models():
    interpreters_loaded = {}

    # Stage 1: Leaf / Non-leaf
    if not os.path.exists(MODEL_PATH_LNL):
        with st.spinner(f"Downloading model {MODEL_PATH_LNL}..."):
            download_from_hf(URL_LNL, MODEL_PATH_LNL)
    try:
        interpreter_lnl = tflite.Interpreter(model_path=MODEL_PATH_LNL)
        interpreter_lnl.allocate_tensors()
        interpreters_loaded['lnl'] = interpreter_lnl
    except Exception as e:
        st.error(f"Error loading model {MODEL_PATH_LNL}: {e}")
        interpreters_loaded['lnl'] = None

    # Stage 2: Disease classification
    if not os.path.exists(MODEL_PATH_DISEASE):
        with st.spinner(f"Downloading model {MODEL_PATH_DISEASE}..."):
            download_from_hf(URL_DISEASE, MODEL_PATH_DISEASE)
    try:
        interpreter_disease = tflite.Interpreter(model_path=MODEL_PATH_DISEASE)
        interpreter_disease.allocate_tensors()
        interpreters_loaded['disease'] = interpreter_disease
    except Exception as e:
        st.error(f"Error loading model {MODEL_PATH_DISEASE}: {e}")
        interpreters_loaded['disease'] = None

    return interpreters_loaded.get('lnl'), interpreters_loaded.get('disease')


interpreter_leaf_non_leaf, interpreter_disease = load_all_models()


# =====================================================================================
# UI setup
# =====================================================================================
st.title("🌿 Two-Stage Leaf Analysis")
st.markdown("Upload an image of a tea leaf. The system will first verify if it is a valid leaf and then classify possible diseases or pests.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    plant_image = st.file_uploader("📂 Upload a leaf image", type=["png", "jpg", "jpeg", "webp", "heic"])
with col2:
    captured_image = st.camera_input("📷 Capture with camera")

results_placeholder = st.empty()


# =====================================================================================
# Classes & disease info
# =====================================================================================
CLASS_NAMES_DISEASE = ['bb', 'gl', 'rr', 'rsm']

disease_info = {
    'gl': """
        **Healthy Leaf**  
        No disease detected. Maintain standard good agricultural practices.
    """,
    'rr': """
        **Disease: Red Rust**  
        Caused by *Cephaleuros virescens*.  
        Appears as orange-brown patches on leaves.  
        **Management:** pruning, shade regulation, copper fungicides.
    """,
    'rsm': """
        **Pest: Red Spider Mites**  
        Small mites sucking sap, leading to bronzing and webbing.  
        **Management:** irrigation, natural predators, approved miticides.
    """,
    'bb': """
        **Disease: Brown Blight**  
        Fungal disease (*Colletotrichum spp.*).  
        Circular brown lesions causing defoliation.  
        **Management:** pruning, fungicides, proper spacing.
    """
}


def show_disease_info(result_class):
    with st.expander(f"ℹ️ Learn more about: {result_class.upper()}", expanded=True):
        if result_class in disease_info:
            st.markdown(disease_info[result_class])
        else:
            st.warning("No detailed information available.")


# =====================================================================================
# Main inference logic
# =====================================================================================
if interpreter_leaf_non_leaf and interpreter_disease:
    image_data = None
    source_info = ""

    if plant_image:
        image_data = plant_image.read()
        source_info = f"Uploaded file: `{plant_image.name}`"
    elif captured_image:
        image_data = captured_image.read()
        source_info = "Captured via camera"

    if image_data:
        with results_placeholder.container():
            st.markdown("### 🔎 Results")
            st.write(source_info)

            # Decode image
            file_bytes = np.asarray(bytearray(image_data), dtype=np.uint8)
            opencv_bgr = cv2.imdecode(file_bytes, 1)

            st.image(opencv_bgr, channels="BGR", caption="Input Image", width=300)
            st.write("---")

            # --- Stage 1: Leaf check ---
            st.info("⚙️ Stage 1: Checking if image is a leaf...")
            rgb = cv2.cvtColor(opencv_bgr, cv2.COLOR_BGR2RGB)
            resized_lnl = cv2.resize(rgb, (160, 160)) / 255.0
            input_lnl = np.expand_dims(resized_lnl, axis=0).astype(np.float32)

            in_det_lnl = interpreter_leaf_non_leaf.get_input_details()
            out_det_lnl = interpreter_leaf_non_leaf.get_output_details()
            interpreter_leaf_non_leaf.set_tensor(in_det_lnl[0]['index'], input_lnl)
            interpreter_leaf_non_leaf.invoke()
            pred_lnl = interpreter_leaf_non_leaf.get_tensor(out_det_lnl[0]['index'])

            pred_val = pred_lnl[0][0]
            is_leaf = pred_val <= 0.5
            leaf_conf = (1 - pred_val if is_leaf else pred_val) * 100

            if is_leaf:
                st.success(f"✅ Stage 1: Image is a leaf (Confidence: {leaf_conf:.2f}%)")

                # --- Stage 2: Disease ---
                st.info("⚙️ Stage 2: Disease / Pest detection...")
                resized_dis = cv2.resize(opencv_bgr, (512, 512)) / 255.0
                input_dis = np.expand_dims(resized_dis, axis=0).astype(np.float32)

                in_det_dis = interpreter_disease.get_input_details()
                out_det_dis = interpreter_disease.get_output_details()
                interpreter_disease.set_tensor(in_det_dis[0]['index'], input_dis)
                interpreter_disease.invoke()
                pred_dis = interpreter_disease.get_tensor(out_det_dis[0]['index'])

                result_class = CLASS_NAMES_DISEASE[np.argmax(pred_dis)]
                conf_dis = np.max(pred_dis) * 100

                st.subheader("🧪 Stage 2 Result")
                if result_class == 'gl':
                    st.markdown(f'#### <span style="color:green;">Healthy Leaf</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'#### <span style="color:red;">Detected: {result_class.upper()}</span>', unsafe_allow_html=True)

                st.progress(int(conf_dis))
                st.write(f"Confidence: {conf_dis:.2f}%")
                show_disease_info(result_class)

            else:
                st.error(f"❌ Stage 1: Not a leaf (Confidence: {leaf_conf:.2f}%)")
                st.warning("Upload a clear single-leaf image.")
else:
    st.error("⚠️ One or both AI models could not be loaded. Please refresh and retry.")


# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:grey;font-size:smaller;'>Developed by Harjinder Singh</p>", unsafe_allow_html=True)
