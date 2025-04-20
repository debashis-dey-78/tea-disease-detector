# Library imports
import numpy as np
import streamlit as st
import cv2
import os
import gdown
from tensorflow.keras.models import load_model

# Model file path and Google Drive link
MODEL_FILE = "Tea_disease.keras"
GDRIVE_URL = "https://drive.google.com/uc?id=13Lbwc4KlHGFSiMta407hKgDU7QDAWV2_"

# Download model if not present
if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading model. Please wait..."):
        gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

# Load the model
model = load_model(MODEL_FILE, compile=False)

# Class names
CLASS_NAMES = ['bb', 'rsm', 'gl', 'rr']

# Disease Information
disease_info = {
    'gl': """
        **This is a Non-diseased tea leaf**
    """,
    'rr': """
        **Description :** 
        Red rust is a common disease of tea plants...
    """,
    'rsm': """
        **Description :** 
        Red spider mites are common pests...
    """,
    'bb': """
        **Description :** 
        Brown blight is a common disease of tea plants...
    """
}

# Streamlit UI
st.markdown('<p style="font-size:24px;"><b>Tea Disease Detection (A Plant Disease Detection Tool)</b></p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;"><b>Upload an image of the plant leaf</b></p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">Choose an image...</p>', unsafe_allow_html=True)

# Uploading the image
plant_image = st.file_uploader("", type=["jpeg", "jpg", "png"])
submit = st.button('Predict')

if submit:
    if plant_image is not None:
        # Convert uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image")
        st.write(f"Image shape: {opencv_image.shape}")

        # Preprocess the image
        opencv_image = cv2.resize(opencv_image, (640, 640))
        opencv_image = opencv_image / 255.0
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Predict
        prediction = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display prediction
        st.markdown(f'<p style="font-size:22px;"><b>This is a tea leaf with {result}</b></p>', unsafe_allow_html=True)
        st.markdown(disease_info[result])
        st.write(f"**Confidence:** {confidence:.2f}%")
