---
title: Tea Disease Detector
emoji: 🍃
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered tea disease detector.
---

# Tea Disease Detector 🍃  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-FF4B4B?logo=streamlit)](https://streamlit.io/)  
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/kd8811/tea_disease_detector)  
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)  

---

This project provides a **cloud-deployed deep learning application** that can automatically detect and classify tea leaf diseases from uploaded or captured images. It is designed for researchers, tea growers, and agricultural practitioners who need quick insights into plant health.

---

## 🌱 Supported Classes
The model classifies **four categories**:

- `gl` → **Green Leaf (Healthy)**
- `rr` → **Red Rust**
- `rsm` → **Red Spider Mite**
- `bb` → **Brown Blight**

Each detection is accompanied by a **disease description** and the **prediction confidence**.

---

## 🧠 Model Details
- Model format: **TensorFlow Lite (`.tflite`)**
- Architecture: Custom **CNN-based two-stage pipeline**
- Optimized for **lightweight deployment** and **fast inference**
- Input size: **640 × 640**
- Output: Probability distribution across four classes

The `.tflite` models are hosted directly in the Hugging Face dataset repository and fetched automatically at runtime.

---

## 📊 Dataset
The training dataset is available on Hugging Face:  
🔗 [TeaLeafNet Dataset](https://huggingface.co/kd8811/TeaLeafNet)

---

## 🚀 Deployment
This application has been deployed on **two platforms**:

1. **Streamlit Cloud**  
   🔗 [Tea Disease Detector (Streamlit Cloud)](https://tea-disease-detector-eg2vlyhyufvuydn5n4jjrg.streamlit.app/)

2. **Hugging Face Spaces**  
   🔗 [Tea Disease Detector (Hugging Face)](https://huggingface.co/spaces/kd8811/tea_disease_detector)

Both deployments run the **same Streamlit `app.py`**, ensuring consistent results.

---

## 🖥️ Features
- 📤 Upload images (`jpg`, `jpeg`, `png`)  
- 📷 Capture leaf image using **webcam**  
- ⚡ Real-time disease detection with probability scores  
- 📑 Disease-specific descriptions  
- 🌐 Works on both **Hugging Face Spaces** and **Streamlit Cloud**  

---

## 🛠️ Local Installation & Usage
To run the app locally:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/tea-disease-detector.git
cd tea-disease-detector

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Run Streamlit app
```bash
streamlit run app.py

The app will be available at http://localhost:8501/

## 📦 Requirements

The app uses the following main libraries:

- streamlit

- tensorflow

- opencv-python-headless

- numpy

## 📜 License

This project is licensed under the Apache 2.0 License.

## 🙌 Acknowledgements

- Dataset: TeaLeafNet

- Deployment: Hugging Face Spaces
 & Streamlit Cloud

-- Developed by: Debashis Dey
