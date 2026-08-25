import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.resnet50 import preprocess_input

# CONFIG
IMG_SIZE = (224, 224)
THRESHOLD = 0.543  # from PR-curve best-F1 sweep in training script (P=0.893, R=0.875)

MODEL_PATH = hf_hub_download(
    repo_id="RAYAN34567/skin_cancer_resnet",
    filename="skin_cancer_resnet50.keras"
)

# PAGE SETUP
st.set_page_config(
    page_title="Skin Cancer Detection AI",
    layout="centered"
)
st.title("DermaVision")
st.write("Upload a skin lesion image to receive a prediction, please make sure the image is in suitable lighting.")

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# PREDICTION
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=233)

    # --- preprocessing must match training exactly ---
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ResNet50 preprocessing (NOT /255)
    # ---------------------------------------------------

    prob = model.predict(img_array, verbose=0)[0][0]
    prediction = "Cancer" if prob >= THRESHOLD else "Non-Cancer"

    st.markdown("---")
    st.subheader("Prediction Result")
    if prediction == "Cancer":
        st.error(f"**Cancer Detected**\n\nProbability: **{prob:.2%}**")
    else:
        st.success(f"**Non-Cancer**\n\nProbability: **{(1 - prob):.2%}**")

# FOOTER
st.markdown("---")
st.write("This tool is for research and educational use only")
