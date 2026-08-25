import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = (224, 224)

# IMPORTANT:
# Make sure this matches the threshold selected during validation.
# If your best-F1 sweep really produced 0.543, use 0.543 instead.
THRESHOLD = 0.5

MODEL_REPO = "RAYAN34567/skin_cancer_resnet"
MODEL_FILENAME = "skin_cancer_resnet50.keras"

MAX_FILE_SIZE_MB = 10
MAX_IMAGE_PIXELS = 20_000_000


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="DermaVision",
    page_icon="🔬",
    layout="centered",
)

st.title("DermaVision")

st.write(
    "Upload a skin lesion image to receive an AI classification. "
    "For best results, use a clear image with suitable lighting."
)

st.warning(
    "This tool is for research and educational purposes only. "
    "It is not a medical diagnostic device and should not be used "
    "to diagnose or rule out skin cancer."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


try:
    model = load_model()

except Exception as e:
    st.error(
        "The AI model could not be loaded. "
        "Please check the model repository and try again."
    )

    with st.expander("Technical details"):
        st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def load_image(uploaded_file):

    # Protect against very large uploads.
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Image is too large. Maximum size is "
            f"{MAX_FILE_SIZE_MB} MB."
        )

    try:
        image = Image.open(uploaded_file)

        # Validate the image file.
        image.verify()

        # verify() invalidates the image object, so reopen it.
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)

        # Correct camera orientation from EXIF metadata.
        image = ImageOps.exif_transpose(image)

        # Always use RGB because ResNet50 expects 3 channels.
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from e

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("The image has invalid dimensions.")

    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "The image contains too many pixels. "
            "Please upload a smaller image."
        )

    return image


def preprocess_image(image):

    # Match the training image size.
    img = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS,
    )

    img_array = np.asarray(
        img,
        dtype=np.float32,
    )

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(
        img_array,
        axis=0,
    )

    # IMPORTANT:
    # This must match the preprocessing used during training.
    img_array = preprocess_input(img_array)

    return img_array


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:
        image = load_image(uploaded_file)

    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Display uploaded image.
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    # Preprocess image.
    img_array = preprocess_image(image)

    # Run model.
    try:
        raw_prediction = model.predict(
            img_array,
            verbose=0,
        )

        prob = float(
            np.asarray(raw_prediction).reshape(-1)[0]
        )

    except Exception as e:
        st.error("Prediction failed.")

        with st.expander("Technical details"):
            st.exception(e)

        st.stop()

    # Make sure model output is a valid probability.
    prob = float(
        np.clip(prob, 0.0, 1.0)
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    prediction = (
        "Cancer"
        if prob >= THRESHOLD
        else "Non-Cancer"
    )

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == "Cancer":

        st.error(
            f"### Model Classification: Cancer\n\n"
            f"Model probability: **{prob:.2%}**"
        )

    else:

        st.success(
            f"### Model Classification: Non-Cancer\n\n"
            f"Model probability: **{(1 - prob):.2%}**"
        )

    # ========================================================
    # ADDITIONAL INFORMATION
    # ========================================================

    st.progress(
        prob,
        text=f"Model cancer probability: {prob:.2%}",
    )

    st.caption(
        f"Classification threshold: {THRESHOLD:.3f}"
    )

    st.info(
        "This probability is the output of the AI model. "
        "It is not a clinically validated probability of having cancer."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "DermaVision is intended for research and educational use only. "
    "It does not replace evaluation by a qualified healthcare professional."
)
