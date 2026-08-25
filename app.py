import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import cv2


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = (224, 224)

# Threshold obtained from your PR-curve best-F1 sweep
# P = 0.893, R = 0.875
THRESHOLD = 0.5


# ============================================================
# DOWNLOAD MODEL
# ============================================================

MODEL_PATH = hf_hub_download(
    repo_id="RAYAN34567/skin_cancer_resnet",
    filename="skin_cancer_resnet50.keras"
)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Skin Cancer Detection AI",
    layout="centered"
)

st.title("DermaVision")

st.write(
    "Upload a skin lesion image to receive a prediction. "
    "Please make sure the image is in suitable lighting."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    # Model that returns:
    # 1. feature maps from the final convolutional layer
    # 2. final prediction

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # Binary classifier output
        class_channel = predictions[:, 0]

    # Gradient of prediction with respect to feature maps
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Average gradients over width and height
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight each feature map by its importance
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # Only keep positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    if max_value != 0:
        heatmap /= max_value

    return heatmap.numpy()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Load image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        width=233
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    # Resize to model input size
    img = image.resize(
        IMG_SIZE
    )

    # Convert to NumPy
    img_array = np.array(
        img,
        dtype="float32"
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ResNet50 preprocessing
    # IMPORTANT:
    # Do NOT divide by 255 here.
    img_array = preprocess_input(
        img_array
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prob = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # Classification
    prediction = (
        "Cancer"
        if prob >= THRESHOLD
        else "Non-Cancer"
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Prediction Result"
    )

    if prediction == "Cancer":

        st.error(
            f"""
            **Cancer Detected**

            Probability: **{prob:.2%}**
            """
        )

    else:

        st.success(
            f"""
            **Non-Cancer**

            Probability: **{(1 - prob):.2%}**
            """
        )

    # ========================================================
    # GRAD-CAM EXPLANATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Why did the model make this prediction?"
    )

    st.write(
        "Grad-CAM highlights the regions of the image "
        "that contributed most strongly to the model's prediction."
    )

    try:

        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "conv5_block3_out"
        )

        # ----------------------------------------------------
        # Convert heatmap to image
        # ----------------------------------------------------

        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        # Apply color map
        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # OpenCV uses BGR
        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Resize heatmap to original image
        # ----------------------------------------------------

        original = np.array(
            image
        )

        heatmap_color = cv2.resize(
            heatmap_color,
            (
                original.shape[1],
                original.shape[0]
            )
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ----------------------------------------------------
        # Display Grad-CAM
        # ----------------------------------------------------

        st.image(
            overlay,
            caption=(
                "Grad-CAM: Regions influencing "
                "the model's prediction"
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Explanation text
        # ----------------------------------------------------

        if prediction == "Cancer":

            st.info(
                """
                **Model explanation:**

                The highlighted regions represent areas that
                contributed most strongly to the model's
                Cancer prediction.
                """
            )

        else:

            st.info(
                """
                **Model explanation:**

                The highlighted regions represent areas that
                contributed most strongly to the model's
                Non-Cancer prediction.
                """
            )

        # ----------------------------------------------------
        # Scientific disclaimer
        # ----------------------------------------------------

        st.caption(
            "Grad-CAM shows which image regions influenced "
            "the neural network. It does not prove that the "
            "highlighted regions are medically cancerous."
        )

    except Exception as e:

        st.warning(
            "Grad-CAM could not be generated."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "This tool is for research and educational use only."
)
