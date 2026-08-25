import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2



IMG_SIZE = (224, 224)

THRESHOLD = 0.5

MODEL_PATH = hf_hub_download(
    repo_id="RAYAN34567/skin_cancer_resnet",
    filename="skin_cancer_resnet50.keras"
)



st.set_page_config(
    page_title="Skin Cancer Detection AI",
    layout="centered"
)

st.title("DermaVision")

st.write(
    "Upload a skin lesion image to receive a prediction. "
    "Please make sure the image is in suitable lighting."
)


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()



def find_last_conv_layer(model):

    # Search from the end of the model
    for layer in reversed(model.layers):

        try:
            output_shape = layer.output.shape

            # Convolutional feature maps normally have:
            # (batch, height, width, channels)

            if len(output_shape) == 4:
                return layer

        except Exception:
            continue

    return None



def make_gradcam_heatmap(img_array, model):

    last_conv_layer = find_last_conv_layer(model)

    if last_conv_layer is None:
        raise ValueError(
            "Could not find a suitable convolutional layer "
            "for Grad-CAM."
        )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )


        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(
            predictions
        )

  

        if len(predictions.shape) == 1:
            predictions = tf.expand_dims(
                predictions,
                axis=-1
            )

        class_channel = predictions[:, 0]

    # Calculate gradients
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:
        raise ValueError(
            "Gradients could not be calculated. "
            "The selected layer may not be connected "
            "to the model's output."
        )

    # Average gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight every feature map according to importance
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Only positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return (
        heatmap.numpy(),
        last_conv_layer.name
    )


uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=233
    )

    img = image.resize(
        IMG_SIZE
    )

    img_array = np.array(
        img,
        dtype="float32"
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )


    raw_prediction = model.predict(
        img_array,
        verbose=0
    )

    if isinstance(
        raw_prediction,
        (list, tuple)
    ):
        raw_prediction = raw_prediction[0]

    raw_prediction = np.asarray(
        raw_prediction
    )

    prob = float(
        raw_prediction.reshape(-1)[0]
    )


    prediction = (
        "Cancer"
        if prob >= THRESHOLD
        else "Non-Cancer"
    )


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


    st.markdown("---")

    st.subheader(
        "Why did the model make this prediction?"
    )

    st.write(
        "Grad-CAM highlights the regions of the image "
        "that contributed most strongly to the model's prediction."
    )

    try:

        # Generate Grad-CAM
        heatmap, layer_used = make_gradcam_heatmap(
            img_array,
            model
        )


        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        # Apply color map
        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # Convert BGR -> RGB
        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )


        original = np.array(
            image
        )

        # Resize heatmap to original image size
        heatmap_color = cv2.resize(
            heatmap_color,
            (
                original.shape[1],
                original.shape[0]
            )
        )


        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        st.image(
            overlay,
            caption=(
                "Grad-CAM: Regions influencing "
                "the model's prediction"
            ),
            use_container_width=True
        )


        st.info(
            f"""
            **Model explanation**

            The highlighted regions represent areas that
            contributed most strongly to the model's
            **{prediction}** prediction.

            **Grad-CAM layer:** `{layer_used}`
            """
        )

        st.caption(
            "Red/yellow regions indicate stronger influence "
            "on the model's prediction, while blue regions "
            "indicate weaker influence."
        )

        st.caption(
            "Important: Grad-CAM shows which regions influenced "
            "the neural network. It does not prove that a "
            "highlighted region is medically cancerous."
        )

    except Exception as e:

        st.error(
            "Grad-CAM could not be generated."
        )

        st.code(
            str(e)
        )


st.markdown("---")

st.write(
    "This tool is for research and educational use only."
)
