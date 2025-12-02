import os
from flask import Blueprint, render_template, request, redirect
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2

main = Blueprint("main", __name__)

# ---- Load Model ----
MODEL_PATH = "app/models/pneumonia_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)


# ---- Grad-CAM FUNCTION ----
def generate_gradcam(model, img_array, last_conv_layer_name="conv2d"):
    """
    Creates a Grad-CAM heatmap for a given image array.
    """

    # Get last convolutional layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found. Available layers: {[l.name for l in model.layers]}"
        )

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Weight the convolution channels
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalize heatmap between 0-1
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


# ---------------- ROUTES ---------------- #

@main.route("/")
def index():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    if "xray" not in request.files:
        return redirect(request.url)

    file = request.files["xray"]
    if file.filename == "":
        return redirect(request.url)

    # Save uploaded image
    img_path = os.path.join("app/static", file.filename)
    file.save(img_path)

    # Preprocess
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Confidence
    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence = round(float(confidence) * 100, 2)

    # Label
    result = "Pneumonia Detected" if prediction > 0.5 else "Normal"

    # ----- Generate Grad-CAM -----
    heatmap = generate_gradcam(model, img_array, last_conv_layer_name="conv2d")

    # Load original image
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (150, 150))

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (150, 150))

    # Colorize heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap on image
    superimposed = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    # Save Grad-CAM output
    heatmap_filename = f"heatmap_{file.filename}"
    heatmap_path = os.path.join("app/static", heatmap_filename)
    cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    # Return everything to template
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=file.filename,
        heatmap=heatmap_filename,
    )
