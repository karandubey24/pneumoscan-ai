import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG = 224
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="PneumoScan AI", page_icon="🫁")


@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(
        str(ROOT / "models" / "pneumoscan_mobilenetv2.keras"), compile=False)
    with open(ROOT / "models" / "threshold.json") as f:
        thr = float(json.load(f)["threshold"])
    grad = tf.keras.Model(model.inputs,
                          [model.get_layer("out_relu").output, model.output])
    return model, grad, thr


def gradcam(grad_model, img_rgb):
    x = cv2.resize(img_rgb, (IMG, IMG)).astype("float32")
    x = preprocess_input(np.expand_dims(x, 0))
    with tf.GradientTape() as tape:
        conv, p = grad_model(x)
        score = p[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam, float(p[0, 0])


def overlay(img_rgb, cam, alpha=0.4):
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)


def fit(img, size=450):
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)))


st.title("🫁 PneumoScan AI")
st.caption("Chest X-ray pneumonia screening demo (BCA minor project)")
st.warning("Educational prototype. NOT a medical device. Never use it to make "
           "health decisions. Always consult a qualified doctor.")

model, grad_model, thr = load_all()

file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = np.array(Image.open(file).convert("RGB"))
    cam, prob = gradcam(grad_model, img)

    if prob >= thr:
        st.error(f"Possible pneumonia pattern detected (score {prob:.2f}, threshold {thr:.2f})")
    else:
        st.success(f"No pneumonia pattern detected (score {prob:.2f}, threshold {thr:.2f})")
    st.progress(min(max(prob, 0.0), 1.0), text="Model score for pneumonia (0 to 1)")

    c1, c2 = st.columns(2)
    c1.image(fit(img), caption="Uploaded image")
    c2.image(fit(overlay(img, cam)), caption="Grad-CAM: where the model looked")

with st.expander("Limitations (please read)"):
    st.write(
        "- Trained on one public dataset of chest X-rays, which (as far as we know) "
        "comes from young patients at a single hospital. It may not work on adults or "
        "other hospitals.\n"
        "- It only separates Normal vs Pneumonia. It does not detect other diseases.\n"
        "- It cannot check whether an uploaded picture is really a chest X-ray.\n"
        "- It can miss real cases (false negatives) and raise false alarms."
    )