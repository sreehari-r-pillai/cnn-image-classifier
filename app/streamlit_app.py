
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from src.data import CLASS_NAMES

st.set_page_config(page_title="Image Classifier", page_icon="🧠")

st.title("🧠 Image Classification (Keras)")
st.write("Upload an image and get predictions from a trained model.")

dataset = st.selectbox("Choose dataset / class mapping", ["fashion_mnist", "cifar10"], index=0)
class_names = CLASS_NAMES[dataset]

model_path = st.text_input("Path to saved Keras model (.h5)", "models/fashion_mnist_simple_cnn.h5")
uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

def preprocess(img: Image.Image, target_shape):
    img = img.convert("RGB")
    img = img.resize((target_shape[1], target_shape[0]))
    arr = np.array(img).astype("float32") / 255.0
    if target_shape[2] == 1:
        arr = np.mean(arr, axis=2, keepdims=True)  # convert to grayscale
    arr = np.expand_dims(arr, 0)  # batch dim
    return arr

if st.button("Load model"):
    try:
        st.session_state["model"] = tf.keras.models.load_model(model_path)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if uploaded and "model" in st.session_state:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded", use_column_width=True)
    model = st.session_state["model"]
    target_shape = model.input_shape[1:]
    x = preprocess(image, target_shape)
    probs = model.predict(x)[0]
    pred_idx = int(np.argmax(probs))
    st.markdown(f"**Prediction:** {class_names[pred_idx]}")
    st.bar_chart(probs)
