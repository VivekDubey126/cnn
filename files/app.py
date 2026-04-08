import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector")

st.title("🌿 Plant Disease Detector")

# Load model
model = tf.keras.models.load_model("plant_model.h5")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]

        class_index = np.argmax(preds)
        confidence = np.max(preds)*100

        st.success(f"Prediction: Class {class_index}")
        st.write(f"Confidence: {confidence:.2f}%")