# smart_recycle_app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Load trained CNN model
cnn_model = load_model("my_cnn_model.h5")

# Class labels
class_names = ['Compostable', 'Non-Recyclable', 'Recyclable']

# Preprocessing function
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.set_page_config(page_title="Smart Recycle Classifier ♻️")
st.title("Smart Recycle Classifier ♻️")
st.write("Upload a waste image to classify it into recyclable, non-recyclable, or compostable.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Convert to PIL for resizing
    image_pil = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    processed_img = preprocess_image(image_pil)

    # Predict
    predictions = cnn_model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")
