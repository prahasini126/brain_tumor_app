import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Load the model
model = load_model('brain_tumor_cnn_model.h5')

# Header
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4B8BBE;">🧠 Brain Tumor Detection</h1>
        <p style="font-size: 18px; color: #666;">Upload an MRI image and let AI predict the presence of a brain tumor.</p>
    </div>
    <hr style="border: 1px solid #ddd;">
""", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("📤 Upload MRI Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Display and predict
if uploaded_file is not None:
    with st.spinner("⏳ Processing Image..."):
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Convert to grayscale
        image = image.convert("L")
        image = image.resize((128, 128))
        img = np.array(image)
        img = img.reshape(1, 128, 128, 1)
        img = img / 255.0

        # Prediction
        prediction = model.predict(img)[0][0]

    # Result
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)
    st.subheader("🧪 Prediction Result:")

    if prediction > 0.5:
        st.error("⚠️ Brain Tumor Detected")
        st.markdown("<p style='color: red;'>Probability: {:.2f}</p>".format(prediction), unsafe_allow_html=True)
    else:
        st.success("✅ No Tumor Detected")
        st.markdown("<p style='color: green;'>Probability: {:.2f}</p>".format(1 - prediction), unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 1px solid #eee;">
    <p style="text-align: center; font-size: 14px; color: #999;">Developed using TensorFlow and Streamlit 💻</p>
""", unsafe_allow_html=True)
