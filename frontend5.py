import streamlit as st
import requests
import numpy as np
from PIL import Image


# Backend API URL (FastAPI)

API_URL = "http://127.0.0.1:8000/predict"


# Page configuration
st.set_page_config(
    page_title="textile Predictor",
    layout="centered"
)

st.title(" appreal textile Image Classifier")

st.warning(
    "⚠️ IMPORTANT RULES\n"
    "- Upload ONLY white or grey images\n"
    "- Colored images (red, pink, blue) may fail\n"
    "- Background should be dark\n"
    "- Image will be resized to 28×28 automatically"
)


# Image upload
uploaded_file = st.file_uploader(
    "Upload an image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    # Convert image to grayscale
    image = image.convert("L")

    # Resize image
    image = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize image
    img_array = img_array / 255.0

    # Convert to list (JSON friendly)
    img_list = img_array.tolist()

    # Prediction button

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"image": img_list}
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success(" Prediction Successful")
                    st.write("### Prediction Result")
                    st.write(f"**Class ID:** {result['predicted_class']}")
                    st.write(f"**Label:** {result['predicted_label']}")

                else:
                    st.error(" Backend returned an error")

            except Exception as e:
                st.error("Cannot connect to backend")
                st.error("Cannot connect to backend")
                st.write(e)