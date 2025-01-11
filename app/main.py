import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up page configuration
st.set_page_config(page_title="Plant Disease Classification", page_icon="🌱")
# Custom CSS for back arrow button and background image
st.markdown("""
    <style>
        .back-button {
            position: fixed;
            top: 10px;
            left: 10px;
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px;
            font-size: 18px;
            border-radius: 50%;
            cursor: pointer;
            z-index: 1000;
        }
        .back-button:hover {
            background-color: #45a049;
        }
        body {
            background-image: url('https://img.freepik.com/free-vector/leaves-background-with-metallic-foil_79603-956.jpg'); 
            background-size: cover; 
            background-position: center ;
            height: 100vh;  /* Ensure body takes up the full height */
            margin: 0;
        }
    </style>
    <a href="#" class="back-button" onclick="window.history.back()">←</a>
""", unsafe_allow_html=True)

# Working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model/plant_disease_classification.h5"

# Load the pre-trained model without compilation
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
    return img_array

# Predict function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_index)]

# App navigation system
if "page" not in st.session_state:
    st.session_state.page = "landing"

# Landing Page
if st.session_state.page == "landing":
    st.title("Welcome to Plant Disease Classification")

    st.markdown("""
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    """)

    st.header("How It Works")
    st.markdown("""
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    """)

    st.header("Why Choose Us?")
    st.markdown("""
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    """)

    if st.button("Get Started"):
        st.session_state.page = "main"

    st.markdown("""
    &copy; 2024 Plant Health AI. All rights reserved.
    """)

# Main Page
elif st.session_state.page == "main":
    st.title("Plant Disease Classification")
    uploaded_image = st.file_uploader("Upload an image of your plant...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button("Classify"):
                if model is not None:
                    prediction = predict_image_class(model, uploaded_image, class_indices)
                    st.success(f"Prediction: {str(prediction)}")
                else:
                    st.error("Model not loaded. Please check the model file.")

    # "Back to Landing Page" button
    if st.button("←"):
        st.session_state.page = "landing"