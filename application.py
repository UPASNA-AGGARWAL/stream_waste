import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import io


st.write("""# WASTE CLASSIFICATION - SCAN YOUR WASTE""")

model = load_model('resnet.h5')
def preprocess_image(image1):
    # Resize the image to match model input size
    image = image1.resize((224, 224))
    # Convert image to numpy array
    image_array = img_to_array(image)
    # Expand dimensions to match the model's expected input shape
    image_array = np.expand_dims(image_array, axis=0)
    # Scale pixel values to [0, 1]
    image_array = image_array / 255.0
    return image_array

def prediction(image_array):

    class_prediction = model.predict(image_array)
    return class_prediction
       

st.title("Waste classification using Machine Learning Model")
file = st.file_uploader("Please upload an image", type=["jpg","jpeg", "png"])


classify = st.button("classify image")
if file is not None:
    st.write("")
    st.write("Classifying...")
    
    try:
        # Read the image
        image = Image.open(file)
        # Preprocess the image
        image_array = preprocess_image(image)
        # Make prediction
        label = prediction(image_array)
        st.write(label)
        classes_x = np.argmax(label, axis = 1)
        if classes_x == 1:
            waste = "Recyclable"
        else:
            waste = "Organic"

        st.write(f"This image most likely belongs to the category of {waste}")
    except Exception as e:
        st.write("Error:", e)
        st.write("Please upload a valid image file.")

