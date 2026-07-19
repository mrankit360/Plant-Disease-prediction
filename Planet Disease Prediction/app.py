import os
import json
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model("plant_disease_prediction_model.h5")
class_indices = json.load(open("class_indices.json"))

def load_and_preprocess_image(image_path, target_size=(224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float') / 255.0
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

st.title('🪴 Planet Disease Classifier')

uploaded_image = st.file_uploader('Upload an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        resized_image = image.resize((200, 200))
        st.image(resized_image)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'prediction: {str(prediction)}')