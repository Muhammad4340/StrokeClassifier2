import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('Klasifikasi Pencitraan Otak Penderita Stroke')
stroke_names = ['hemorrhagic', 'ischemic']

# Load the pre-trained model
model = load_model('Stroke_Recog_Modelll.keras')

def classify_images(image_path):
    # Preprocess the image
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Make predictions
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class = np.argmax(result)
    confidence = np.max(result) * 100
    return stroke_names[predicted_class], confidence

def display_information(stroke_type):
    if stroke_type == 'hemorrhagic':
        st.subheader('Definisi Stroke Hemorrhagic')
        st.markdown(
            """
            Stroke hemorrhagic terjadi ketika pembuluh darah di otak pecah, menyebabkan perdarahan.
            Hal ini dapat disebabkan oleh tekanan darah tinggi, aneurisma, atau trauma.
            """
        )
        st.subheader('Pencegahan Stroke Hemorrhagic')
        st.markdown(
            """
            - Menjaga tekanan darah tetap normal
            - Menghindari merokok dan konsumsi alkohol berlebih
            - Mengelola stres dengan baik
            - Mengadopsi pola makan sehat rendah garam dan lemak jenuh
            - Rutin berolahraga
            """
        )
    elif stroke_type == 'ischemic':
        st.subheader('Definisi Stroke Ischemic')
        st.markdown(
            """
            Stroke ischemic terjadi ketika aliran darah ke otak terhambat oleh gumpalan darah atau plak.
            Penyebab utama adalah aterosklerosis atau emboli darah.
            """
        )
        st.subheader('Pencegahan Stroke Ischemic')
        st.markdown(
            """
            - Mengontrol kolesterol dan gula darah
            - Menjaga berat badan ideal
            - Menghindari makanan tinggi lemak dan gula
            - Berhenti merokok
            - Rutin memeriksakan kesehatan ke dokter
            """
        )
uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)

    stroke_type, confidence = classify_images(uploaded_file)
    st.markdown(f"### Hasil Klasifikasi: {stroke_type.capitalize()} dengan tingkat kepercayaan {confidence:.2f}%")

    display_information(stroke_type)
