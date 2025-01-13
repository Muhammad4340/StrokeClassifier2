import os #menyimpan file gambar untuk klasifikasi
import keras #melakukan klasifikasi dan load model
from keras.models import load_model #load model
import streamlit as st #framework frontend python
import tensorflow as tf #operasi klasifikasi
import numpy as np #memuat hasil prediksi klasifikasi

# Header Aplikasi dan jenis klasifikasi stroke
st.header('Klasifikasi Pencitraan Otak Penderita Stroke')
stroke_names = ['hemorrhagic', 'ischemic']

# Load model untuk klasifikasi stroke
model = load_model('Stroke_Recog_Modelll.keras')

# fungsi klasifikasi
def classify_images(image_path):
    # merubah ukuran inputan gambar ke ukuran 180x180 piksel
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    # merubah gambar menjadi array numerik
    input_image_array = tf.keras.utils.img_to_array(input_image)
    # menambahkan batch sesuai dengan inputan model
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Membuat prediksi klasifikasi
    predictions = model.predict(input_image_exp_dim)
    # Menghitung prediksi klasifikasi dengan hitungan probabilitas
    result = tf.nn.softmax(predictions[0])
    # memuat hasil prediksi
    predicted_class = np.argmax(result)
    # menyesuaikan tingkat keakuratan model dikalikan 100
    confidence = np.max(result) * 100
    # menampilkan hasil klasifikasi, lalu ditampilkan tingkat probabilitas prediksi klasifikasi
    return stroke_names[predicted_class], confidence

# fungsi menampilkan informasi hasil klasifikasi untuk setiap jenis kelas
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

# Membuat tombol untuk upload Gambar
uploaded_file = st.file_uploader('Upload an Image')
# Mengecek apakah gambar yang diupload pengguna sudah masuk
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Menampilkan gambar yang diunggah pada Antarmuka Web dengan ukuran 200 piksel
    st.image(uploaded_file, width=200)

    # Melakukan klasifikasi dan menghitung nilai probabilitas klasifikasi
    stroke_type, confidence = classify_images(uploaded_file)
    st.markdown(f"### Hasil Klasifikasi: {stroke_type.capitalize()} dengan tingkat kepercayaan {confidence:.2f}%")

    # Menampilkan hasil informasi untuk pengguna berkaitan jenis klasifikasi yang telah di prediksi
    display_information(stroke_type)
