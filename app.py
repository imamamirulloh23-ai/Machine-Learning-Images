import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(
    page_title="Klasifikasi Sampah AI",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Klasifikasi Sampah Menggunakan Machine Learning")

# ============================
# Load Model
# ============================
model = tf.keras.models.load_model("keras_model.keras")

# ============================
# Load Label
# ============================
labels = []

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ============================
# Upload Gambar
# ============================
uploaded_file = st.file_uploader(
    "Upload gambar sampah",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Gambar yang dipilih", use_container_width=True)

    # Resize sesuai Teachable Machine
    img = image.resize((224,224))

    img = np.asarray(img)

    img = img.astype(np.float32)

    img = (img / 127.5) - 1

    img = np.expand_dims(img, axis=0)

    # Prediksi
    prediction = model.predict(img)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    st.subheader("Hasil Prediksi")

    st.success(labels[index])

    st.write(f"Confidence : **{confidence*100:.2f}%**")

    st.subheader("Seluruh Probabilitas")

    for i, label in enumerate(labels):

        st.write(
            f"{label} : {prediction[0][i]*100:.2f}%"
        )