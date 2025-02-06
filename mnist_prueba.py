import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gzip
import pickle

# Definir carpetas
UPLOAD_FOLDER = "uploaded_images"
IMAGE_FOLDER = "images"

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    """Guarda la imagen subida en el directorio UPLOAD_FOLDER."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_model():
    """Cargar el modelo desde un archivo comprimido."""
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    """Preprocesa la imagen para que sea compatible con el modelo."""
    image = image.convert('L')  # Escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar
    image_array = image_array.reshape(1, -1)  # Vector de características
    return image_array

def classify_image(image):
    """Clasifica la imagen con el modelo."""
    preprocessed_image = preprocess_image(image)
    model = load_model()
    predicted_class = model.predict(preprocessed_image)
    return predicted_class

def main():
    st.title("Clasificador de imágenes MNIST")

    # Opción de subir una imagen
    uploaded_file = st.file_uploader("Sube una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    # Opción de seleccionar una imagen de la carpeta (solo si no se subió una imagen)
    selected_image = None
    if not uploaded_file:
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
        if image_files:
            st.subheader("Selecciona una imagen de la carpeta:")
            selected_image = st.selectbox("Elige una imagen:", image_files)

    # Mostrar y procesar solo una imagen
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        file_path = save_image(uploaded_file)
        st.subheader("Imagen subida")
        st.image(image, caption="Imagen original", use_column_width=True)
    elif selected_image:
        image_path = os.path.join(IMAGE_FOLDER, selected_image)
        image = Image.open(image_path)
        st.subheader("Imágenes antes y después del preprocesamiento")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen original", use_container_width=True, output_format="auto")
        with col2:
            preprocessed_imag = preprocess_image(image)
            st.image(preprocessed_imag.reshape(28, 28), caption="Imagen preprocesada", use_container_width=True, output_format="auto")
            
    # Botón para clasificar la imagen mostrada
    if image and st.button("Clasificar imagen"):
        with st.spinner("Clasificando..."):
            predicted_class = classify_image(image)
            st.success(f"La imagen fue clasificada como: {predicted_class[0]}")

if __name__ == "__main__":
    main()
