import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gzip
import pickle

# Crear un directorio para guardar las imágenes si no existe
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    """Guarda la imagen subida en el directorio UPLOAD_FOLDER."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    """Preprocesa la imagen para que sea compatible con el modelo."""
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar los píxeles
    image_array = image_array.reshape(1, -1)  # Convertir a vector de 784 características
    return image_array
def main():
  # Widget de subida de archivos
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.subheader("Vista previa de la imagen subida")
        image = Image.open(uploaded_file)

        # Procesar la imagen
        preprocessed_image = preprocess_image(image)

        # Mostrar imágenes antes y después del preprocesamiento
        st.subheader("Imágenes antes y después del preprocesamiento")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen original", use_container_width=True, output_format="auto")
        with col2:
            st.image(preprocessed_image.reshape(28, 28), caption="Imagen preprocesada", use_container_width=True, output_format="auto")

        # Guardar la imagen
        file_path = save_image(uploaded_file)
        st.success(f"Imagen guardada")

        # Diccionario de clases para MNIST
        mnist_classes = {i: str(i) for i in range(10)}

        # Botón para clasificar la imagen
        if st.button("Clasificar imagen"):
            with st.spinner("Cargando modelo y clasificando..."):
                model = load_model()
                prediction = model.predict(preprocessed_image)
                
                # Verificar valores de predicción
                st.success(f"La imagen fue clasificada como: {prediction}")


if __name__ == "__main__":
    main()
