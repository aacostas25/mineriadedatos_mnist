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
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

def main():
    st.title("Clasificador de imágenes MNIST")

    # Opción de subir una imagen
    uploaded_file = st.file_uploader("Sube una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    # Opción de seleccionar una imagen de la carpeta
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
    selected_image = None
    if image_files:
        st.subheader("Selecciona una imagen de la carpeta:")
        selected_image = st.selectbox("Elige una imagen:", image_files)

    # Procesar imagen seleccionada de la carpeta
    if selected_image:
        image_path = os.path.join(IMAGE_FOLDER, selected_image)
        image = Image.open(image_path)
        st.image(image, caption=f"Imagen seleccionada: {selected_image}", use_column_width=True)

        if st.button("Clasificar imagen seleccionada"):
            with st.spinner("Clasificando..."):
                predicted_class = classify_image(image)
                st.success(f"La imagen fue clasificada como: {predicted_class}")

    # Procesar imagen subida por el usuario
    if uploaded_file:
        image = Image.open(uploaded_file)
        file_path = save_image(uploaded_file)

        st.subheader("Vista previa de la imagen subida")
        st.image(image, caption="Imagen original", use_column_width=True)

        if st.button("Clasificar imagen subida"):
            with st.spinner("Clasificando..."):
                predicted_class = classify_image(image)
                st.success(f"La imagen fue clasificada como: {predicted_class}")

if __name__ == "__main__":
    main()
