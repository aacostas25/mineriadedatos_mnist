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
    # Título de la aplicación
    st.title("MNIST")
    # Descripción inicial
    
    st.write("""
    ### Conoce un poco sobre la base de datos.
    La base de datos MNIST (por sus siglas en inglés, Modified National Institute of Standards and Technology database)1​ es 
    una extensa colección de base de datos que se utiliza ampliamente para el entrenamiento de diversos sistemas de procesamiento de imágenes.
    Y además, transformar los datos mediante la imputación de datos faltantes, la codificación de variables categoricas y la estandarización de los datos.
    
    El conjunto de imágenes de la base de datos MNIST fue creado en 1994 mediante la combinación de dos bases de datos del NIST: la Base de Datos Especial 1 y la Base de Datos Especial 3. La Base de Datos Especial 1 contiene dígitos escritos por estudiantes de secundaria, 
    mientras que la Base de Datos Especial 3 consiste en dígitos escritos por empleados de la Oficina del Censo de Estados Unidos.
    
    
    """)
    st.image('MNISTpicture.png', caption="MNIST")
    st.title("Clasificador de imágenes MNIST")
    st.write("""
    Se clasificó el conjunto de datos MNIST utilizando un modelo basado en **Kernel Ridge Regression (KRR)** 
    con un núcleo RBF y una penalización de `alpha` de 0.1, incorporado en un Pipeline que incluye StandardScaler para la normalización de las imágenes. 
    """)
    st.write("""
    ### Evaluación de Modelos y Técnicas de Preprocesamiento
    
    Para la clasificación de MNIST, se probaron varias configuraciones de modelos y técnicas de preprocesamiento:
    
    - **ElasticNet**: Se evaluaron diferentes valores de `alpha` ([0.1, 0.2, 0.5, 1.0, 10.0, 100.0]) y `l1_ratio` ([0.1, 0.2, 0.5, 1.0]).
      
    - **Kernel Ridge Regression (KRR)**: Se probaron diferentes valores de `alpha` y se evaluaron varios tipos de núcleos: **linear**, **poly**, **rbf**, y **sigmoid**.
    
    Además, se compararon diferentes métodos de escalado:
    - **StandardScaler**
    - **MinMaxScaler**
    - **Sin escalado** (None)
    
    Estas combinaciones fueron probadas para determinar cuál proporciona el mejor rendimiento en la clasificación de los dígitos MNIST.
    """)
    st.write("""
### Opciones para Probar el Modelo

Puedes elegir entre seleccionar una imagen predeterminada de nuestra carpeta de imágenes o subir tu propia imagen para probar el modelo. Si decides subir una imagen, asegúrate de que esté en formato **PNG, JPG o JPEG**. El modelo clasificará la imagen cargada según los dígitos de **MNIST**. Ambas opciones te permiten experimentar con el clasificador y ver los resultados en tiempo real.
""")

    # Opción de subir una imagen
    uploaded_file = st.file_uploader("Sube una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    # Opción de seleccionar una imagen de la carpeta (solo si no se subió una imagen)
    selected_image = None
    if not uploaded_file:
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
        if image_files:
            st.write("Selecciona una imagen de la carpeta:")
            selected_image = st.selectbox("Elige una imagen:", image_files)

    # Mostrar y procesar solo una imagen
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        file_path = save_image(uploaded_file)
        st.subheader("Imágenes antes y después del preprocesamiento")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen original", use_container_width=True, output_format="auto")
        with col2:
            preprocessed_imag = preprocess_image(image)
            st.image(preprocessed_imag.reshape(28, 28), caption="Imagen preprocesada", use_container_width=True, output_format="auto")
            
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
            
    st.markdown('<div class="footer">© Mineria de datos - Clasificación de imágenes MNIST</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
