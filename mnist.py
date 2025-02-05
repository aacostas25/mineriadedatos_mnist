import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Título de la aplicación
st.title("Exploración de datos: Titanic")
st.write("""
### ¡Bienvenidos!
Esta aplicación interactiva permite explorar el dataset de Mnist.
""")
