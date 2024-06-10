import streamlit as st
import pandas as pd
import pickle
import base64

# Cargar el modelo K-Modes preentrenado
with open("data/best_kmodes.pkl", "rb") as f:
    kmodes_model = pickle.load(f)

# CSS para el fondo
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file('images/background.png')

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{img_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Interfaz de usuario en Streamlit
st.title('Predicción de Clúster')

categoria = st.selectbox("Seleccione la categoría:", ["Deportes", "Política", "Economía", "Salud y Bienestas", "El Mundo", "Edición Impresa"])
sentimiento = st.selectbox("Seleccione el sentimiento:", [0, 1, 2])
titulo = st.text_input("Ingrese el título:")
subtitulo = st.text_input("Ingrese el subtítulo:")
autor = st.selectbox("Seleccione el tipo de autor:", [0, 1])

if st.button('Predecir Clúster'):
    # Transformar los inputs a un formato adecuado para el modelo
    bins_titulo = [0, 13, 17, float('inf')]
    labels_titulo = ['Corto', 'Mediano', 'Largo']
    rango_titulo = pd.cut([len(titulo.split())], bins=bins_titulo, labels=labels_titulo)[0]

    bins_subtitulo = [0, 22, 30, float('inf')]
    labels_subtitulo = ['Corto', 'Mediano', 'Largo']
    rango_subtitulo = pd.cut([len(subtitulo.split())], bins=bins_subtitulo, labels=labels_subtitulo)[0]

    input_data = pd.DataFrame({
        'categoria': [categoria],
        'sentiment': [sentimiento],
        'tipo_autor': [autor],
        'rangotitulo': [rango_titulo],
        'rangosubtitulo': [rango_subtitulo],
        'topic': 7
    })

    # Predecir el clúster
    cluster = kmodes_model.predict(input_data)
    st.write(f"El clúster de la nota es: {cluster[0]}")

# Para ejecutar:
# streamlit run app.py
