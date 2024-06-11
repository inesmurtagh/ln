import streamlit as st
import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pickle
import nltk
import requests
from nltk.stem import WordNetLemmatizer
from kmodes.kmodes import KModes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
import os
import base64

# CSS para el fondo y el color del texto
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    img_base64 = get_base64_of_bin_file('images/background.png')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
            color: white;
        }}
        .stSelectbox, .stTextInput, .stButton > button {{
            color: white;
        }}
        .stSelectbox label, .stTextInput label, .stButton > button {{
            color: white;
        }}
        .stSelectbox div[role="combobox"] > div {{
            color: #008cff;
        }}
        .stMarkdown {{
            color: white;
        }}
        /* Ocultar la barra de menú superior */
        header {{
            visibility: hidden;
        }}
        /* Ocultar el botón de gestión de la aplicación */
        footer {{
            visibility: hidden;
        }}
        .stButton > button {{
            color: #008cff;
            background-color: white;
            border: 2px solid #008cff;
        }}
        .stButton > button:hover {{
            color: white;
            background-color: #008cff;
            border-color: #008cff;  /* Cambiar el color del borde a negro cuando se pasa el cursor por encima */
        }}
        .stButton > button:active {{
            color: white;
            background-color: #0056b3;
            border-color: #0056b3;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.error("No se encontró la imagen de fondo. Asegúrate de que 'background.png' está en la carpeta 'images'.")

# Interfaz de usuario en Streamlit
condition_options = ['Autos', 'Construcción y Diseño','Propiedades e Inmuebles','Deportes','Negocios y Economía','Salud y Bienestar','El Mundo','Entretenimiento','Lifestyle','Edición impresa','Política','Sociedad']
categoria = st.selectbox("Seleccione la categoria", options=condition_options)
sentimiento = st.selectbox("Seleccione el sentimiento:", ["Negativo", "Neutral", "Positivo"])
titulo = st.text_input("Ingrese el título:")
subtitulo = st.text_input("Ingrese el subtítulo:")
autor = st.selectbox("Seleccione el tipo de autor:", ["Usuario", "Firma"])

# Función para codificar 'categoria'
def encode_categoria(categoria):
    if categoria == 'Autos':
        return 0
    elif categoria == 'Construcción y Diseño':
        return 1
    elif categoria == 'Deportes':
        return 2
    elif categoria == 'Edición impresa':
        return 3
    elif categoria == 'El Mundo':
        return 4
    elif categoria == 'Entretenimiento':
        return 5
    elif categoria == 'Lifestyle':
        return 6
    elif categoria == 'Negocios y Economía':
        return 7
    elif categoria == 'Política':
        return 8
    elif categoria == 'Propiedades e Inmuebles':
        return 9
    elif categoria == 'Salud y Bienestar':
        return 10
    elif categoria == 'Sociedad':
        return 11

# Función para codificar 'rangotitulo' y 'rangosubtitulo'
def encode_rango(rango):
    if rango == 'Corto':
        return 0
    elif rango == 'Largo':
        return 1
    elif rango == 'Mediano':
        return 2

# Función para codificar 'rangotitulo' y 'rangosubtitulo'
def de_encode_rango(rango):
    if rango == 0:
        return 'Corto'
    elif rango == 1:
        return 'Largo'
    else:
        return 'Mediano'

# Función para decodificar 'sentimiento'
def de_encode_sentimiento(sentimiento):
    if sentimiento == 0:
        return 'Negativo'
    elif sentimiento == 1:
        return 'Neutral'
    else:
        return 'Positivo'

# Función para decodificar 'pregunta'
def de_encode_pregunta(pregunta):
    if pregunta == 0:
        return 'Sin Pregunta'
    else:
        return 'Con Pregunta'

def cargar_modelo_lda(filename):
    return LdaModel.load(filename)

lda_model = cargar_modelo_lda("data/lda_model.gensim")

def cargar_diccionario(filename):
    return Dictionary.load(filename)

dictionary = cargar_diccionario("data/diccionario.gensim")

# Cargar el DataFrame
df = pd.read_csv('clusters.csv')

# Función para modelo de clasificación
def modelo_clas(df):
    df['categoria_encoded'] = df['categoria'].apply(encode_categoria)
    X = df[["categoria_encoded","tipo_autor","sentiment","rangotitulo_encoded","rangosubtitulo_encoded","topics"]]
    y = df["cluster"]
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar un modelo de clasificación (Random Forest en este caso)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Predecir con el conjunto de prueba
    y_pred = model.predict(X_test)

    return model

# Función para pre-procesamiento del texto
def preprocess_text(text):
    # División del texto en tokens por espacios en blanco
    tokens = text.split()

    # Verificar si el archivo de stopwords existe
    stopwords_path = "data/spanish"
    if not os.path.exists(stopwords_path):
        st.error(f"No se encontró el archivo de stopwords en {stopwords_path}.")
        return tokens
    
    # Carga de stopwords en español desde el archivo CSV
    stop_words_df = pd.read_csv(stopwords_path, delimiter="\t", header=None)
    stop_words = set(stop_words_df[0].tolist())
    
    # Eliminación de stopwords del texto
    preprocessed_text = [token for token in tokens if token.lower() not in stop_words]
    
    return preprocessed_text

# Función para predecir el tópico utilizando LDA
def predict_topic(text, lda_model, dictionary):
    tokens = preprocess_text(text)
    bow = dictionary.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0]  # Retorna el tópico con la mayor probabilidad

# Función para predecir el cluster
def predict_cluster(categoria, sentimiento, titulo, subtitulo, autor):
    categoria = encode_categoria(categoria)

    # Transformar el sentimiento a valores numéricos
    sentiment_value = {'Negativo': 0, 'Neutral': 1, 'Positivo': 2}[sentimiento]
    
    # Transformar el sentimiento a valores numéricos
    autor_value = {'Usuario': 0, 'Firma': 1}[autor]
    
    # Calcular rangotitulo y rangosubtitulo
    bins_titulo = [0, 13, 17, float('inf')]
    labels_titulo = ['Corto', 'Mediano', 'Largo']
    rango_titulo = pd.cut([len(titulo.split())], bins=bins_titulo, labels=labels_titulo)[0]
    rango_titulo = encode_rango(rango_titulo)

    bins_subtitulo = [0, 22, 30, float('inf')]
    labels_subtitulo = ['Corto', 'Mediano', 'Largo']
    rango_subtitulo = pd.cut([len(subtitulo.split())], bins=bins_subtitulo, labels=labels_subtitulo)[0]
    rango_subtitulo = encode_rango(rango_subtitulo)

    # Predecir el tópico usando LDA
    texto_completo = titulo + " " + subtitulo
    topic = predict_topic(texto_completo, lda_model, dictionary)
    
    st.write(f"Tópico asignado: {topic}")  # Mostrar el tópico asignado

    # Crear un DataFrame con los valores procesados
    input_data = pd.DataFrame({
        'categoria_encoded': [categoria],
        'tipo_autor': [autor_value],
        'sentiment': [sentiment_value],
        'rangotitulo_encoded': [rango_titulo],
        'rangosubtitulo_encoded': [rango_subtitulo],
        'topics': [topic]
    })

    # Predecir el cluster
    cluster = modelo_clasificacion.predict(input_data)
    st.write(f"Cluster asignado: {cluster[0]}")  # Mostrar el cluster asignado
    return cluster[0]

def evaluar_individuo(individuo, df_cluster, benchmark_cluster):
    sentimiento, tipo_autor, titulo, subtitulo, pregunta = individuo

    pageviews_mean = df_cluster[
        (df_cluster['sentiment'] == sentimiento) &
        (df_cluster['tipo_autor'] == tipo_autor) &
        (df_cluster['rangotitulo_encoded'] == titulo) &
        (df_cluster['rangosubtitulo_encoded'] == subtitulo) &
        (df_cluster['pregunta'] == pregunta)
    ]['pageviews'].mean()

    if np.isnan(pageviews_mean):
        return -np.inf,

    noise = np.random.normal(0, df_cluster['pageviews'].std())

    if pageviews_mean > 0:
        variation = ((pageviews_mean - benchmark_cluster) / benchmark_cluster) * 100 + noise
    else:
        variation = -np.inf

    if variation > 0:
        variation = np.log1p(variation)

    return variation,

# Función para algoritmos geneticos
def aplicar_algoritmos_geneticos_para_cluster(clusters, cluster_objetivo):
    total_notas = len(clusters)
    df = clusters[clusters['cluster'] == cluster_objetivo]
    benchmark_cluster = df['pageviews'].mean()
    estrategias_recomendadas = []
    
    combination_counts = df.groupby(['sentiment', 'tipo_autor', 'rangotitulo_encoded', 'rangosubtitulo_encoded', 'pregunta']).size()
    percentil_25 = combination_counts.quantile(0.25)
    valid_combinations = combination_counts[combination_counts > percentil_25].index

    def attr_sentimiento():
        return np.random.choice([comb[0] for comb in valid_combinations])

    def attr_tipo_autor():
        return np.random.choice([comb[1] for comb in valid_combinations])

    def attr_titulo():
        return np.random.choice([comb[2] for comb in valid_combinations])

    def attr_subtitulo():
        return np.random.choice([comb[3] for comb in valid_combinations])

    def attr_pregunta():
        return np.random.choice([comb[4] for comb in valid_combinations])

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_sentimiento", attr_sentimiento)
    toolbox.register("attr_tipo_autor", attr_tipo_autor)
    toolbox.register("attr_titulo", attr_titulo)
    toolbox.register("attr_subtitulo", attr_subtitulo)
    toolbox.register("attr_pregunta", attr_pregunta)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_sentimiento, toolbox.attr_tipo_autor, toolbox.attr_titulo, toolbox.attr_subtitulo, toolbox.attr_pregunta), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluar_individuo, df_cluster=df, benchmark_cluster=benchmark_cluster)
    toolbox.register("mate", tools.cxTwoPoint)

    def custom_mutate(individual, indpb):
        for i in range(len(individual)):
            if np.random.rand() < indpb:
                if i == 0:
                    individual[i] = attr_sentimiento()
                elif i == 1:
                    individual[i] = attr_tipo_autor()
                elif i == 2:
                    individual[i] = attr_titulo()
                elif i == 3:
                    individual[i] = attr_subtitulo()
                elif i == 4:
                    individual[i] = attr_pregunta()
        return individual,

    toolbox.register("mutate", custom_mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=100)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hall_of_fame, verbose=True)

    best_ind = hall_of_fame[0]
    variacion = best_ind.fitness.values[0]
    pv_esperadas = benchmark_cluster * (1 + variacion / 100)

    peso_ponderado = len(df) / total_notas
    variacion_ponderada = variacion * peso_ponderado

    estrategias_recomendadas.append(best_ind)    
    return estrategias_recomendadas

if st.button('Obtener recomendaciones'):
    if not titulo and not subtitulo:
        st.markdown('<p style="color:white;background-color:#f44336;padding:8px;border-radius:5px;">Por favor ingrese un título y un subtítulo.</p>', unsafe_allow_html=True)
    elif not titulo:
        st.markdown('<p style="color:white;background-color:#f44336;padding:8px;border-radius:5px;">Por favor ingrese un título.</p>', unsafe_allow_html=True)
    elif not subtitulo:
        st.markdown('<p style="color:white;background-color:#f44336;padding:8px;border-radius:5px;">Por favor ingrese un subtítulo.</p>', unsafe_allow_html=True)
    else:
        try:
            modelo_clasificacion = modelo_clas(df)
            cluster = predict_cluster(categoria, sentimiento, titulo, subtitulo, autor)
            estrategia_recomendada = aplicar_algoritmos_geneticos_para_cluster(df, cluster)
            st.write(f"Estrategia recomendada para el sentimiento: {de_encode_sentimiento(estrategia_recomendada[0][0])}")
            st.write(f"Estrategia recomendada para el titulo: {de_encode_rango(estrategia_recomendada[0][2])}")
            st.write(f"Estrategia recomendada para el subtitulo: {de_encode_rango(estrategia_recomendada[0][3])}")
            st.write(f"Estrategia recomendada para el pregunta: {de_encode_pregunta(estrategia_recomendada[0][4])}")
        except ValueError as e:
            st.write(f"Error: {e}")
