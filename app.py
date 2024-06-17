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
import re
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

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
        .stSelectbox, .stTextArea, .stTextInput, .stButton > button {{
            color: white;
        }}
        .stSelectbox label, .stTextArea label, .stTextInput label, .stButton > button {{
            color: white;
        }}
        .stTextArea div[role="textbox"] > div {{
            color: white;
        }}
        .stSelectbox div[role="combobox"] > div {{
            color: #008cff;
        }}
        .stMarkdown {{
            color: white;
            line-height: 1; /* Adjust line height for markdown */
        }}
        header {{
            visibility: hidden;
        }}
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
            border-color: #008cff;
        }}
        .stButton > button:active {{
            color: white;
            background-color: #0056b3;
            border-color: #0056b3;
        }}
        .block-container {{
            max-width: 90%;
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .css-1lcbmhc {{
            display: flex;
            justify-content: space-between;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
except FileNotFoundError:
    st.error("No se encontró la imagen de fondo. Asegúrate de que 'background.png' está en la carpeta 'images'.")

# Dividir la página en dos columnas
col1, col2 = st.columns(2, gap="large")


# Columna izquierda: entrada del usuario
with col1:
    condition_options = ['Autos', 'Construcción y Diseño','Propiedades e Inmuebles','Deportes','Negocios y Economía','Salud y Bienestar','El Mundo','Entretenimiento','Lifestyle','Edición impresa','Política','Sociedad']
    categoria = st.selectbox("Seleccione la categoria", options=condition_options)
    sentimiento = st.selectbox("Seleccione el sentimiento:", ["Negativo", "Neutral", "Positivo"])
    titulo = st.text_area("Ingrese el título:")
    subtitulo = st.text_area("Ingrese el subtítulo:")
    autor = st.selectbox("Seleccione el tipo de autor:", ["Usuario", "Firma"])

def encode_categoria(categoria):
    mapping = {
        'Autos': 0, 'Construcción y Diseño': 1, 'Deportes': 2, 'Edición impresa': 3, 
        'El Mundo': 4, 'Entretenimiento': 5, 'Lifestyle': 6, 'Negocios y Economía': 7, 
        'Política': 8, 'Propiedades e Inmuebles': 9, 'Salud y Bienestar': 10, 'Sociedad': 11
    }
    return mapping.get(categoria, -1)

def encode_rango(rango):
    mapping = {'Corto': 0, 'Mediano': 2, 'Largo': 1}
    return mapping.get(rango, -1)

def de_encode_rango(rango):
    mapping = {0: 'Corto', 1: 'Largo', 2: 'Mediano'}
    return mapping.get(rango, 'Desconocido')

def de_encode_sentimiento(sentimiento):
    mapping = {0: 'Negativo', 1: 'Neutral', 2: 'Positivo'}
    return mapping.get(sentimiento, 'Desconocido')

def de_encode_pregunta(pregunta):
    return 'Sin Pregunta' if pregunta == 0 else 'Con Pregunta'

def cargar_modelo_lda(filename):
    return LdaModel.load(filename)

lda_model = cargar_modelo_lda("data/lda_model.gensim")

def cargar_diccionario(filename):
    return Dictionary.load(filename)

dictionary = cargar_diccionario("data/diccionario.gensim")

df = pd.read_csv('clusters.csv')

def modelo_clas(df):
    df['categoria_encoded'] = df['categoria'].apply(encode_categoria)
    X = df[["categoria_encoded","tipo_autor","sentiment","rangotitulo_encoded","rangosubtitulo_encoded","topics"]]
    y = df["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    tokens = text.split()

    stopwords_path = "data/spanish"
    if not os.path.exists(stopwords_path):
        st.error(f"No se encontró el archivo de stopwords en {stopwords_path}.")
        return tokens
    
    stop_words_df = pd.read_csv(stopwords_path, delimiter="\t", header=None)
    stop_words = set(stop_words_df[0].tolist())
    
    preprocessed_text = [token for token in tokens if token not in stop_words]
    
    return preprocessed_text

def predict_topic(text, lda_model, dictionary):
    tokens = preprocess_text(text)
    bow = dictionary.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0]

def predict_cluster(categoria, sentimiento, titulo, subtitulo, autor):
    categoria = encode_categoria(categoria)
    sentiment_value = {'Negativo': 0, 'Neutral': 1, 'Positivo': 2}[sentimiento]
    autor_value = {'Usuario': 0, 'Firma': 1}[autor]
    
    bins_titulo = [0, 13, 17, float('inf')]
    labels_titulo = ['Corto', 'Mediano', 'Largo']
    rango_titulo = pd.cut([len(titulo.split())], bins=bins_titulo, labels=labels_titulo)[0]
    rango_titulo = encode_rango(rango_titulo)

    bins_subtitulo = [0, 22, 30, float('inf')]
    labels_subtitulo = ['Corto', 'Mediano', 'Largo']
    rango_subtitulo = pd.cut([len(subtitulo.split())], bins=bins_subtitulo, labels=labels_subtitulo)[0]
    rango_subtitulo = encode_rango(rango_subtitulo)

    texto_completo = titulo + " " + subtitulo
    texto_completo = preprocess_text(texto_completo)
    texto_completo = " ".join(texto_completo)
    topic = predict_topic(texto_completo, lda_model, dictionary)
    
    input_data = pd.DataFrame({
        'categoria_encoded': [categoria],
        'tipo_autor': [autor_value],
        'sentiment': [sentiment_value],
        'rangotitulo_encoded': [rango_titulo],
        'rangosubtitulo_encoded': [rango_subtitulo],
        'topics': [topic]
    })

    cluster = modelo_clasificacion.predict(input_data)
    return cluster[0]

def evaluar_individuo(individuo, df_cluster, benchmark_cluster):
    sentimiento, tipo_autor, rangotitulo, rangosubtitulo, pregunta = individuo

    pageviews_mean = df_cluster[
        (df_cluster['sentiment'] == sentimiento) &
        (df_cluster['tipo_autor'] == tipo_autor) &
        (df_cluster['rangotitulo_encoded'] == rangotitulo) &
        (df_cluster['rangosubtitulo_encoded'] == rangosubtitulo) &
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


def mostrar_noticia(categoria, autor, titulo_rec, subtitulo_rec, tono_rec, pregunta_rec):
    tono_rec = tono_rec.lower()
    retorica = "con" if pregunta_rec.lower() != "sin pregunta" else "sin"
    st.markdown(
        f"""
        <div style="background-color: white; padding: 20px; position: relative; font-family: 'Libre Baskerville', serif;">
            <div style="border-bottom: 1px solid black; margin-bottom: 1px;">
                <p style='font-size: 18px; color: black; font-weight: bold;'>{categoria.upper()}</p>
            </div>
            <div style="padding-top: 1px; line-height: 1;">
                <h2 style='font-size: 24px; margin-bottom: 0;'>Titulo: {titulo_rec}</h2>
                <h3 style='font-size: 20px; color: grey; margin-top: 0;'>Subtitulo: {subtitulo_rec}</h3>
                <p style='font-size: 16px; color: white;'>espacio</p>
                <p style='font-size: 16px; color: black;'>Agregar un tono {tono_rec}, {retorica} pregunta retórica</p>
                <p style='font-size: 16px; color: white;'>espacio</p>
                <p style='font-size: 16px; color: grey;'>Escrita por: {autor}</p>
            </div>
            <img src='https://raw.githubusercontent.com/inesmurtagh/ln/main/images/portada.jpg' style='width: 200px; height: 200px; object-fit: cover; position: absolute; right: 20px; top: 70px;'>
        </div>
        """, unsafe_allow_html=True
    )

# Columna derecha: respuestas
with col2:
    st.write("")
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

                tono = de_encode_sentimiento(estrategia_recomendada[0][0]).upper()
                rangotitulo = de_encode_rango(estrategia_recomendada[0][2]).upper()
                rangosubtitulo = de_encode_rango(estrategia_recomendada[0][3]).upper()
                pregunta = de_encode_pregunta(estrategia_recomendada[0][4])

                st.markdown(f"Para este tipo de nota se recomienda un tono **{tono}**,")
                st.markdown(f"un título **{rangotitulo}**, con un subtítulo **{rangosubtitulo}**.")
                if pregunta == 'Sin Pregunta':
                    st.markdown("**No hace falta incluir una pregunta retórica.**")
                else:
                    st.markdown("**Hace falta incluir una pregunta retórica.**")
        
                # Mostrar la noticia formateada    
                st.write("")
                st.write("") 
                mostrar_noticia(categoria, autor, rangotitulo, rangosubtitulo, tono, pregunta)

            except ValueError as e:
                st.write(f"Error: {e}")
