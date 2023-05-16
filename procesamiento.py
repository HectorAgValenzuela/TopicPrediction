import re
import spacy
import numpy as np

from pandas import DataFrame
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from gensim import corpora
from gensim.models import LdaModel

def pre(sin_procesar: DataFrame):
    def quitar_stopwords(text):
        sw_es = set(stopwords.words('spanish'))
        text = ' '.join([word for word in text.split() if word.lower() not in sw_es])
        return text

    def lematizar_texto(text):
        nlp = spacy.load("es_core_news_sm")
        doc = nlp(text)
        lematizar_texto = " ".join([token.lemma_ for token in doc])
        return lematizar_texto

    def limpiar_caracteres(text):
        text = text.lower()
        text = re.sub('[áäàâ]', 'a', text)
        text = re.sub('[éëèê]', 'e', text)
        text = re.sub('[íïìî]', 'i', text)
        text = re.sub('[óöòô]', 'o', text)
        text = re.sub('[úüùû]', 'u', text)
        return re.findall(r'[a-zñ]+', text)
    
    dataframe = sin_procesar[['description']].copy()
    dataframe['description'] = dataframe['description'].apply(quitar_stopwords)
    dataframe['description'] = dataframe['description'].apply(lematizar_texto)
    dataframe['description'] = dataframe['description'].apply(limpiar_caracteres).apply(lambda x: " ".join(x))
    return dataframe

def principal(preprocesada: DataFrame):
    result = {}
    #SECCION I
    # Creamos una instancia que hará la vectorización TF-IDF
    vectorizador = TfidfVectorizer()

    # Tokenizamos como Penn Treebank
    tokenizer = TreebankWordTokenizer()
    vectorizador.set_params(tokenizer=tokenizer.tokenize)

    # Incluimos 1-grams y 2-grams
    vectorizador.set_params(ngram_range=(1, 3))

    # Ignoramos términos que aparecen en más del 70% de los documentos
    vectorizador.set_params(max_df=0.7) 

    # Solo mantenemos los términos que aparecen en al menos 2 documentos
    vectorizador.set_params(min_df=2)

    # Lo aplicamos
    vector = vectorizador.fit_transform(preprocesada['description']) 

    #SECCION II
    # Descomposicion por valores singulares (SVD)
    # Creamos la instancia 
    scaler = StandardScaler(with_mean=False)
    # Calculamos el promedio y la desviacion estandar
    scaler.fit(vector)
    # Re escalamos
    vector_escalado_estandar = scaler.transform(vector)

    # SECCION III
    # Aplicamos TruncatedSVD
    truncater_uno = TruncatedSVD(n_components = preprocesada.shape[0], random_state = 2)
    # Ajustamos la transfromacion con nuestros datos
    truncater_uno.fit(vector_escalado_estandar)
    # Reducimos la dimensionalidad
    vector_escalado_truncado_uno = truncater_uno.transform(vector_escalado_estandar)
    #'Variancia explicada (%)'
    variancia_explicada_uno = np.cumsum(truncater_uno.explained_variance_ratio_ * 100)

    # result['vector_escalado_truncado_uno'] = vector_escalado_truncado_uno;
    # result['variancia_explicada_uno'] = variancia_explicada_uno;

    truncater_dos = TruncatedSVD(n_components = 2, random_state = 2)
    truncater_dos.fit(vector_escalado_estandar)
    vector_escalado_truncado_dos = truncater_dos.transform(vector_escalado_estandar)
    variancia_explicada_dos = round(sum(truncater_dos.explained_variance_ratio_ * 100), 3)
    
    # result['vector_escalado_truncado_dos'] = vector_escalado_truncado_dos;
    # result['variancia_explicada_dos'] = variancia_explicada_dos;

    truncater_tres = TruncatedSVD(n_components = 3, random_state = 2)
    truncater_tres.fit(vector_escalado_estandar)
    vector_escalado_truncado_tres = truncater_tres.transform(vector_escalado_estandar)
    variancia_explicada_tres = round(sum(truncater_tres.explained_variance_ratio_ * 100), 3)
    # result['vector_escalado_truncado_tres'] = vector_escalado_truncado_tres;
    # result['variancia_explicada_tres'] = variancia_explicada_tres;

    # SECCION IV
    sumExpVariance = 0
    for i in range(truncater_uno.explained_variance_ratio_.shape[0]):
        if sumExpVariance > 0.95:
            print(f"{i} componentes explican {round(sumExpVariance * 100, 3)} de la variancia")
            break
        sumExpVariance += truncater_uno.explained_variance_ratio_[i]

    # SECCION V
    scaler = MaxAbsScaler()
    scaler.fit(vector)
    vector_escalado_max = scaler.transform(vector)
    # result['vector_escalado_max'] = vector_escalado_max;

    # Obtener el vocabulario de términos
    feature_names = vectorizador.get_feature_names_out()

    # Obtener las frecuencias de términos en el primer documento
    doc_freqs = vector[0].toarray()[0]

    # Crear una lista de tuplas que contienen el término y su frecuencia en el primer documento
    doc_term_freqs = [(feature_names[i], doc_freqs[i]) for i in range(len(feature_names))]

    # Ordenar la lista de tuplas por la frecuencia descendente
    doc_term_freqs_sorted = sorted(doc_term_freqs, key=lambda x: x[1], reverse=True)

    # Obtener las 10 palabras clave más importantes para el primer documento
    doc_keywords = [term for term, freq in doc_term_freqs_sorted[:30]]

    result['doc_keywords'] = doc_keywords;

    # Tokenizar los documentos utilizando el tokenizador de Penn Treebank
    tokenizer = TreebankWordTokenizer()
    documentos = []
    for documento in preprocesada['description'].tolist():
        tokens = tokenizer.tokenize(documento)
        documentos.append(tokens)

    # Crear un diccionario de términos a partir de los documentos
    dictionary = corpora.Dictionary(documentos)

    # Crear una representación de la bolsa de palabras de los documentos
    corpus = [dictionary.doc2bow(documento) for documento in documentos]

    # Entrenar un modelo LDA con 10 tópicos
    model_lda = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

    keywords = []
    # Imprimir los tópicos
    for idx, topic in model_lda.print_topics(num_topics=10):
        keywords.append([idx + 1, topic])
    result['keywords'] = keywords
    return result