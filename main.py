import json
from os import path

from youtube_api import obtener_datos_youtube
from utilidades import obtener_dataframe
from procesamiento import pre
from procesamiento import principal

def main():
    print('iniciando ejecución:')
    if not path.exists('datos_youtube.json'):
        print('no se encontró archivo, descargando información...')
        datos_youtube = obtener_datos_youtube()
        with open("datos_youtube.json", "w") as youtube_json:
            json.dump(datos_youtube, youtube_json)
    else:
        print('cargando información de archivo...')
        with open('datos_youtube.json') as youtube_json:
            datos_youtube = json.load(youtube_json)
    print('generando dataframe de la información obtenida...')
    dataframe_sin_procesar = obtener_dataframe(datos_youtube)
    if not path.exists('mañaneras.csv'):
        print('guardando corpus [mañaneras.csv]...')
        dataframe_sin_procesar['description'].to_csv('mañaneras.csv')
    else:
        print('no se generó un nuevo corpus...')
    print('aplicando preprocesamiento a dataframe...')
    dataframe_preprocesada = pre(dataframe_sin_procesar)
    print('aplicando procesamiento principal a dataframe...')
    resultados = principal(dataframe_preprocesada)
    print('guardando procesamiento [resultados.json]...')
    print(resultados)
    with open("resultados.json", "w") as resultados_json:
        json.dump(resultados, resultados_json)
    print('ejecutado correctamente...')
  
if __name__=="__main__":
    main()