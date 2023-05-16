import re
import pandas as pd
from pandas import DataFrame
from datetime import datetime

def replace_patterns(row, patterns):
    text = row['description']
    for pattern, replacement in patterns.items():
        regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        text = regex.sub(replacement, text)
    return text

patterns = {

    # Borra palabras clave dentro de la descripción
    "Sigue las actividades del presidente de México:": "",
    "\s+Más información:": "",
    "Sitio web:\s+": "",
    "YouTube:\s+": "",
    "Facebook:\s+": "",
    "Twitter:\s+": "",
    "Instagram:\s+": "",
    "Telegram:\s+": "",
    "Spotify:": "",
    "#EnVivo\s+": "",
    "#AMLO": "",
    
    # Borrar separadores
    "\|": "", 

    # Borra saltos de línea:
    "\n": " ", 

    # Borra los saltos a minutos en YouTube:
    "[0-9]?[0-9]?:[0-9][0-9]": "", 

    # Borra cualquier link de internet:
    "http(s)?://[a-zA-Z0-9./?=_-]+": "",

    # Elemina todos los # (hashtags)
    "#\w+": "",

    #Elimina todas las palabras con @ al principio
    "@\w+": "",

    # Eliminando links a redes sociales
    "\w+.com/\w+.\w+.\w+": "",

    #Eliminando la palabra lopezobrador
    "lopezobrador": "",

    # Borra el inicio la mención del presidente
    "Presidente AMLO.": "",

    # Borra la el inicio de la transmisión
    "Inicio de transmisión": "",
    "Comienza la conferencia de prensa del presidente Andrés Manuel López Obrador": "",

    # Los espacios dobles se reemplazan por un solo espacio
    "  +": " "

}

def remove_beginning(string):
    pattern = r'Conferencia de prensa (matutina|en vivo),?( desde (\b\w+\b,?\s)?(\b\w+\b\s)?\b\w+\b)\. \b(Lunes|Martes|Miércoles|Jueves|Viernes|Sábado|Domingo)\b [0-3]?[0-9] de \b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b 20[0-9][0-9]\.?'
    replace_with = ''
    return re.sub(pattern, replace_with, string)

def obtener_dataframe(json: dict):
    data = []
    for item in json:
        snippet = item['snippet']
        published_at = snippet['publishedAt']
        title = snippet['title']
        description = snippet['description']
        data.append([published_at, title, description])
    dataframe = pd.DataFrame(data, columns = ['published_at', 'title', 'description'])
    dataframe['description'] = dataframe.apply(replace_patterns, patterns = patterns, axis = 1).apply(remove_beginning)

    clean_dates = []
    for dates in dataframe['published_at']:
        clean_dates.append( datetime.strptime(dates[:10], '%Y-%m-%d') )
    
    dataframe['clean_dates'] = clean_dates
    dataframe = dataframe.drop('published_at', axis = 1)
    dataframe['clean_dates'] = pd.to_datetime(dataframe['clean_dates'])
    dataframe.set_index('clean_dates', inplace = True)
    weekly_ts = dataframe.resample('W').sum()
    return weekly_ts
