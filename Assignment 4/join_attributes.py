import pandas as pd 

# import datasets
df_original = pd.read_csv("dataset_newattributes.csv", encoding='utf8')
df_lda = pd.read_csv("nlp part/dataset_lda.csv", encoding='utf8')
df_features = pd.read_csv("nlp part/dataset_nlp_features.csv", encoding='utf8')

# delete unnecesary columns
del df_original['isyuc']
del df_original['Agno']
del df_original['Minutos']
del df_original['operador']
del df_original['clave_cog']
del df_original['clave_uc']
del df_original['caracter_nacional']
del df_original['caracter_internacional']
del df_original['forma_procedimiento_presencial']
del df_original['tipo_contratacion_adquisiciones']
del df_original['NP1_AA']
del df_original['articulo_41']
del df_original['fraccion_XII']
del df_original['parrafo_primero']
del df_original['origen_correo_gobierno']
del df_original['Mes_3']
del df_original['NP4_2020']

dict_i = {
    
'aguascalientes': 'a'
, 'baja california': 'a'
, 'campeche': 'a'
, 'coahuila': 'a'
, 'colima': 'a'
, 'ciudad de mexico': 'a'
, 'durango': 'a'
, 'mexico': 'a'
, 'guanajuato': 'a'
, 'nayarit': 'a'
, 'sinaloa': 'a'
, 'tamaulipas': 'a'
, 'baja california sur': 'b'
, 'chihuahua': 'b'
, 'guerrero': 'b'
, 'hidalgo': 'b'
, 'jalisco': 'b'
, 'michoacan': 'b'
, 'morelos': 'b'
, 'nuevo leon': 'b'
, 'puebla': 'b'
, 'queretaro': 'b'
, 'san luis potosi': 'b'
, 'sonora': 'b'
, 'tabasco': 'b'
, 'veracruz': 'b'
, 'chiapas': 'c'
, 'oaxaca': 'c'
, 'quintana roo': 'c'
, 'tlaxcala': 'c'
, 'yucatan': 'c'
, 'zacatecas': 'c'
    
}

df_original = df_original.replace({"entidades": dict_i})

# join basic features
df_original['descripcion_anuncio_word_count'] = df_features['word_count']
df_original['descripcion_anuncio_char_count_w_spaces'] = df_features['char_count_w_spaces']
df_original['descripcion_anuncio_char_count_wo_spaces'] = df_features['char_count_wo_spaces']
df_original['descripcion_anuncio_avg_word'] = df_features['avg_word']
df_original['descripcion_anuncio_stopwords'] = df_features['stopwords']
df_original['descripcion_anuncio_numerics'] = df_features['numerics']
df_original['descripcion_anuncio_punctuation'] = df_features['punctuation']
df_original['descripcion_anuncio_upper'] = df_features['upper']

# join lda topic
df_original['lda_topic'] = df_lda['Dominant_Topic']

# save dataset pattern ready
df_original.to_csv("dataset_ready.csv", index=False, encoding='utf8')