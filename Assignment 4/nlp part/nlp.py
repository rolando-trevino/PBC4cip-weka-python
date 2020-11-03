import pandas as  pd
import numpy as np
import spacy
import nltk
import re
import unicodedata
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer

nlp = spacy.load("es_core_news_lg", disable=['ner', 'parser'])
SpacyStopwords = nlp.Defaults.stop_words

# StopwordsOriginal = SpacyStopwords.words("spanish") # stopwords de spacy o nltk cuales son mejor?
StopwordsSP = []

for Word in SpacyStopwords:
    Stop = Word.lower()
    Stop_NFKD = unicodedata.normalize('NFKD', Stop)
    Stop = u"".join([c for c in Stop_NFKD if not unicodedata.combining(c)])
    StopwordsSP.append(Stop)

preposiciones = ['a', 'durante', 'según', 'ante', 'en', 'sin', 'bajo', 'entre', 'so', 'cabe', 'hacia', 'sobre', 'con', 'hasta', 'tras', 'contra', 'mediante', 'versus', 'de', 'para', 'via', 'desde', 'por', 'y', 'o']

StopwordsSP = StopwordsSP + preposiciones

# df = pd.read_excel("dataset.xlsx")
df = pd.read_csv("dataset_csv.csv", encoding='utf8')

df['nlp'] = df['Descripción del Anuncio'].str.strip()
df['nlp'] = df['nlp'].str.lower()
df['nlp'] = df['nlp'].str.normalize('NFKD')\
       .str.encode('ascii', errors='ignore')\
       .str.decode('utf-8')

df['nlp'] = df['nlp'].replace(np.NaN, '')

lista_lemma = []
with tqdm(total = len(df), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar:
    for doc in nlp.pipe(iter(df['nlp'])):
        lista_sent = []
        for token in doc:
            if (token.text not in StopwordsSP) and (re.match('([a-zA-Z]+)',token.text) != None):
                lista_sent.append(token.lemma_)
        lista_lemma.append(' '.join(lista_sent))
        pbar.update(1)

df['nlp_lemma'] = lista_lemma
df['nlp_lemma'] = df['nlp_lemma'].replace(np.NaN, '')

nlp_col = df['nlp_lemma']

# Freq Visualization
# Load the text data
corpus = nlp_col

vectorizer = CountVectorizer(token_pattern='\S+')
docs       = vectorizer.fit_transform(text for text in corpus)
features   = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(
    features=features, size=(1080, 720)
)
visualizer.fit(docs)
visualizer.show()

