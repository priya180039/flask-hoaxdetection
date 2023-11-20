import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle
from gensim.models import word2vec

def preprocess(text: str):
    news = []

    for sent in tqdm(text):
        
        #case folding
        sent = sent.lower()

        #remove html content
        news_text = BeautifulSoup(sent).get_text()
        
        #remove non-alphabetic characters
        news_text = re.sub("[^a-zA-Z]"," ", news_text)
    
        #tokenize the sentences
        words = word_tokenize(news_text)
    
        #stop words removal
        omit_words = set(stopwords.words('indonesian'))
        words = [x for x in words if x not in omit_words]
        
        #lemmatize each word to its lemma
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        hasil = [stemmer.stem(i) for i in words]
        news.append(hasil)

    return(news)

def rearrange(text):
    r = ''
    for sent in tqdm(text):
        for i in range(len(sent)):
            if i==len(sent)-1:
                r += sent[i]
            else:
                r += sent[i]
                r += ' '

    return r

def w2v():
    model = pickle.load(open('static/model/3word2vec-100-10.pkl','rb'))
    return model

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.wv.key_to_index]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return []
    
def avgFeatureVector(sentences, model):
    overallFeatureVector = []
    for sentence in tqdm(sentences):
        overallFeatureVector.append(get_mean_vector(model, sentence)) 
    return overallFeatureVector

def rfc():
    model_rf = pickle.load(open('static/model/bestmodelrf.pkl','rb'))
    return model_rf

def identify(model_rf, vector):
    result = model_rf.predict(vector)
    return result

