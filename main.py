import pandas as pd
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize   
from nltk.corpus import stopwords         
from nltk.stem import WordNetLemmatizer  
from nltk.classify import NaiveBayesClassifier
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from text_to_string import text

def format_features(tokens):
    return {word: True for word in tokens}


def result(text_from_photo):
    
    stop_words = set(stopwords.words('english'))
    important_words = {'not', 'no', 'nor', 'but', 'very', 'too', 'more', 'most', 'against'}
    stop_words = stop_words - important_words
    lemmatizer = WordNetLemmatizer()

    model_path = "sentiment_model.pickle"

    with open(model_path, 'rb') as f:
        model_after_training = pickle.load(f)

    new_opinion = text_from_photo
    new_tokens = word_tokenize(text_from_photo.lower())

    new_clean = []

    for w in new_tokens:
        if w not in stop_words:
            if w.isalnum():
                lematized_word = lemmatizer.lemmatize(w)
                new_clean.append(lematized_word)

    new_bigrams = list(ngrams(new_clean, 2))
    ready_to_test = format_features(new_clean)
    ready_to_test.update({bg: True for bg in new_bigrams})

    result = model_after_training.classify(ready_to_test)
    print(text_from_photo, "\n")
    print(f"Result: {result}")

if __name__ == '__main__':
    found_texts_list = text()

    for single_sentence in found_texts_list:
        result(single_sentence)
