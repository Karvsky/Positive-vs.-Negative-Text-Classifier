import nltk
from nltk.tokenize import word_tokenize   
from nltk.corpus import stopwords         
from nltk.stem import PorterStemmer       
from nltk.stem import WordNetLemmatizer  
from nltk.util import ngrams
import random

nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('wordnet')
nltk.download('omw-1.4')  
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def format_features(tokens):
    return {word: True for word in tokens}

dataset = []

with open("./Dataset/negative.txt", "r", encoding="utf-8") as file_negative:
    for line in file_negative:
        tokens = word_tokenize(line.lower())
        
        clean_tokens = [
            lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in stop_words and word.isalnum()
        ]

        trigrams = list(ngrams(clean_tokens, 3))

        features = format_features(trigrams)

        if features:
            dataset.append((features, "NEGATIVE"))

with open("./Dataset/pos.txt", "r", encoding="utf-8") as file_positive:
     for line in file_positive:
        tokens = word_tokenize(line.lower())
        
        clean_tokens = [
            lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in stop_words and word.isalnum()
        ]

        trigrams = list(ngrams(clean_tokens, 3))

        features = format_features(trigrams)

        if features:
            dataset.append((features, "POSITIVE"))

random.shuffle(dataset)
