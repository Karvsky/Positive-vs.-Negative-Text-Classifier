import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize   
from nltk.corpus import stopwords         
from nltk.stem import PorterStemmer       
from nltk.stem import WordNetLemmatizer  
from nltk.classify import NaiveBayesClassifier
from nltk.util import ngrams
from sklearn.model_selection import train_test_split

#nltk.download('punkt')
#nltk.download('punkt_tab') 
#nltk.download('wordnet')
#nltk.download('omw-1.4')  
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def format_features(tokens):
    return {word: True for word in tokens}

dataset = []

df = pd.read_csv('./Dataset2/file.csv')


def remove_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


df['review'] = df['review'].apply(remove_html)
for index, row in df.iterrows():
    text = row['review']
    label = row['sentiment'].upper()
print(label)









print(label)

