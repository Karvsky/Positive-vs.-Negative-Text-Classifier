import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize   
from nltk.corpus import stopwords         
from nltk.stem import WordNetLemmatizer  
from nltk.classify import NaiveBayesClassifier
from nltk.util import ngrams
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))
important_words = {'not', 'no', 'nor', 'but', 'very', 'too', 'more', 'most', 'against'}
stop_words = stop_words - important_words
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
    tokens = word_tokenize(text.lower())

    clean_words = []

    for word in tokens:
        if word not in stop_words and word.isalnum():
            lematized_word = lemmatizer.lemmatize(word)
            clean_words.append(lematized_word)

    list_of_bigrams = list(ngrams(clean_words, 2))

    features = format_features(clean_words)
    features.update({bg: True for bg in list_of_bigrams}) 

    if features:
        dataset.append((features, label))

train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=70)
data_after_training = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.accuracy(data_after_training, test_set)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

new_opinion = input("Type a text: ")
new_tokens = word_tokenize(new_opinion.lower())

new_clean = []

for w in new_tokens:
    if w not in stop_words:
        if w.isalnum():
            lematized_word = lemmatizer.lemmatize(w)
            new_clean.append(lematized_word)

new_bigrams = list(ngrams(new_clean, 2))
ready_to_test = format_features(new_clean)
ready_to_test.update({bg: True for bg in new_bigrams})

result = data_after_training.classify(ready_to_test)
print(f"Result: {result}")
