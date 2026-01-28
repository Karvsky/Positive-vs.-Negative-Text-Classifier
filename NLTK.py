import nltk
from nltk.tokenize import word_tokenize   
from nltk.corpus import stopwords         
from nltk.stem import PorterStemmer       
from nltk.stem import WordNetLemmatizer  
from nltk.classify import NaiveBayesClassifier
from nltk.util import ngrams
from sklearn.model_selection import train_test_split

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

        features = format_features(clean_tokens)

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

        features = format_features(clean_tokens)

        if features:
            dataset.append((features, "POSITIVE"))

train_set, test_set = train_test_split(dataset, test_size=0.1, random_state=70)
data_after_training = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.accuracy(data_after_training, test_set)
print(f"\nDokładność modelu: {accuracy * 100:.2f}%")

new_opinion = "I love this film"
new_opinion_tokenize = word_tokenize(new_opinion.lower())
clean_tokens = [
    lemmatizer.lemmatize(word) 
    for word in new_opinion_tokenize          
    if word not in stop_words and word.isalnum()
]
ready_opinion_to_test = format_features(clean_tokens)
result = data_after_training.classify(ready_opinion_to_test)
print(f"Tekst: {new_opinion}")
print(f"Wynik klasyfikacji: {result}")
