import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = []

    for word in tokens:
        if word not in stop_words and word not in string.punctuation:
            filtered_tokens.append(word)

    return " ".join(filtered_tokens)