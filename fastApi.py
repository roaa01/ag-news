from fastapi import FastAPI,HTTPException
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re, string, pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences  # Corrected import
from keras.src.initializers import Orthogonal
import keras.initializers
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Patch for Orthogonal initializer
def custom_get(identifier):
    if isinstance(identifier, dict) and identifier.get('class_name') == 'Orthogonal':
        return Orthogonal(**identifier.get('config', {}))
    return keras.initializers.get(identifier)

keras.initializers.get = custom_get

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
model = load_model("my_model.keras")  # Ensure this file exists

import re
import string

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

def remove_html_tags(text):
    return re.compile('<.*?>').sub(r'', text)

def remove_url(text):
    return re.compile(r'https?://\S+|www\.\S+').sub(r'', text)

def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def chat_conversion(text):
    chat_words = {"AFAIK": "As Far As I Know", "AFK": "Away From Keyboard", "ASAP": "As Soon As Possible"}
    return " ".join(chat_words.get(i.upper(), i) for i in text.split())

def remove_stopwords(text):
    stopword = stopwords.words('english')
    return " ".join('' if word in stopword else word for word in text.split())

def lem_words(word_list):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in word_list]

def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_url(text)
    text = remove_punc(text)
    text = chat_conversion(text)
    text = remove_stopwords(text)
    text = word_tokenize(text)
    text = lem_words(text)
    return " ".join(text)
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    try:
        clean = preprocess_text(input.text)

        # Convert to sequence
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=200)

        # Predict
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map to label
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        predicted_label = labels[predicted_class]

        return {"prediction": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
#curl -X POST http://127.0.0.1:8000/predict 
