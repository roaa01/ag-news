import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import re
import string
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.src.initializers import Orthogonal
import keras.initializers

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize preprocessor components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Patch for Orthogonal initializer (if needed by model)
def custom_get(identifier):
    if isinstance(identifier, dict) and identifier.get('class_name') == 'Orthogonal':
        return Orthogonal(**identifier.get('config', {}))
    return keras.initializers.get(identifier)
keras.initializers.get = custom_get

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
model = load_model("my_model.keras")

# Streamlit page config
st.set_page_config(page_title="Article Classifier", page_icon="📰", layout="wide")

# Preprocessing function
def preprocess_text(text):
    def remove_html_tags(t): return re.compile('<.*?>').sub('', t)
    def remove_url(t): return re.compile(r'https?://\S+|www\.\S+').sub('', t)
    def remove_punc(t): return t.translate(str.maketrans('', '', string.punctuation))
    def chat_conversion(t):
        chat_words = {"AFAIK": "As Far As I Know", "AFK": "Away From Keyboard", "ASAP": "As Soon As Possible"}
        return " ".join(chat_words.get(word.upper(), word) for word in t.split())

    text = text.lower()
    text = remove_html_tags(text)
    text = remove_url(text)
    text = remove_punc(text)
    text = chat_conversion(text)
    text = " ".join('' if word in stop_words else word for word in text.split())
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

# Prediction function
def predict(text):
    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)[0]
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return labels[predicted_class]

# Sidebar menu
with st.sidebar:
    choose = option_menu(None, ["Home", "About", "Graph", "Contact"], 
                         icons=["house", "info-circle", "bar-chart", "envelope"],
                         menu_icon="cast", default_index=0, orientation="vertical")

# Pages
if choose == "Home":
    st.title("Article Classifier")
    st.write("Classify news articles into: World, Sports, Business, or Sci/Tech.")
    
    input_text = st.text_area("Enter the article text below:", height=200)
    if st.button("Classify"):
        if input_text:
            result = predict(input_text)
            st.success(f"he article is classified as: **{result}**")
        else:
            st.error(" Please enter some text to classify.")

elif choose == "About":
    st.title(" About")
    st.write("This app uses a deep learning model trained on news articles to classify them into categories.")
    st.write("Developed by Roaa Hazem.")

elif choose == "Contact":
    st.title("Contact")
    st.write("Email: roaa.hazem.ismail@gmail.com")
    st.write(" LinkedIn: [Roaa Ismail](https://www.linkedin.com/in/roaa-ismail-89811525b/)")
    st.write("GitHub: [roaa01](https://github.com/roaa01)")

elif choose == "Graph":
    st.title(" Graph")
    st.write("Accuracy Vs Loss Graph:")
    st.write("Graphs without K-fold Cross Validation:")
    st.image("download.png", caption="Example Graph", use_column_width=True)
    st.image("image.png", caption="Example Graph", use_column_width=True)
    st.write("Graphs  K-fold Cross Validation:")
    st.image("K-fold.png", caption="Example Graph", use_column_width=True)
    st.image("K-fold-loss.png", caption="Example Graph", use_column_width=True)