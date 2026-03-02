import os
import gdown

# Download model folder automatically

if not os.path.exists("news_model.keras"):

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1I3xzpMNVm1ryE64WO2FroW37Pb8d3p-c",
        quiet=False,
        use_cookies=False
    )


import streamlit as st
import numpy as np
import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load Model
model = load_model("news_model.keras")

tokenizer = pickle.load(open("tokenizer.pkl","rb"))

max_len = 100


# Clean Text
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    text = re.sub(r'\s+',' ',text)

    return text


labels = {

0:"🌍 World News",
1:"🏏 Sports",
2:"💰 Business",
3:"💻 Technology"

}


def predict_news(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    pad = pad_sequences(seq,maxlen=max_len)

    pred = model.predict(pad)

    label = np.argmax(pred)

    return labels[label]


# UI
st.title("📰 News Category Classifier")
# st.write("Classify news into categories using RNN + NLP")

user_input = st.text_area("Enter News Text")

if st.button("Predict Category"):

    if user_input != "":

        result = predict_news(user_input)

        st.success("Category: " + result)

    else:


        st.warning("Please enter news text")
