import streamlit as st
import numpy as np
import pickle
import re
import os
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------
# Download Model Files from Drive
# -------------------------------

FOLDER_URL = "https://drive.google.com/drive/folders/1I3xzpMNVm1ryE64WO2FroW37Pb8d3p-c"

# Download only if not present
if not os.path.exists("news_model.keras") or not os.path.exists("tokenizer.pkl"):

    st.write("Downloading model files... ⏳")

    gdown.download_folder(
        FOLDER_URL,
        quiet=False,
        use_cookies=False
    )

    st.write("Download complete ✅")


# -------------------------------
# Load Model
# -------------------------------

model = load_model("news_model.keras")

tokenizer = pickle.load(open("tokenizer.pkl","rb"))

max_len = 100


# -------------------------------
# Text Cleaning Function
# -------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    text = re.sub(r'\s+',' ',text)

    return text


# -------------------------------
# Labels
# -------------------------------

labels = {

0:"🌍 World News",
1:"🏏 Sports",
2:"💰 Business",
3:"💻 Technology"

}


# -------------------------------
# Prediction Function
# -------------------------------

def predict_news(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    pad = pad_sequences(seq,maxlen=max_len)

    pred = model.predict(pad,verbose=0)

    label = np.argmax(pred)

    return labels[label]


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📰 News Category Classifier")

st.write("RNN + NLP based News Classification")

user_input = st.text_area("Enter News Text")


if st.button("Predict Category"):

    if user_input.strip() != "":

        result = predict_news(user_input)

        st.success("Predicted Category: " + result)

    else:

        st.warning("Please enter some text")
