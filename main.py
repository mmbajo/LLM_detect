import streamlit as st
import pandas as pd
import numpy as np
import random
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import re
import unicodedata

import language_tool_python

TOOL = language_tool_python.LanguageToolPublicAPI('en-US')
MODEL_PATH = Path("./models")
@st.cache_resource
def init_model():
    def fbeta(y_true, y_pred, beta = 1.0):
        y_true_count = tf.reduce_sum(y_true)
        ctp = tf.reduce_sum(y_true * y_pred)
        cfp = tf.reduce_sum((1.0 - y_true) * y_pred)
        beta_squared = beta * beta
        c_precision = tf.where(ctp + cfp == 0.0, 0.0, ctp / (ctp + cfp))
        c_recall =  tf.where(y_true_count == 0.0, 0.0, ctp / y_true_count)
        return tf.where(
            c_precision + c_recall == 0, 
            0.0, 
            tf.divide((1.0 + beta_squared) * (c_precision * c_recall),  (beta_squared * c_precision + c_recall))
        )
        
    models = []
    for fold in range(5):
        model = load_model(
            MODEL_PATH / f"model_{fold}.tf"
        )
        models.append(model)
        
    return models

def correct_sentence(sentence):
    return TOOL.correct(sentence)

def how_many_typos(text):    
    return len(TOOL.check(text))

def clean_essay(text):
    m = re.search(r"\n\n(References|Work Cited|Works Cited)", text, flags=re.IGNORECASE)
    if m:
        text = text[: m.start()]
        
    text = text.replace('\n', ' ')
    text = re.sub(r"(MONTH_DAY_YEAR|TEACHER_NAME|STUDENT_NAME|PROPER_NAME|SCHOOL_NAME|LOCATION_NAME|Generic_Name|\[Last Name\])", "[MASK]", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("''", '"')
    return text.strip()

@st.cache_data
def preprocess(text):
    num_typos = how_many_typos(text)
    text = correct_sentence(text)
    text = clean_essay(text)
    return text, num_typos

MODELS = init_model()

@st.cache_data
def infer(text):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def inference(model, X_val):
        if "keras" in str(type(model)):
            y_pred = model.predict(X_val, verbose=2).reshape(-1)
        else:
            y_pred = model.predict_proba(X_val)[:, 1].reshape(-1)
        return y_pred
    
    # corrected_text = correct_sentence(text)
    probability = np.mean([sigmoid(inference(model, [text])) for model in MODELS], axis=0)[0]
    return probability

def main():
    st.title("LLM Text Analysis Tool")
    with st.form(key='text_form'):
        text = st.text_area("Enter the text to analyze:", height=150)
        submit_button = st.form_submit_button(label='Analyze')

    if submit_button and text:
        with st.spinner('Analyzing...'):
            clean_text, num_typos = preprocess(text)
            probability = infer(clean_text)
            st.success("Analysis Complete!")

            # Display cleaned text
            st.subheader("Cleaned Text:")
            st.text_area("Here's the cleaned version of your text:", value=clean_text, height=150, key='clean_text')

            # Display number of typos
            st.metric(label="Number of Typos Detected", value=num_typos)

            # Display probability
            st.metric(label="Probability of LLM Generation", value=f"{probability * 100:.2f}%")
            if probability > 0.5:
                st.warning("This text is likely generated by a LLM.")
            else:
                st.info("This text is unlikely generated by a LLM.")

if __name__ == "__main__":
    main()