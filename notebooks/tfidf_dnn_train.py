import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import StratifiedKFold

from joblib import dump
from pathlib import Path
import unicodedata
import re
import os

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import language_tool_python

DATA_PATH = "/workspaces/LLM_detect/data/daigt-proper-train-dataset/train_v2_drcat_02.csv"
OUT_DIR = Path("../_OUTPUT/exp_020_tfidf_dnn")
OUT_DIR.mkdir(exist_ok=True, parents=True)

tool = language_tool_python.LanguageTool('en-US')

def correct_sentence(sentence):
    return tool.correct(sentence)

def denoise_text(text):
    # Use language_tool_python for spell checking
    corrected_text = tool.correct(text)

    return corrected_text

def correct_df(df):
    with ProcessPoolExecutor() as executor:
        df['text'] = list(executor.map(correct_sentence, df['text']))
        
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

def get_chars_to_remove():
    train = pd.read_csv(DATA_PATH, sep=",")
    not_persuade_df = train[train['source'] != 'persuade_corpus']
    persuade_df = train[train['source'] == 'persuade_corpus']
    sampled_persuade_df = persuade_df.sample(n=8000, random_state=42)

    all_human = set(list(''.join(sampled_persuade_df.text.to_list())))
    other = set(list(''.join(not_persuade_df.text.to_list())))
    chars_to_remove = ''.join([x for x in other if x not in all_human])
    
    translation_table = str.maketrans('', '', chars_to_remove)
    return translation_table

TRANSLATION_TABLE = get_chars_to_remove()
def remove_chars(s):
    return s.translate(TRANSLATION_TABLE)

def preprocess(text):
    text = remove_chars(text)
    text = correct_sentence(text)
    text = clean_essay(text)
    return text

def get_train_data():
    if os.path.isfile(str(OUT_DIR / "processed.csv")):
        train = pd.read_csv(OUT_DIR / "processed.csv", sep=",")
    else:
        train = pd.read_csv(DATA_PATH, sep=",")
        train["text"] = train.text.apply(preprocess)
        train.to_csv(OUT_DIR / "processed.csv", index=False)
    return train
        
def get_vectorizers(train):
    v_3_5 = TextVectorization(max_tokens=50000, output_mode="tf-idf", ngrams=(3, 5))
    v_3_5.adapt(train["text"])
    return v_3_5

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
    
def inference(model, X_val):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    y_pred = sigmoid(model.predict(X_val, verbose=2).reshape(-1))
    return y_pred

def evaluate_model(model, X_val, y_val):
    y_pred = inference(model, X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(f"AUC for {model}: {auc}")
    return {
        "model": model,
        "auc": auc
    }

def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
        dataset = dataset.shuffle(batch_size * 4) 
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset

def get_model(vectorizer):
    inputs = keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inputs)
    x = layers.Dense(32, activation="swish")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="swish")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1)(x)
    model = keras.Model(inputs, output, name="model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(4e-4), 
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=0.2,
        ), 
        metrics=[
            "accuracy", 
            keras.metrics.AUC(name="auc"),
            fbeta
        ]
    )
    return model

def train_models(X_train, y_train, X_val, y_val, vectorizer, fold):
    model_path = OUT_DIR / f"model_{fold}.tf"
    checkpoints = []
    model = get_model(vectorizer)
    train_ds = make_dataset(X_train, y_train, 128, "train")
    valid_ds = make_dataset(X_val, y_val, 128, "valid")
    model.fit(
        train_ds, 
        epochs=30, 
        validation_data=valid_ds,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(model_path, monitor="val_auc", mode="max", save_best_only=True)
        ]
    )
    checkpoints.append(evaluate_model(model, X_val, y_val))
    return checkpoints

def main():
    train = get_train_data()
    vectorizer = get_vectorizers(train)
    kfold = StratifiedKFold(5, shuffle=True, random_state=42)
    models = []
    for fold, (train_index, valid_index) in enumerate(kfold.split(train, train["label"])):
        X_train = train.iloc[train_index]["text"]
        y_train = train.iloc[train_index]["label"]
        X_val = train.iloc[valid_index]["text"]
        y_val = train.iloc[valid_index]["label"]
        models += train_models(X_train, y_train, X_val, y_val, vectorizer, fold)
        
    result = np.mean([model["auc"] for model in models])
    print(result)
    

if __name__ == "__main__":
    main() 