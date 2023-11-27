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

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import language_tool_python


tool = language_tool_python.LanguageTool('en-US')
train_data = "/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv"
test_data = "/kaggle/input/llm-detect-ai-generated-text/test_essays.csv"
out_dir = "/kaggle/working/exp006"

out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)


def correct_sentence(sentence):
    return tool.correct(sentence)

def denoise_text(text):
    # Use language_tool_python for spell checking
    corrected_text = tool.correct(text)

    return corrected_text

def correct_df(df):
    with ProcessPoolExecutor() as executor:
        df['text'] = list(executor.map(correct_sentence, df['text']))
        

def main():
    train = pd.read_csv(train_data, sep=",")
    # train1 = train[train.RDizzl3_seven == False].reset_index(drop=True)
    # train1 = train1[train1["label"] == 1].sample(8050, random_state=42)
    train = train[train.RDizzl3_seven == True].reset_index(drop=True)
    # train = pd.concat([train, train1])
    train["text"] = train["text"].str.replace('\n', '')

    cache = []
    for text in tqdm(train["text"]):
        cache.append(correct_sentence(text))
        
    train["text"] = cache

    print(train.value_counts("label"))
    print(train["text"].head(3))