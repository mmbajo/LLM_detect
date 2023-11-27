from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    LlamaForSequenceClassification,
    Trainer, 
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, roc_auc_score
from peft import PeftModel


import argparse

def infer(model_path, pretrain_path, data_path):
    def cleaning(dataset):
        dataset['text'] = dataset['text'].str.strip()
        dataset["text"] = dataset["text"].replace('\\n',' ')
        dataset["text"] = dataset["text"].str.split('ubject: ').str[-1].str.strip()
        dataset["text"] = dataset["text"].str.split('Zip').str[-1].str.strip()
        dataset["text"] = dataset["text"].str.split('ZIP').str[-1].str.strip()

        # dataset = dataset.rename(columns = {'generated':'label'})

        return dataset
    test_df = pd.read_csv(data_path, sep=',')
    test_df = cleaning(test_df)
    test_ds = Dataset.from_pandas(test_df)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess_function(examples, max_length=512):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)
    
    test_tokenized_ds = test_ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = LlamaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        quantization_config=bnb_config,
        device_map={"":0}
    )
    base_model.config.pretraining_tp = 1 # 1 is 7b
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    model = PeftModel.from_pretrained(base_model, pretrain_path)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    pred_output = trainer.predict(test_tokenized_ds)
    logits = pred_output.predictions
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    probs = sigmoid(logits[:, 1])
    sub = pd.DataFrame()
    sub['id'] = test_df['id']
    sub['generated'] = probs
    sub.to_csv('submission.csv', index=False)
    
    features = pd.DataFrame()
    features['id'] = test_df['id']
    features['logit_0'] = logits[:, 0]
    features['logit_1'] = logits[:, 1]
    features.to_csv('features.csv', index=False)
    
    # def encode_text(model, tokenizer, text):
    #     inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     print(outputs)
    #     return outputs.last_hidden_state.flatten(start_dim=1).detach().numpy()
    
    # embeddings_cache = []
    # for text in test_df['text']:
    #     embeddings_cache.append(encode_text(model, tokenizer, text))
    
    # embeddings_df = pd.DataFrame()
    # embeddings_df['id'] = test_df['id']
    # embeddings_df['embeddings'] = embeddings_cache
    # embeddings_df.to_csv("embeddings.csv", index=False)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions_hard = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions_hard),
            "roc_auc": roc_auc_score(labels, sigmoid(predictions[:, 1]))
        }
    
    if "label" in test_df.columns:
        eval_pred = (logits, test_df["label"].values)
        results = compute_metrics(eval_pred)
        print(results)
    
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Script to run inference.')

    # Add arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--pretrain_path', type=str, required=True, help='Path to the pretraining data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the inference data')

    # Parse the arguments
    args = parser.parse_args()

    # Extract values from args and run the infer function
    TARGET_MODEL = args.model_path
    PRETRAIN_PATH = args.pretrain_path
    DATA_PATH = args.data_path
    
    infer(TARGET_MODEL, PRETRAIN_PATH, DATA_PATH)