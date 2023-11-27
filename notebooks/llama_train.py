from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import os
os.environ["WANDB_PROJECT"] = "llm_detect"

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    LlamaForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from peft import (
    PeftModel,
    get_peft_model, 
    LoraConfig, 
    TaskType
)
from shutil import rmtree

# Constants
TARGET_MODEL = "mistralai/Mistral-7B-v0.1"
RUN_NAME = "llama_128batch_eval"
OUTPUT_DIR = Path("../_OUTPUT/llama_exp_002")
INPUT_DIR = Path("../data/")
DEBUG = False

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Function definitions
def load_data(file_name: str) -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / file_name, sep=',')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'generated': 'label'})
    return df

def tokenize_data(df: pd.DataFrame, tokenizer) -> Dataset:
    def preprocess_function(examples, max_length=512):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)
    
    ds = Dataset.from_pandas(df)
    return ds.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, predictions)
    }

def init_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = LlamaForSequenceClassification.from_pretrained(
        TARGET_MODEL,
        num_labels=2,
        quantization_config=bnb_config
    )
    base_model.config.pretraining_tp = 1  # 1 is 7b
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(base_model, peft_config)
    
    return model, tokenizer

def get_training_args() -> TrainingArguments:
    steps = 5 if DEBUG else 20
    return TrainingArguments(
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=16,
        max_grad_norm=0.3,
        optim='paged_adamw_32bit',
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=False,
        warmup_steps=steps,
        eval_steps=steps,
        logging_steps=steps,
        report_to='wandb'  # or 'wandb' if not DEBUG
    )

def clear_cuda_cache():
    torch.cuda.empty_cache()
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def create_submission(valid_df: pd.DataFrame, logits) -> pd.DataFrame:
    probs = sigmoid(logits[:, 1])
    sub = pd.DataFrame()
    sub['id'] = valid_df['id']
    sub['generated'] = probs
    return sub

# Main script logic
def main():
    # Load data
    train_df = load_data("train_essays_RDizzl3_seven_v1.csv")
    # test_df = load_data("llm-detect-ai-generated-text/test_essays.csv")
    external_df = load_data("daigt-external-dataset/daigt_external_dataset.csv")
    
    # Preprocess data
    train_df = preprocess_data(train_df)
    # test_df = preprocess_data(test_df)
    external_df = preprocess_data(external_df)

    # Further data preprocessing steps...
    external_df = external_df[["id", "source_text"]]
    external_df.columns = ["id", "text"]
    external_df['text'] = external_df['text'].str.replace('\n', '')
    external_df["label"] = 1

    train_df = pd.concat([train_df, external_df])
    train_df.reset_index(inplace=True, drop=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = train_df.loc[:, train_df.columns != "label"]
    y = train_df.loc[:, train_df.columns == "label"]

    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
        train_df.loc[valid_index, "fold"] = i
        
    print(train_df.groupby("fold")["label"].value_counts())
    train_df.head()
    
    valid_df = train_df[train_df["fold"] == 0]
    train_df = train_df[train_df["fold"] != 0]
    if DEBUG:
        train_df = train_df.sample(300)
        valid_df = valid_df.sample(50)

    # Initialize model and tokenizer
    model, tokenizer = init_model_and_tokenizer()
    
    # Tokenize data
    train_tokenized_ds = tokenize_data(train_df, tokenizer)
    valid_tokenized_ds = tokenize_data(valid_df, tokenizer)

    # Initialize Trainer
    training_args = get_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_ds,
        eval_dataset=valid_tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save model and cleanup
    trainer.save_model(output_dir=str(OUTPUT_DIR))
    for path in Path(training_args.output_dir).glob("checkpoint-*"):
        if path.is_dir():
            rmtree(path)

    # Clear CUDA cache
    clear_cuda_cache()

    # Prediction and submission
    # pred_output = trainer.predict(valid_tokenized_ds)
    # logits = pred_output.predictions
    # sub = create_submission(test_df, logits)
    # sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

if __name__ == "__main__":
    main()
