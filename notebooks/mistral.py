# %%
from __future__ import annotations

TARGET_MODEL = "mistralai/Mistral-7B-v0.1"

DEBUG = False

# %%
from pathlib import Path

OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

INPUT_DIR = Path("../data/")

# %%
import pandas as pd

train_df = pd.read_csv(INPUT_DIR / "train_essays_RDizzl3_seven_v1.csv", sep=',')
test_df = pd.read_csv(INPUT_DIR / "llm-detect-ai-generated-text/test_essays.csv", sep=',')
external_df = pd.read_csv(INPUT_DIR / "daigt-external-dataset/daigt_external_dataset.csv", sep=',')
train_prompts_df = pd.read_csv(INPUT_DIR / "llm-detect-ai-generated-text/train_prompts.csv", sep=',')

# show shape
print(f'train_df.shape: {train_df.shape}')
print(f'test_df.shape: {test_df.shape}')
print(f'external_df.shape: {external_df.shape}')
print(f'train_prompts_df.shape: {train_prompts_df.shape}')

# %%
train_df = train_df.rename(columns={'generated': 'label'})
test_df = test_df.rename(columns={'generated': 'label'})
external_df = external_df.rename(columns={'generated': 'label'})

# %%
train_df.label.value_counts()

# %%
train_df.head(3)

# %%
test_df.head(3)

# %%
external_df.head(3)

# %%
external_df = external_df[["id", "source_text"]]
external_df.columns = ["id", "text"]
external_df['text'] = external_df['text'].str.replace('\n', '')
external_df["label"] = 1

train_df = pd.concat([train_df, external_df])
train_df.reset_index(inplace=True, drop=True)
print(f"Train dataframe has shape: {train_df.shape}")
train_df.head()

# %%
train_df.value_counts("label")

# %%
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_df.loc[:, train_df.columns != "label"]
y = train_df.loc[:, train_df.columns == "label"]

for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
    train_df.loc[valid_index, "fold"] = i
    
print(train_df.groupby("fold")["label"].value_counts())
train_df.head()

# %%
valid_df = train_df[train_df["fold"] == 0]
train_df = train_df[train_df["fold"] != 0]

# %%
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType # type: ignore
from transformers import BitsAndBytesConfig
import torch

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

# %%
from transformers import AutoTokenizer, LlamaForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# %%
base_model = LlamaForSequenceClassification.from_pretrained(
    TARGET_MODEL,
    num_labels=2,
    quantization_config=bnb_config
)
base_model.config.pretraining_tp = 1 # 1 is 7b
base_model.config.pad_token_id = tokenizer.pad_token_id

# %%
model = get_peft_model(base_model, peft_config)

# %%
model.print_trainable_parameters()

# %%
# debug
if DEBUG:
    train_df = train_df.sample(300)
    valid_df = valid_df.sample(50)
#train_df = train_df.sample(100)
#valid_df = valid_df.sample(30)
print(train_df.label.value_counts(), valid_df.label.value_counts())

# %%
# datasets
from datasets import Dataset

# from pandas
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

# %%
def preprocess_function(examples, max_length=512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)

# %%
train_tokenized_ds = train_ds.map(preprocess_function, batched=True)
valid_tokenized_ds = valid_ds.map(preprocess_function, batched=True)

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# %%
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy_val = accuracy_score(labels, predictions)
    roc_auc_val = roc_auc_score(labels, predictions)
    
    return {
        "accuracy": accuracy_val,
        "roc_auc": roc_auc_val,
    }

# %%
from transformers import TrainingArguments, Trainer

steps = 5 if DEBUG else 20

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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
    report_to='none' # if DEBUG else 'wandb',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_ds,
    eval_dataset=valid_tokenized_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# %%
from shutil import rmtree

trainer.save_model(output_dir=str(OUTPUT_DIR))

for path in Path(training_args.output_dir).glob("checkpoint-*"):
    if path.is_dir():
        rmtree(path)

# %%
del trainer, model, base_model

# %%
# cuda cache clear
import torch
torch.cuda.empty_cache()

# %%
# load model / tokenizer with 4bit bnb

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType # type: ignore
from transformers import BitsAndBytesConfig
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# %%
from transformers import AutoTokenizer, LlamaForSequenceClassification

base_model = LlamaForSequenceClassification.from_pretrained(
    TARGET_MODEL,
    num_labels=2,
    quantization_config=bnb_config,
    device_map={"":0}
)
base_model.config.pretraining_tp = 1 # 1 is 7b
base_model.config.pad_token_id = tokenizer.pad_token_id

# %%
model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))

# %%
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# %%
pred_output = trainer.predict(valid_tokenized_ds)
logits = pred_output.predictions

# %%
logits = pred_output.predictions

# %%
valid_df.label

# %%
logits

# %%
# from scipy.special import expit as sigmoid
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  
probs = sigmoid(logits[:, 1])
probs.shape, probs[0:5]

# %%
sub = pd.DataFrame()
sub['id'] = valid_df['id']
sub['generated'] = probs
# sub.to_csv('submission.csv', index=False)
sub.head()


