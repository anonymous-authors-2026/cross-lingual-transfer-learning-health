import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Data", "Memo_Dataset.csv")
df = pd.read_csv(DATA_PATH)

# Keep English + Arabic + Trigger
df = df[['Question', 'Question_eng', 'Trigger']]
df = df.rename(columns={'Trigger': 'label'})

# Create Arabic and English copies
df_arabic = df[['Question', 'label']].rename(columns={'Question': 'text'})
df_english = df[['Question_eng', 'label']].rename(columns={'Question_eng': 'text'})

# Combine
df_combined = pd.concat([df_arabic, df_english], ignore_index=True)
df_combined['label'] = df_combined['label'].astype(int)

print("Dataset combined: ", df_combined.shape)
print(df_combined['label'].value_counts())

# Downsample
min_count = df_combined['label'].value_counts().min()
df_balanced = (
    df_combined.groupby('label', group_keys=False)
    .apply(lambda x: x.sample(min_count, random_state=42))
    .reset_index(drop=True)
)

print("Balanced dataset:", len(df_balanced))
print(df_balanced['label'].value_counts())

# Train / Val split
train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced['label']
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(df_balanced['label'].unique())
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./xlmr_trigger_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model
SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "trigger_xlmr_model_eng_ar")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("Model saved to:", SAVE_PATH)
