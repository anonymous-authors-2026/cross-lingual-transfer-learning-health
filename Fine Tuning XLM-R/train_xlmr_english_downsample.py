import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Load Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Data", "Memo_Dataset.csv")

df = pd.read_csv(DATA_PATH)

# Keeping  ONLY English and label columns
df = df[['Question_eng', 'Trigger']]
df = df.rename(columns={'Question_eng': 'text', 'Trigger': 'label'})

# Converting the labels to int
df['label'] = df['label'].astype(int)

print("\nOriginal English dataset:", df.shape)
print(df['label'].value_counts())

# Downsample to balance labels

min_count = df['label'].value_counts().min()

df_balanced = (
    df.groupby('label', group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
      .reset_index(drop=True)
)

print("\nBalanced English dataset:", df_balanced.shape)
print(df_balanced['label'].value_counts())

# Train/Val Split

train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced['label']
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer + Preprocessing

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Loading Model

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(df_balanced['label'].unique())
)

# Metrics

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training Arguments

training_args = TrainingArguments(
    output_dir="./xlmr_english_model_downsampled",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs_english_downsampled",
    do_eval=True,
)

# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Saving Model

SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "trigger_xlmr_model_eng_downsampled")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"\n English-only DOWN-SAMPLED model saved to: {SAVE_PATH}")
