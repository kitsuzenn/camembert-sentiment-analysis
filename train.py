from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch
import numpy as np

# Charger le dataset
df = pd.read_csv('dataset_3classes_v3.csv')
print(f"Dataset : {df.shape}")
print(df['label'].value_counts())

# Charger tokenizer et modèle
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=3)

# Convertir en Dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Tokeniser
def tokenize_function(example):
    return tokenizer(example['review'], truncation=True, max_length=512, padding='max_length')

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(['review'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Découper train/test
split = dataset.train_test_split(test_size=0.2, seed=42)

# Configurer l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=50,
)

# Créer le Trainer et entraîner
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split['train'],
    eval_dataset=split['test'],
)

trainer.train()

# Sauvegarder
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
print("Modèle sauvegardé ✅")
