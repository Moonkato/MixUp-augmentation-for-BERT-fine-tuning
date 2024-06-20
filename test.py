import logging
import os

import gdown
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

logging.basicConfig(level=logging.INFO)
logging.info("Downloading dataset and preparing test part...")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# Функция токенизации текста
def tokenizer_function(data):
    return tokenizer(data['text'], return_tensors='pt', padding="max_length", truncation=True)


dataset = load_dataset("rotten_tomatoes")
test_data_main = dataset['test'].map(tokenizer_function, batched=True)
test_data_main.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

logging.info("Preparing to load model...")

# Загрузка модели и токенизатора из облака, если отсутствует результат выполнения этапа train
if not os.path.isdir('bert-base-cased-rotten-tomatoes-mixup'):
    url = r'https://drive.google.com/drive/u/0/folders/1eTPr3ZxCFUjIc3_W7g8K_C6xfWh-AmPR'
    gdown.download_folder(url)

model = BertForSequenceClassification.from_pretrained('bert-base-cased-rotten-tomatoes-mixup')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased-rotten-tomatoes-mixup')

test_dataset = TensorDataset(
    test_data_main['input_ids'],
    test_data_main['attention_mask'],
    test_data_main['label']
)

test_loader = DataLoader(test_dataset, batch_size=8)

model.eval()

predictions = []
true_labels = []

logging.info("Model testing...")

# Перебор тестового набора данных и выполнение предсказаний
with torch.no_grad():
    counter = 0
    for batch in test_loader:
        counter = counter + 1
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.numpy())
        true_labels.extend(labels.numpy())
        if counter == 34:
            logging.info("Processed 25%")
        elif counter == 68:
            logging.info("Processed 50%")
        elif counter == 102:
            logging.info("Processed 75%")
        elif counter == 134:
            logging.info("Processed 100%")

predictions = np.array(predictions)
true_labels = np.array(true_labels)

logging.info("Model quality assessment")

# Оценка модели
print(classification_report(true_labels, predictions))
print(accuracy_score(true_labels, predictions))
