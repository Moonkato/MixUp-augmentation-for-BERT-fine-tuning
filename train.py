import logging
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer

from data_preparation import tokenizer, train_dataset, validation_dataset


# Задание всех seed
def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_all(42)


# Расчет метрики
def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, predictions)}


model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Параметры, которые будут использоваться для обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    learning_rate=1e-5,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    seed=21)

# Передача в trainer предообученной модели bert-base-cased, tokenizer, данные для обучения, данные для валидации и способ расчета метрики:
trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=validation_dataset,
                  compute_metrics=compute_metrics)

logging.info("Training process begins")

trainer.train()

# Сохраняем модель
model.save_pretrained('./bert-base-cased-rotten-tomatoes-mixup')
tokenizer.save_pretrained('./bert-base-cased-rotten-tomatoes-mixup')
