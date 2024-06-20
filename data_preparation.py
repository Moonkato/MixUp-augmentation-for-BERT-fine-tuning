import random
import logging

import torch
from datasets import load_dataset
from transformers import BertTokenizer

from embeddings_operations import get_embeddings, from_embeddings_get_data, mixup_embeddings

logging.basicConfig(level=logging.INFO)

logging.info("Downloading dataset")
dataset = load_dataset("rotten_tomatoes")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Оборачиваем токенизированные текстовые данные в torch Dataset
class Data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# Функция токенизации текста
def tokenizer_function(data):
    return tokenizer(data['text'], return_tensors='pt', padding="max_length", truncation=True)


# Функция аугментации, принимающая на вход dataset записей {'text': text, 'label': label}
def data_augmentation(dataset):
    # Список с аугментированными данными, полученные embedding MixUp методом
    new_data = []

    # Получаем токены каждой записи в dataset
    all_tokens = get_all_tokens(dataset)

    # Для каждого элемента из dataset найдем случайную пару для MixUp алгоритма
    for sample_1 in dataset:
        tokenized_data_1 = tokenizer_function(sample_1)

        sample_2 = sample_1
        while sample_1 is sample_2:
            sample_2 = random.choice(dataset)

        tokenized_data_2 = tokenizer_function(sample_2)

        label_a = sample_1['label']
        label_b = sample_2['label']

        # Переменная, куда складываем токены каждой записи dataset
        combined_data = tokenizer_function(sample_1)
        combined_data['input_ids'] = all_tokens

        # Получаем эмбеддинги каждого выбранного экземпляра из dataset
        embeddings_first = get_embeddings(tokenized_data_1)
        embeddings_second = get_embeddings(tokenized_data_2)

        # Embedding MixUp метод
        augmented_embeddings, augmented_label = mixup_embeddings(embeddings_1=embeddings_first,
                                                                 embeddings_2=embeddings_second,
                                                                 label_a=label_a, label_b=label_b, alpha=0.9)

        # Полученные эмбеддинги преобразуем обратно в токены
        augmented_processed = from_embeddings_get_data(embeddings=augmented_embeddings, all_tokens=all_tokens,
                                                       token_type_ids=tokenized_data_1['token_type_ids'],
                                                       attention_mask=tokenized_data_1['attention_mask'],
                                                       label=augmented_label)

        new_data.append(augmented_processed)

    return new_data


def get_all_tokens(dataset):
    # Инициализируем список для хранения всех input_ids
    all_input_ids = []

    # Обрабатываем первый элемент датасета отдельно
    f_sample = dataset[0]
    tokenized_data = tokenizer_function(f_sample)
    all_input_ids.append(tokenized_data['input_ids'])

    # Обрабатываем остальные элементы датасета
    for index, data in enumerate(dataset, start=1):
        tokenized_data = tokenizer_function(data)
        all_input_ids.append(tokenized_data['input_ids'])

    # Соединяем все input_ids в один большой тензор по первому измерению (конкатенация)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_input_ids = all_input_ids.view(-1)

    return all_input_ids


def from_tensor_to_list(data):
    input_ids_list = []
    labels_list = []
    for elem in data:
        input_ids_list.append(elem['input_ids'].tolist())
        labels_list.append(int(elem['label'].item()))

    return input_ids_list, labels_list


row_train_data = dataset['train']

train_data_main = dataset['train'].map(tokenizer_function, batched=True)
validation_data_main = dataset['validation'].map(tokenizer_function, batched=True)

train_data_main.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
validation_data_main.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

row_train_data = row_train_data.shuffle(seed=42)
row_train_data = row_train_data.select(range(3500))

augmented_data = data_augmentation(row_train_data)

original_data, original_labels = from_tensor_to_list(train_data_main)
aug_data, aug_labels = from_tensor_to_list(augmented_data)
train_data = original_data + aug_data
train_labels = original_labels + aug_labels

train_data = {
    'input_ids': train_data
}

val_data, val_labels = from_tensor_to_list(validation_data_main)

val_data = {
    'input_ids': val_data
}

# формируем датасеты для обучения
train_dataset = Data(train_data, train_labels)
validation_dataset = Data(val_data, val_labels)