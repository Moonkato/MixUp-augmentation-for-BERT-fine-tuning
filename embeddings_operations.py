import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# Из токенов получаем эмбеддинги
def get_embeddings(data):
    with torch.no_grad():
        outputs = model(**data)
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings


# Из эмбеддингов получаем токены
def from_embeddings_get_data(embeddings, all_tokens, token_type_ids, attention_mask, label):
    normalized_embeddings = torch.nn.functional.normalize(embeddings)
    tokens = []

    # Проход по каждому эмбеддингу
    for index, embedding in enumerate(normalized_embeddings):
        embedding_np = embedding.reshape(1, -1)
        similarities = cosine_similarity(embedding_np, normalized_embeddings)
        most_similar_index = np.argmax(similarities)
        token_id = all_tokens[most_similar_index].item()
        tokens.append(token_id)

    tokens_torch = torch.tensor(tokens).unsqueeze(0)

    return {
        'label': torch.tensor(label),
        'input_ids': tokens_torch.squeeze(0),
        'token_type_ids': token_type_ids.squeeze(0),
        'attention_mask': attention_mask.squeeze(0)
    }


# MixUp embeddings метод
def mixup_embeddings(embeddings_1, embeddings_2, label_a, label_b, alpha):
    lam = np.random.beta(alpha, alpha)
    aug_emb = lam * embeddings_1 + (1 - lam) * embeddings_2
    aug_label = lam * label_a + (1 - lam) * label_b
    rounded_label = round(aug_label)

    return aug_emb, rounded_label
