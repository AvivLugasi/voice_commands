from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
import torch

# Custom dataset class
class SentencePairDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_a, text_b = self.texts[idx]
        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Prepare your data
texts = [("open it clear use c2 throw flashbang", "breach with c2 use flashbang"), ("make entry use c2 deploy flashbang", "arrest him"), ("make entry use c2 deploy flashbang", "breach with c2 use flashbang"), ("open it clear use c2 throw flashbang", "ake entry use c2 deploy flashbang")]
labels = [1, 0, 1, 1]  # 1 if sentences have the same meaning, 0 otherwise
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentencePairDataset(texts, labels, tokenizer, max_length=128)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)
train_dataset = SentencePairDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = SentencePairDataset(val_texts, val_labels, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 19

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_train_loss}')

    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')

from Model.BertModel import SentenceEmbedderModel
from Model.Utils import cosine_sim

embedding_model = SentenceEmbedderModel(model_name='bert-base-uncased')
embedding_model.load_weights_from_trained_model(model)

sentence_1 = "make entry use c2 throw flashbang"
sentence_2 = "arrest him"
sentence_vectors_np_1 = embedding_model.sentence_to_vector(processed_sentence=sentence_1)
sentence_vectors_np_2 = embedding_model.sentence_to_vector(processed_sentence=sentence_2)

print(cosine_sim(sentence_vectors_np_1, sentence_vectors_np_2, is_1d = True))
