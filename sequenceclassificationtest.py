from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
import torch
from DataHandling.TrainDataManager import TrainDataManager

model_name = 'bert-base-uncased'

tdm = TrainDataManager()
tdm.load_train_data()
tdm.train_validation_split()

tokenizer = BertTokenizer.from_pretrained(model_name)

inputs = tokenizer(tdm.train_pairs,
                   padding=True)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))
print(inputs["attention_mask"][1])