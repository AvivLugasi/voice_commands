from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
from DataHandling.TrainDataManager import TrainDataGenerator

model_name = 'bert-base-uncased'

tdg = TrainDataGenerator()
tdg.load_train_data()
tdg.train_validation_split()

tokenizer = BertTokenizer.from_pretrained(model_name)

inputs = tokenizer(tdg.train_pairs,
                   padding=True)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))
print(inputs["attention_mask"][1])