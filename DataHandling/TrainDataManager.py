from DataHandling.DataLoader import DataLoader
from DataHandling.DataLoader import COMMAND_VARIATIONS_KEY
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import random
import csv
from sklearn.model_selection import train_test_split

PARAPHRASES = 1
NOT_PARAPHRASES = 0
CLASSES_RATIO = 2
DEFAULT_TRAIN_DATA_PATH = "Assets/Data/ReadyOrNot/BertSequanceClassiferDataSet.csv"

MAX_PADDING = 30


class TrainDataGenerator:
    def __init__(self):
        self.dataloader = DataLoader()
        self.sentences_pairs = []
        self.pairs_labels = []
        self.train_pairs = []
        self.val_pairs = []
        self.train_labels = []
        self.val_labels = []

    def generate_bert_train_data(self):
        formatted_commands_list = self.dataloader.get_commands_list()
        sentences_pairs = []
        pairs_labels = []
        sentences_pairs, pairs_labels = _generate_positive_samples(formatted_commands_list,
                                                                       sentences_pairs,
                                                                       pairs_labels)
        sentences_pairs, pairs_labels = _generate_negative_samples(formatted_commands_list,
                                                                       sentences_pairs,
                                                                       pairs_labels)
        self.sentences_pairs, self.pairs_labels = sentences_pairs, pairs_labels

    def write_train_data_to_file(self,
                                 path=DEFAULT_TRAIN_DATA_PATH):
        # Writing one line at a time
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write each row one at a time
            for index, sentence_pair in enumerate(self.sentences_pairs):
                writer.writerow([sentence_pair[0], sentence_pair[1], self.pairs_labels[index]])

    def load_train_data(self,
                        path=DEFAULT_TRAIN_DATA_PATH):
        # Read the CSV file
        with open(path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.sentences_pairs.append((row[0], row[1]))
                self.pairs_labels.append(row[2])

    def train_validation_split(self):
        self.train_pairs, self.val_pairs, self.train_labels, self.val_labels = train_test_split(self.sentences_pairs,
                                                                                      self.pairs_labels,
                                                                                      test_size=0.25)

class TrainDataSample:
    def __init__(self, sentence_pair, label):
        self.sentence_pair = sentence_pair
        self.label = label


class TrainDataBatchLoader(Dataset):
    def __init__(self,
                 train_data_samples_list:list,
                 tokenizer:BertTokenizer,
                 max_length_padding=MAX_PADDING):
        self.train_data_samples_list = train_data_samples_list
        self.tokenizer = tokenizer
        self.max_length_padding = max_length_padding

    def __len__(self):
        return len(self.train_data_samples_list)

    def __getitem__(self, idx):
        text_a, text_b = self.train_data_samples_list[idx]
        sentence_pair_input = self.tokenizer(
            text_a, text_b,
            padding='max_length',
            max_length=self.max_length_padding,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': sentence_pair_input['input_ids'].squeeze(0),
            'attention_mask': sentence_pair_input['attention_mask'].squeeze(0),
            'token_type_ids': sentence_pair_input['token_type_ids'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def _generate_positive_samples(formatted_commands_list: list,
                              sentences_pairs: list,
                              pairs_labels: list):
    for command in formatted_commands_list:
        command_variations = command[COMMAND_VARIATIONS_KEY]
        for curr, variation in enumerate(command_variations):
            pair_index = curr + 1
            while pair_index < len(command_variations):
                sentences_pairs.append((command_variations[curr], command_variations[pair_index]))
                pairs_labels.append(PARAPHRASES)
                pair_index+=1

    return sentences_pairs, pairs_labels


def _generate_negative_samples(formatted_commands_list: list,
                              sentences_pairs: list,
                              pairs_labels: list):
    negative_pair_set = set()

    while len(negative_pair_set) != CLASSES_RATIO * len(sentences_pairs):
        command1, command2 = random.sample(formatted_commands_list, 2)
        command1_variation = random.sample(command1[COMMAND_VARIATIONS_KEY], 1)[0]
        command2_variation = random.sample(command2[COMMAND_VARIATIONS_KEY], 1)[0]
        negative_pair_set.add((command1_variation, command2_variation))

    negative_labels_list = [NOT_PARAPHRASES] * len(negative_pair_set)
    sentences_pairs.extend(list(negative_pair_set))
    pairs_labels.extend(negative_labels_list)
    return sentences_pairs, pairs_labels
