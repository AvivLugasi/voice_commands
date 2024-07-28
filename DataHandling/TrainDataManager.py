<<<<<<< HEAD
from DataHandling.DataIO import DataIO
from DataHandling.DataIO import COMMAND_VARIATIONS_KEY
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
=======
from DataHandling.DataLoader import DataLoader
from DataHandling.DataLoader import COMMAND_VARIATIONS_KEY
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3
import random
import csv
from sklearn.model_selection import train_test_split

PARAPHRASES = 1
NOT_PARAPHRASES = 0
CLASSES_RATIO = 2
DEFAULT_TRAIN_DATA_PATH = "Assets/Data/ReadyOrNot/BertSequanceClassiferDataSet.csv"

<<<<<<< HEAD

class TrainDataGenerator:
    def __init__(self):
        self.data_io = DataIO()
=======
MAX_PADDING = 30


class TrainDataGenerator:
    def __init__(self):
        self.dataloader = DataLoader()
>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3
        self.sentences_pairs = []
        self.pairs_labels = []
        self.train_pairs = []
        self.val_pairs = []
        self.train_labels = []
        self.val_labels = []

    def generate_bert_train_data(self):
<<<<<<< HEAD
        formatted_commands_list = self.data_io.get_commands_list()
=======
        formatted_commands_list = self.dataloader.get_commands_list()
>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3
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
<<<<<<< HEAD
        self.label = int(label)


class TrainDataSet(Dataset):
    def __init__(self,
                 train_data_samples_list:list,
                 tokenizer:BertTokenizer):
        self.train_data_samples_list = train_data_samples_list
        self.tokenizer = tokenizer
=======
        self.label = label


class TrainDataBatchLoader(Dataset):
    def __init__(self,
                 train_data_samples_list:list,
                 tokenizer:BertTokenizer,
                 max_length_padding=MAX_PADDING):
        self.train_data_samples_list = train_data_samples_list
        self.tokenizer = tokenizer
        self.max_length_padding = max_length_padding
>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3

    def __len__(self):
        return len(self.train_data_samples_list)

    def __getitem__(self, idx):
<<<<<<< HEAD
        sentence_pair = self.train_data_samples_list[idx].sentence_pair
        return sentence_pair[0],\
               sentence_pair[1],\
               self.train_data_samples_list[idx].label,\
               self.tokenizer

def _collate_fn_custom_padding(batch):
    sentences_a = [item[0] for item in batch]
    sentences_b = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    tokenizer = batch[0][3]

    # Tokenize the sentences
    tokenized_inputs = tokenizer(
        sentences_a,
        sentences_b,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = tokenized_inputs['input_ids'].squeeze(0)
    attention_masks = tokenized_inputs['attention_mask'].squeeze(0)
    token_type_ids = tokenized_inputs['token_type_ids'].squeeze(0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids,
        'labels': labels
    }
=======
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

>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3

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

<<<<<<< HEAD
=======

>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3
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
<<<<<<< HEAD

def _create_data_loader(dataset,
                        batch_size=8,
                        shuffle=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=_collate_fn_custom_padding)


def assemble_train_val_data_loaders(batch_size=8,
                                    shuffle_train_data=True):
    tdg = TrainDataGenerator()
    tdg.load_train_data()
    tdg.train_validation_split()

    train_dataset_samples = [TrainDataSample(sen_pair, label)
                             for sen_pair, label in zip(tdg.train_pairs, tdg.train_labels)]
    val_dataset_samples = [TrainDataSample(sen_pair, label)
                           for sen_pair, label in zip(tdg.val_pairs, tdg.val_labels)]

    train_data_set = TrainDataSet(train_dataset_samples, tokenizer)
    val_data_set = TrainDataSet(val_dataset_samples, tokenizer)

    train_data_loader = create_data_loader(train_data_set, shuffle=shuffle_train_data, batch_size=batch_size)
    val_data_loader = create_data_loader(val_data_set, shuffle=False, batch_size=batch_size)
    return train_data_loader, val_data_loader
=======
>>>>>>> aa980eb66d9260bb6ed7943ab5c3ff2b27fb8bb3
