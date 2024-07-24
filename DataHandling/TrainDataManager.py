from DataLoader import DataLoader
from DataLoader import COMMAND_VARIATIONS_KEY
import random
import csv

PARAPHRASES = 1
NOT_PARAPHRASES = 0
CLASSES_RATIO = 2
DEFAULT_TRAIN_DATA_PATH = "Assets/Data/ReadyOrNot/BertSequanceClassiferDataSet.csv"


class TrainDataManager:
    def __init__(self):
        self.dataloader = DataLoader()
        self.sentences_pairs = None
        self.pairs_labels = None


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
