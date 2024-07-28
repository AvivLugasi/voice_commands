from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_scheduler
import torch
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from DataHandling.TrainDataManager import assemble_train_val_data_loaders
from tqdm.auto import tqdm
import numpy as np
from Model.Utils import cosine_sim, get_running_device
import heapq

PREFIX_TOKEN = "[CLS] "
SUFFIX_TOKEN = " [SEP]"

TRAINED_SEQUENCE_CLASSIFIER_DIR = "Assets/Model/BertSequenceClassifier"
BASE_MODEL = 'bert-base-uncased'

class SentenceEmbedderModel:
    def __init__(self,
                 model_name=BASE_MODEL,
                 output_hidden_states=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states = output_hidden_states)

    def sentence_to_vector(self,
                           processed_sentence: str,
                           mod='SENT'):
        self.model.eval()
        indexed_tokens_tensor, segments_ids_tensor = self.get_model_inputs(processed_sentence)
        with torch.no_grad():
            output = self.model(indexed_tokens_tensor, segments_ids_tensor)
        hidden_states = output[2]
        if mod == 'SENT' or mod == 1:
            embedded_tensor = _get_sentence_vector(hidden_states)
            embedded_tensor = embedded_tensor.detach().numpy()
        elif mod == 'WORD' or mod == 2:
            output_grouped_by_tokens = _group_output_by_tokens(hidden_states)
            embedded_tensor = _get_word_vectors(output_grouped_by_tokens)

        return embedded_tensor

    def get_model_inputs(self, processed_sentence: str):
        processed_sentence = _preprocess_sentence(processed_sentence)
        tokenized_sentence = self.tokenizer.tokenize(processed_sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        segments_ids = [1] * len(tokenized_sentence)
        indexed_tokens_tensor = torch.tensor([indexed_tokens])
        segments_ids_tensor = torch.tensor([segments_ids])
        return indexed_tokens_tensor, segments_ids_tensor

    def load_weights_from_trained_model(self, trained_model):

        # Extract state dictionaries
        fine_tuned_state_dict = trained_model.state_dict()
        base_model_state_dict = self.model.state_dict()

        # Filter out unnecessary keys in the state_dict
        filtered_state_dict = {k: v for k, v in fine_tuned_state_dict.items() if k in base_model_state_dict}

        # Load the filtered state_dict into the base model
        self.model.state_dict = self.model.load_state_dict(filtered_state_dict, strict=False)

"""
TODO: implement classifier class, use traindatamanager batchloader class
implement training loop in keras
check for a way to extract sentence embedding from the trained model/ pass relevant weights to the bert model
"""
class SequenceClassificationModel:
    def __init__(self,
                 model_name=TRAINED_SEQUENCE_CLASSIFIER_DIR,
                 tokenizer_name=BASE_MODEL,
                 output_hidden_states=True,
                 num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name,
                                                                   output_hidden_states=output_hidden_states,
                                                                   num_labels=num_labels)
        self.device = get_running_device()

    def sentence_to_vector(self, processed_sentence:str):
        self.model.eval()
        inputs = self.tokenizer(processed_sentence,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')

        with torch.no_grad():
            hidden_states = self.model(**inputs).hidden_states
        return _get_sentence_vector(hidden_states).detach().numpy()

    def train_model(self,
                    saved_model_dir=TRAINED_SEQUENCE_CLASSIFIER_DIR,
                    num_of_epochs=8,
                    batch_size=8,
                    shuffle_train_data=True):
        train_data_loader, val_data_loader = assemble_train_val_data_loaders(batch_size=batch_size,
                                                                             shuffle_train_data=shuffle_train_data)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        num_training_steps = num_of_epochs * len(train_data_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        model.to(self.device)
        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for epoch in range(num_of_epochs):
            for batch in train_data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                print(f"epoch:{epoch} loss: {loss} batch progress:{progress_bar}")

        self.eval_model(val_data_loader)
        self.save_model(output_dir=saved_model_dir)

    def save_model(self,output_dir=TRAINED_SEQUENCE_CLASSIFIER_DIR):
        self.model.save_pretrained(output_dir)

    def eval_model(self,val_data_loader):
        self.model.eval()
        accuracy = Accuracy(task="binary").to(self.device)
        for batch in val_data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Update the metric with predictions and labels
            accuracy.update(predictions, batch["labels"])

        # Compute the final accuracy
        final_accuracy = accuracy.compute()
        print(f"Accuracy: {final_accuracy.item()}")

def _get_word_vectors(output_grouped_by_tokens):
    # Stores the token vectors, with shape [num of tokens x 3,072]
    # Where 3072 comes from concatenating the last 4 layers, each has 768 features
    token_vecs_cat = []

    # `output_grouped_by_tokens` is a [num of tokens x 12 x 768] tensor.

    # For each token in the sentence...
    for token in output_grouped_by_tokens:
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    return token_vecs_cat

def _get_sentence_vector(hidden_states):
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    return torch.mean(token_vecs, dim=0)


def _group_output_by_tokens(hidden_states):
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    # now the output is arranged as follows: [tokens, layers, features]
    # instead of [layers, batches, tokens, features]
    token_embeddings = token_embeddings.permute(1, 0, 2)
    return token_embeddings


def _preprocess_sentence(processed_sentence: str):
    if not processed_sentence.startswith(PREFIX_TOKEN):
        processed_sentence = PREFIX_TOKEN + processed_sentence
    if not processed_sentence.endswith(SUFFIX_TOKEN):
        processed_sentence = processed_sentence + SUFFIX_TOKEN

    return processed_sentence

def find_top_n_similar_sentences(model,
                                 processed_sentence: str,
                                 commands_embedding_dict: dict,
                                 n=1):
    vectors_similarity_map = {}
    input_vector = model.sentence_to_vector(processed_sentence)
    for command_variation in commands_embedding_dict.keys():
        vectors_similarity_map[command_variation] = cosine_sim(commands_embedding_dict[command_variation],
                                                               input_vector,
                                                               is_1d = True)

    top_n_commands = heapq.nlargest(n, vectors_similarity_map.items(), key=lambda item: item[1])
    commands, similarities = zip(*top_n_commands)
    return commands, similarities