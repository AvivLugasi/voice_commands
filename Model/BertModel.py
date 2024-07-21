from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import numpy as np
from Model.Utils import cosine_sim, get_running_device
import heapq

PREFIX_TOKEN = "[CLS] "
SUFFIX_TOKEN = " [SEP]"

class SentenceEmbedderModel:
    def __init__(self,
                 model_name='bert-base-uncased',
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

    def find_top_n_similar_sentences(self,
                                     processed_sentence: str,
                                     processed_commands: list,
                                     n=1):
        vectors_similarity_map = {}
        input_vector = self.sentence_to_vector(processed_sentence)

        for sentence in processed_commands:
            vectorized_sentence = self.sentence_to_vector(sentence)
            vectors_similarity_map[sentence] = cosine_sim(vectorized_sentence, input_vector, is_1d = True)

        return heapq.nlargest(n, vectors_similarity_map.items(), key=lambda item: item[1])

    def load_weights_from_trained_model(self, trained_model):

        # Extract state dictionaries
        fine_tuned_state_dict = trained_model.state_dict()
        base_model_state_dict = self.model.state_dict()

        # Filter out unnecessary keys in the state_dict
        filtered_state_dict = {k: v for k, v in fine_tuned_state_dict.items() if k in base_model_state_dict}

        # Load the filtered state_dict into the base model
        self.model.state_dict = self.model.load_state_dict(filtered_state_dict, strict=False)


class SequenceClassificationModel:
    def __init__(self):
        pass


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
