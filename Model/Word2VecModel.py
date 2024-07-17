import heapq
import time

from gensim.models import Word2Vec
from numpy import ndarray
import numpy as np
import multiprocessing
from gensim.models import KeyedVectors
import os
from Model.Utils import cosine_sim

MODEL_PATH = "Assets/Model/voice_command_word2vec.model"
GOOGLE_WORD2VEC_PATH = "Assets/Model/GoogleNews-vectors-negative300.bin"
VECTOR_SIZE = 300


class Model:
    def __init__(self, window=5, min_count=1):
        self.vector_size = VECTOR_SIZE
        self.window = window
        self.min_count = min_count
        self.model = self.load_model(path=MODEL_PATH)

    def sentence_to_vector(self, sentence: str):
        word_vectors = [self.model.wv[word] for word in sentence if word in self.model.wv.key_to_index.keys()]
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.median(word_vectors, axis=0)

    def load_model(self, path: str):
        if os.path.exists(path):
            return Word2Vec.load(path)
        else:
            return self.load_google_model(path=GOOGLE_WORD2VEC_PATH)

    def load_google_model(self, path: str):
        w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
        model = Word2Vec(vector_size=self.vector_size,
                         window=self.window,
                         min_count=self.min_count,
                         workers=multiprocessing.cpu_count() - 1)
        model.build_vocab([list(w2v_model.key_to_index.keys())])
        model.wv.vectors = w2v_model.vectors
        return model

    def train_model(self, processed_corpus, epochs=100):
        self.model.build_vocab(processed_corpus, update=True)
        self.model.train(processed_corpus,
                         total_examples=len(processed_corpus),
                         epochs=epochs)
        self.model.save(MODEL_PATH)

    def find_top_n_similar_sentences(self, processed_sentences: list,
                                     input_sentence: str,
                                     n=1):
        vectors_simmilarity_map = {}
        input_vector = self.sentence_to_vector(input_sentence)

        for sentence in processed_sentences:
            vectorized_sentence = self.sentence_to_vector(sentence)
            vectors_simmilarity_map[sentence] = cosine_sim(vectorized_sentence, input_vector)

        return heapq.nlargest(n, vectors_simmilarity_map.items(), key=lambda item: item[1])
