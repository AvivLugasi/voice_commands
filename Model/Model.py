import heapq
import time
import TextProcessing

from gensim.models import Word2Vec
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import multiprocessing
from gensim.models import KeyedVectors
import os

from TextProcessing.TextProcessor import TextProcessor

MODEL_PATH = "../Assets/Model/voice_command_word2vec.model"
GOOGLE_WORD2VEC_PATH = "../Assets/Model/GoogleNews-vectors-negative300.bin"
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
        return np.mean(word_vectors, axis=0)

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
            vectors_simmilarity_map[sentence] = _cosine_sim(vectorized_sentence, input_vector)

        return heapq.nlargest(n, vectors_simmilarity_map.items(), key=lambda item: item[1])



def _cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# start_time = time.time()
# model = Model()
# processor = TextProcessor(remove_stopwords=False)
# text = '''
# Gold open and clear. Open and clear use shotgun.
# Arrest suspect. Zip the suspect.
# Stack up.
# Breach and clear use c2.
# Breaching using shotgun.
# Police dont move.
# hand's up LSPD.
# LSPD swat, dont move!
# gold team open and clear.
# open and clear use explosive.
# use flash bang and clear.
# '''
# processed_sentences = processor.process_text(text)
# print(processed_sentences)
# #model.train_model(processed_sentences, epochs=100)
# end_time = time.time()
# print(f"Time taken to load and train the model: {end_time - start_time} seconds")
# start_time = time.time()
# print(f"most similar to tie up: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processor.process_text('tie_up')[0], n=1)}")
# print(f"most similar to gold team open and clear: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processor.process_text('gold team open and clear')[0], n=1)}")
# print(f"most similar to open and clear use explosive: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processor.process_text('open and clear use explosive')[0], n=1)}")
# print(f"most similar to open use shotgun: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processor.process_text('open use shotgun')[0], n=1)}")
# print(f"most similar to flashbang and clear: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processor.process_text('flashbang and clear')[0], n=1)}")
# print(f"most similar to blow and clear: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processor.process_text('blow and clear')[0], n=1)}")
# end_time = time.time()
# print(f"Time taken for inference: {end_time - start_time} seconds")
