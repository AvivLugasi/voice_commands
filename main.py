# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import whisper
import nltk

from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
import string
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Load the Whisper Model
model = whisper.load_model("base")

# Transcribe the Ogg file
result = model.transcribe("TestingAudioFiles/wav/[CALL]C2Placed_2.wav", fp16=False)

# Print the transcription result
print(result["text"])

text = result["text"]
clean_text = text.translate(str.maketrans('', '', string.punctuation))
lowercase_text = clean_text.lower()
lemmatizer = WordNetLemmatizer()
words = lowercase_text.split()  # Assuming 'text' is the transcribed text
base_forms = [lemmatizer.lemmatize(word) for word in words]

# Join the words with a space delimiter
result_string = " ".join(base_forms)
print(result_string)


# Define the sentences
sentences = ["breach c2 and clear", "stack up", "move and clear", "breach with shotgun and clear", "pick the lock and clear", "arrest suspect"]

def sentence_to_vector(sentence, model):
    words = sentence.lower().split()
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

start_time = time.time()
#Model = api.load("word2vec-google-news-300")
# Save the Model locally
#Model.save("word2vec-google-news-300.Model")
# Load the Model locally
model = KeyedVectors.load_word2vec_format("Assets/Model/GoogleNews-vectors-negative300.bin", binary=True)
w2v = Word2Vec(vector_size=300,
               window=5,
               min_count=1)
w2v.build_vocab([list(model.key_to_index.keys())])
w2v.wv.vectors = model.vectors
w2v.build_vocab(sentences, update=True)
w2v.train(sentences,
          total_examples=len(sentences),
          epochs=100)
w2v.init_sims(replace=True)
end_time = time.time()
print(f"Time taken to load Model: {end_time - start_time} seconds")
# Convert sentences to vectors
start_time = time.time()
vector1 = sentence_to_vector(sentences[0], w2v.wv)
vector2 = sentence_to_vector(sentences[1], w2v.wv)
end_time = time.time()
print(f"Time taken to convert sentences to vectors: {end_time - start_time} seconds")
# Calculate cosine similarity
start_time = time.time()
similarity = cosine_sim(vector1, vector2)
end_time = time.time()
print(f"Time taken to calculate cosine similarity: {end_time - start_time} seconds")
print(f"Cosine Similarity: {similarity}")

vector1 = sentence_to_vector(sentences[0], w2v.wv)
vector2 = sentence_to_vector("breach with c2 and clear".lower(), w2v.wv)
vector1 = sentence_to_vector("explosive", w2v.wv)
vector2 = sentence_to_vector("c2".lower(), w2v.wv)
similarity = cosine_sim(vector1, vector2)
print(f"Cosine Similarity: {similarity}")