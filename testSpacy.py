from SpeachToText.AudioCapture import Recorder
import time

from TextProcessing.TextProcessor import TextProcessor
import whisper
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
sentences = [
    "This is the first sentence.",
    "Here's the second sentence."
]

# Tokenize and encode the sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print(encoded_input)
# Forward pass
with torch.no_grad():
    outputs = model(**encoded_input)

# Obtain the embeddings from BERT's output (last hidden states)
sentence_embeddings = outputs.last_hidden_state

print(sentence_embeddings)
# # Load the Whisper Model
# start_time = time.time()
# whisper_model = whisper.load_model("medium")
# end_time = time.time()
# print(f"Time taken to load model: {end_time - start_time} seconds")
# processor = TextProcessor(remove_stopwords=False)
#
# corpus = '''
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
# breach with c2 use flashbang.
# open with c2 use flashbang.
# make entry use c2.
# make entry use c2 and stun grenade.
# open with c2 throw tear gas.
# cover the area.
# move there.
# follow me.
# bag it.
# recover evidence.
# stack up the door.
# cover the door.
# '''
# processed_sentences = processor.process_text(corpus)
# print(processed_sentences)
#
# # Load the spaCy pipeline for transformers
# nlp = spacy.load("en_core_web_trf")
#
# def _cosine_sim(vec1, vec2):
#     return cosine_similarity([vec1], [vec2])[0][0]
#
# def find_most_similar(model, processed_input_sentence, processed_commands_list):
#     embeddings = []
#     for sentence in processed_commands_list:
#         doc = model(sentence)
#         print(doc.vector)
#         embeddings.append(doc.vector)
#
#     embedded_input = model(processed_input_sentence)
#     similarity = -1
#     most_similar = processed_commands_list[0]
#     for i, embedded_command in enumerate(embeddings):
#         if _cosine_sim(embedded_input, embedded_command) > similarity:
#             similarity = _cosine_sim(embedded_input, embedded_command)
#             most_similar = processed_commands_list[i]
#
#     return most_similar, similarity
#
#
#
# while True:
#     start_time = time.time()
#     recorder = Recorder()
#     recorder.capture_voice_command(record_method_param="space")
#     end_time = time.time()
#     print(f"Time taken to recording: {end_time - start_time} seconds")
#     start_time = time.time()
#     # Transcribe the Ogg file
#     result = whisper_model.transcribe("Assets/AudioFiles/player_recorded_command_temp.wav", fp16=False, language='en')
#     end_time = time.time()
#     print(f"Time taken to transcribe text: {end_time - start_time} seconds")
#     # Print the transcription result
#     print(result["text"])
#     text = result["text"]
#     processed_sentence = processor.process_text(text)
#     if len(processed_sentence) > 1:
#         processed_sentence = ' '.join(processed_sentence)
#     print(processed_sentence)
#
#     most_similar, similarity = find_most_similar(nlp, processed_sentence[0], processed_sentences)
#     print(f"most similar to {processed_sentence}: {most_similar} {similarity}")