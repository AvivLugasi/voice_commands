from SpeachToText.AudioCapture import Recorder
import time

from TextProcessing.TextProcessor import TextProcessor
import whisper
from Model.BertModel import SentenceEmbedderModel
from Model.Utils import cosine_sim
import json
import numpy as np
from DataLoader.DataLoader import DataLoader


dataloader = DataLoader()
variations_embedding_dict = dataloader.load_variations_embedding_dict()

model = SentenceEmbedderModel(model_name='bert-base-uncased')

sentence_vectors_np_1 = variations_embedding_dict['open and clear with explosives use stun grenade']
sentence_2 = "open and clear with explosive then throw stun grenade"
sentence_vectors_np_2 = model.sentence_to_vector(processed_sentence=sentence_2)

print(cosine_sim(sentence_vectors_np_1, sentence_vectors_np_2, is_1d = True))

# Load the Whisper Model
start_time = time.time()
whisper_model = whisper.load_model("medium")
end_time = time.time()
print(f"Time taken to load model: {end_time - start_time} seconds")
processor = TextProcessor(remove_stopwords=False)

while True:
    start_time = time.time()
    recorder = Recorder()
    recorder.capture_voice_command(record_method_param="space")
    end_time = time.time()
    print(f"Time taken to recording: {end_time - start_time} seconds")
    start_time = time.time()
    # Transcribe the Ogg file
    result = whisper_model.transcribe("Assets/AudioFiles/player_recorded_command_temp.wav", fp16=False, language='en')
    end_time = time.time()
    print(f"Time taken to transcribe text: {end_time - start_time} seconds")
    # Print the transcription result
    print(result["text"])
    text = result["text"]
    processed_sentence = processor.process_text(text)
    if isinstance(processed_sentence, list):
        processed_sentence_as_list = ' '.join(processed_sentence)
        processed_sentence = ''.join(processed_sentence_as_list)
    print(processed_sentence)

    start_time = time.time()
    most_similar = model.find_top_n_similar_sentences(processed_sentence, variations_embedding_dict)
    end_time = time.time()
    print(f"most similar to {processed_sentence}: {most_similar} time taken for inference: {end_time - start_time} sec")
