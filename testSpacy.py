from SpeachToText.AudioCapture import Recorder
import time
from TextProcessing.TextProcessor import TextProcessor
import whisper
from Model.BertModel import SentenceEmbedderModel, find_top_n_similar_sentences, SequenceClassificationModel
from Model.Utils import cosine_sim
import json
import numpy as np
from DataHandling.DataIO import DataIO

import pyautogui

def perform_key_sequence(key_sequence:str):
    if key_sequence is not None:
        splited_key_sequence = key_sequence.split("+")
        for key in splited_key_sequence:
            if key == "mmb":
                pyautogui.mouseDown(button='middle')
                time.sleep(0.1)
                pyautogui.mouseUp(button='middle')
            else:
                pyautogui.press(key)
                time.sleep(0.1)



data_io = DataIO(formatted_commands_file_path="Assets/Data/ReadyOrNot/ReadyOrNotCommandsFormatted.json21")
model = SequenceClassificationModel()

# commands_variations = data_io.get_commands_variations()
# variations_embedding_dict = {}
# for variation in commands_variations:
#     sentence_vectors = model.sentence_to_vector(variation)
#     variations_embedding_dict[variation] = sentence_vectors
#
# data_io.write_variations_embedding_dict(variations_embedding_dict)

variations_embedding_dict = data_io.load_variations_embedding_dict()
variations_keys_sequence_dict = data_io.get_variations_keys_sequence_dict()


sentence_vectors_np_1 = model.sentence_to_vector('c2')
sentence_vectors_np_2 = model.sentence_to_vector(processed_sentence='charges')

print(cosine_sim(sentence_vectors_np_1, sentence_vectors_np_2, is_1d = True))

# Load the Whisper Model
start_time = time.time()
whisper_model = whisper.load_model("base")
end_time = time.time()
print(f"Time taken to load model: {end_time - start_time} seconds")
processor = TextProcessor(remove_stopwords=False)

while True:
    start_time = time.time()
    recorder = Recorder()
    recorder.capture_voice_command(record_method_param="v")
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
    command, similarity = find_top_n_similar_sentences(model,processed_sentence,variations_embedding_dict)
    end_time = time.time()
    print(f"most similar to {processed_sentence}: {command[0]} : {similarity[0]} time taken for inference: {end_time - start_time} sec")
    print(f"key sequence: {variations_keys_sequence_dict[command[0]]}")
    perform_key_sequence(variations_keys_sequence_dict[command[0]])