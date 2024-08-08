from SpeachToText.AudioCapture import Recorder
from SpeachToText import SpeachToText
import time
from TextProcessing.TextProcessor import TextProcessor
from TextProcessing.PostProcessing import *
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



data_io = DataIO(formatted_commands_file_path="Assets/Data/ReadyOrNot/ReadyOrNotCommandsFormatted.json")
model = SequenceClassificationModel()

words_set = data_io.get_commands_variations_words_set()
# words_embedding_dict = {}
# for word in words_set:
#     word_vector = model.sentence_to_vector(word)
#     words_embedding_dict[word] = word_vector
#
# data_io.write_variations_embedding_dict(words_embedding_dict,
#                                         file_path="Assets/Data/ReadyOrNot/WordsCorpusEmbeddings")

# commands_variations = data_io.get_commands_variations()
# variations_embedding_dict = {}
# for variation in commands_variations:
#     sentence_vectors = model.sentence_to_vector(variation)
#     variations_embedding_dict[variation] = sentence_vectors
#
# data_io.write_variations_embedding_dict(variations_embedding_dict)

variations_embedding_dict = data_io.load_variations_embedding_dict()
words_embedding_dict = data_io.load_variations_embedding_dict(file_path="Assets/Data/ReadyOrNot/WordsCorpusEmbeddings")
variations_keys_sequence_dict = data_io.get_variations_keys_sequence_dict()

# words_phonetic_codes_dict = calc_corpus_phonetic_codes(words_set)
# data_io.write_words_phonetic_codes_dict(words_phonetic_codes_dict,
#                                         file_path="Assets/Data/ReadyOrNot/WordsMetaphonePhoneticCodes")
words_phonetic_codes_dict = data_io.load_words_phonetic_codes_dict(file_path="Assets/Data/ReadyOrNot/WordsMetaphonePhoneticCodes")
words_phonetic_codes_dict = data_io.load_words_phonetic_codes_dict(file_path="Assets/Data/ReadyOrNot/WordsSoundexPhoneticCodes")

print(find_closest_word("charger", words_phonetic_codes_dict, method="soundex"))
print(find_closest_word("stung", words_phonetic_codes_dict, method="soundex"))
print(find_closest_word("rich", words_phonetic_codes_dict, method="soundex"))
print(find_closest_word("bleach", words_phonetic_codes_dict, method="soundex"))
print(find_closest_word("freshbed", words_phonetic_codes_dict, method="soundex"))
print(find_closest_word("dough", words_phonetic_codes_dict, method="soundex"))
# from metaphone import doublemetaphone
# import Levenshtein
# code1 = doublemetaphone("stung")
# code2 = doublemetaphone("stun")
# code3 = doublemetaphone("stinger")
# print(code1)
# print(code2)
# print(code3)

# words_phonetic_codes_dict = calc_corpus_phonetic_codes(words_set, method="soundex")
# data_io.write_words_phonetic_codes_dict(words_phonetic_codes_dict,
#                                         file_path="Assets/Data/ReadyOrNot/WordsSoundexPhoneticCodes")
# words_phonetic_codes_dict = data_io.load_words_phonetic_codes_dict(file_path="Assets/Data/ReadyOrNot/WordsSoundexPhoneticCodes")
# print(words_phonetic_codes_dict)

sentence_vectors_np_1 = model.sentence_to_vector('breach')
sentence_vectors_np_2 = model.sentence_to_vector(processed_sentence='open')

print(cosine_sim(sentence_vectors_np_1, sentence_vectors_np_2, is_1d = True))

#print(cosine_sim(words_embedding_dict['charger'], model.sentence_to_vector(processed_sentence='charges'), is_1d = True))
print(find_top_n_similar_sentences(model,'charger',words_embedding_dict))
print(find_top_n_similar_sentences(model,'bleach',words_embedding_dict))
print(find_top_n_similar_sentences(model,'charred',words_embedding_dict))
print(find_top_n_similar_sentences(model,'rich',words_embedding_dict))
# Load the Whisper Model
start_time = time.time()
whisper_model = SpeachToText.WhisperSpeachToText(model_name=SpeachToText.TINY)
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
    text = whisper_model.transcribe_audio_file()
    end_time = time.time()
    print(f"Time taken to transcribe text: {end_time - start_time} seconds")
    # Print the transcription result
    print(text)
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