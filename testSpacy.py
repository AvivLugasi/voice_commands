from SpeachToText.AudioCapture import Recorder
import time

from TextProcessing.TextProcessor import TextProcessor
import whisper
from Model.BertModel import SentenceEmbedderModel
from Model.Utils import cosine_sim
import json
import numpy as np

# Read the JSON file
with open('Assets/Data/ReadyOrNot/ReadyOrNotCommandsFormatted.json', 'r') as file:
    data = json.load(file)

# List to hold all variations
commands_list = []

# Iterate through the commands groups and collect variations
for group in data.get("commands groups", []):
    for command in group.get("commands list", []):
        variations = command.get("variations", [])
        commands_list.extend(variations)

# Print the list of all variations
print(commands_list)

model = SentenceEmbedderModel(model_name='bert-base-uncased')

# for command in commands_list:
#     embedded_command = model.sentence_to_vector(processed_sentence=command)
#     # Convert the numpy tensor to a string
#     tensor_string = np.array2string(embedded_command, separator=',', formatter={'all': lambda x: str(x)})
#
#     # Create the line with the string and tensor string separated by a colon
#     line_to_write = f"{command}:{tensor_string}"
#
#     # Write the line to a file
#     with open('Assets/Data/ReadyOrNot/CommandsAndEmbedding', 'a') as file:
#         file.write(line_to_write + '\n')


sentence_1 = "breach and clear with shotgun throw cs"
sentence_2 = "breach with shotgun and clear throw cs"
sentence_vectors_np_1 = model.sentence_to_vector(processed_sentence=sentence_1)
sentence_vectors_np_2 = model.sentence_to_vector(processed_sentence=sentence_2)

print(cosine_sim(sentence_vectors_np_1, sentence_vectors_np_2, is_1d = True))

# Load the Whisper Model
start_time = time.time()
whisper_model = whisper.load_model("medium")
end_time = time.time()
print(f"Time taken to load model: {end_time - start_time} seconds")
processor = TextProcessor(remove_stopwords=False)


# processed_sentences = processor.process_text(commands_list)
# print(processed_sentences)


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
    if len(processed_sentence) > 1:
        processed_sentence = ' '.join(processed_sentence)
    print(processed_sentence)

    most_similar = model.find_top_n_similar_sentences(processed_sentence[0], commands_list)
    print(f"most similar to {processed_sentence}: {most_similar}")
