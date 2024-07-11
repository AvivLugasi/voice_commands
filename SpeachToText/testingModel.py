from AudioCapture import Recorder
import time

from TextProcessing.TextProcessor import TextProcessor
import whisper

# Load the Whisper Model
start_time = time.time()
whisper_model = whisper.load_model("medium")
end_time = time.time()
print(f"Time taken to load model: {end_time - start_time} seconds")
processor = TextProcessor(remove_stopwords=False)

corpus = '''
Gold open and clear. Open and clear use shotgun.
Arrest suspect. Zip the suspect.
Stack up.
Breach and clear use c2.
Breaching using shotgun.
Police dont move.
hand's up LSPD.
LSPD swat, dont move!
gold team open and clear.
open and clear use explosive.
use flash bang and clear.
breach with c2 use flashbang.
open with c2 use flashbang.
make entry use c2.
make entry use c2 and stun grenade.
open with c2 throw tear gas.
cover the area.
move there.
follow me.
bag it.
recover evidence.
stack up the door.
cover the door.
'''
processed_sentences = processor.process_text(corpus)
print(processed_sentences)

from Model.Model import Model
start_time = time.time()
model = Model()
end_time = time.time()
print(f"Time taken to load model: {end_time - start_time} seconds")
# start_time = time.time()
# model.train_model(processed_sentences, epochs=100)
# end_time = time.time()
# print(f"Time taken to train model: {end_time - start_time} seconds")
while True:
    start_time = time.time()
    recorder = Recorder()
    recorder.capture_voice_command(record_method_param="space")
    end_time = time.time()
    print(f"Time taken to recording: {end_time - start_time} seconds")
    start_time = time.time()
    # Transcribe the Ogg file
    result = whisper_model.transcribe("../Assets/AudioFiles/player_recorded_command_temp.wav", fp16=False, language='en')
    end_time = time.time()
    print(f"Time taken to transcribe text: {end_time - start_time} seconds")
    # Print the transcription result
    print(result["text"])
    text = result["text"]
    processed_sentence = processor.process_text(text)
    if len(processed_sentence) > 1:
        processed_sentence = ' '.join(processed_sentence)
    print(processed_sentence)
    print(f"most similar to {processed_sentence}: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processed_sentence[0], n=1)}")
    end_time = time.time()
    print(f"Time taken to predict: {end_time - start_time} seconds")