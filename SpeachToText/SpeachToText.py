import time
import whisper


start_time = time.time()
model = whisper.load_model("medium")
end_time = time.time()
print(f"Time taken to load model: {end_time - start_time} seconds")
start_time = time.time()
# Transcribe the Ogg file
result = model.transcribe("../Assets/AudioFiles/player_recorded_command_temp.wav", fp16=False, language='en')
end_time = time.time()
print(f"Time taken to transcribe text: {end_time - start_time} seconds")
# Print the transcription result
print(result["text"])
