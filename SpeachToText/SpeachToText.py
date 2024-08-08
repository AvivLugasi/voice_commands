import time
import whisper

DEFAULT_FILE_PATH = "Assets/AudioFiles/player_recorded_command_temp.wav"

# list of available models
TINY = "tiny"
BASE = "base"
SMALL = "small"
MEDIUM = "medium"
LARGE = "large"

DEFAULT_MODEL = MEDIUM

class WhisperSpeachToText:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = whisper.load_model(model_name)

    def transcribe_audio_file(self, audio_file_path=DEFAULT_FILE_PATH):
        return self.model.transcribe(audio_file_path, fp16=False, language='en')["text"]


