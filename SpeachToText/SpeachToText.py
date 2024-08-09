import time
from faster_whisper import WhisperModel
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["PATH"] += os.environ["PATH"] + ";" +  r"D:\anaconda\envs\voice_command_mode\Lib\site-packages\nvidia\cudnn\bin" + ";" + r"D:\anaconda\envs\voice_command_mode\Lib\site-packages\nvidia\cudnn\bin"
os.environ["PATH"] += os.environ["PATH"] + ";" +  r"D:\anaconda\envs\voice_command_mode\Lib\site-packages\nvidia\cublas\bin" + ";" + r"D:\anaconda\envs\voice_command_mode\Lib\site-packages\nvidia\cublas\bin"
DEFAULT_FILE_PATH = "Assets/AudioFiles/player_recorded_command_temp.wav"

# list of available models
TINY = "tiny.en"
BASE = "base.en"
SMALL = "small.en"
MEDIUM = "medium.en"
LARGE = "large-v3"

DEFAULT_MODEL = MEDIUM

class WhisperSpeachToText:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")

    def transcribe_audio_file(self, audio_file_path=DEFAULT_FILE_PATH):
        segments, _ = self.model.transcribe(audio_file_path, beam_size=5, language="en")
        segments = [segment.text for segment in segments]
        return ' '.join(segments)
