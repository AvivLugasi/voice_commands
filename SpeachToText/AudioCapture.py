import time
import pyaudio
import wave
import keyboard
import whisper
PLAYER_COMMAND_RECORD = "Assets/AudioFiles/player_recorded_command_temp.wav"
RECORD_DURATION = 8
SAMPLES_PER_SECONDS = 16000
IS_STREAM_FROM_INPUT = True
RECORD_KEY_PRESSED = 'v'


class Recorder:
    def __init__(self,
                 chunk=1024,
                 sample_format=pyaudio.paInt16,
                 channels=1,
                 fs=SAMPLES_PER_SECONDS):
        # Parameters for recording
        self.chunk = chunk  # Record in chunks of 1024 samples
        self.sample_format = sample_format  # 16 bits per sample
        self.channels = channels
        self.fs = fs  # Record at SAMPLES_PER_SECONDS(16000 by default) samples per second

        # Init interface for port audio
        self.port_audio = pyaudio.PyAudio()
        self.stream = None

    def open_connection(self):
        self.stream = self.port_audio.open(format=self.sample_format,
                                           channels=self.channels,
                                           rate=self.fs,
                                           frames_per_buffer=self.chunk,
                                           input=IS_STREAM_FROM_INPUT)

    def close_connection(self):
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.port_audio.terminate()

    def record_by_duration(self, duration=RECORD_DURATION):
        frames = []  # Initialize array to store frames
        print("recording")
        # Store data in chunks for the specified duration
        for _ in range(0, int(self.fs / self.chunk * duration)):
            data = self.stream.read(self.chunk)
            frames.append(data)

        print("finished recording")
        return frames

    def record_by_key_pressed(self, key=RECORD_KEY_PRESSED):
        frames = []  # Initialize array to store frames
        keyboard.wait(key)
        print("recording")
        try:
            # Record while the key is pressed
            while keyboard.is_pressed(key):
                data = self.stream.read(self.chunk)
                frames.append(data)
        except KeyboardInterrupt:
            # Handle interrupt for gracefully stopping the recording
            pass
        except OSError as e:
            pass
        except Exception as e:
            # Handle any exception
            print(f"An error occurred: {e}")
        print("finished recording")
        return frames

    def save_record(self, frames,
                    output_filename=PLAYER_COMMAND_RECORD):
        # Save the recorded data as a WAV file
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.port_audio.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def capture_voice_command(self, record_method_param,
                              output_filename=PLAYER_COMMAND_RECORD):
        self.open_connection()
        print("ready to record")
        if isinstance(record_method_param, str):
            frames = self.record_by_key_pressed(key=record_method_param)
        # Check if the variable is a number (int or float)
        elif isinstance(record_method_param, (int, float)):
            frames = self.record_by_duration(duration=record_method_param)
        else:
            raise ValueError("record_method_param must be an integer for duration method),\n"
                             "or a string that indicate a key for key pressed method")
        self.close_connection()

        self.save_record(frames=frames, output_filename=output_filename)


# start_time = time.time()
# recorder = Recorder()
# recorder.capture_voice_command(record_method_param=RECORD_KEY_PRESSED)
# end_time = time.time()
# print(f"Time taken to recording: {end_time - start_time} seconds")
# # Load the Whisper Model
# start_time = time.time()
# model = whisper.load_model("medium")
# end_time = time.time()
# print(f"Time taken to load model: {end_time - start_time} seconds")
# start_time = time.time()
# # Transcribe the Ogg file
# result = model.transcribe("../Assets/AudioFiles/player_recorded_command_temp.wav", fp16=False, language='en')
# end_time = time.time()
# print(f"Time taken to transcribe text: {end_time - start_time} seconds")
# # Print the transcription result
# print(result["text"])
# text = result["text"]
# processor = TextProcessor()
# processed_sentence = processor.process_text(text)
# text = '''
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
# '''
# processed_sentences = processor.process_text(text)
# print(processed_sentence)
# model = Model()
# print(
#     f"most similar to {processed_sentence}: {model.find_top_n_similar_sentences(processed_sentences=processed_sentences, input_sentence=processed_sentence[0], n=1)}")
