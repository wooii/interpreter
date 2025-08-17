import os
import yaml
import whisper
import wavio
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI


class AudioDataProcessor:
    def __init__(self, audio_data: np.ndarray, sampling_rate: int):
        self.audio_data = audio_data
        self.sampling_rate = sampling_rate

    def play(self):
        return sd.play(data=self.audio_data, samplerate=self.sampling_rate)

    def plot_waveform(self):
        duration_seconds = len(self.audio_data) / self.sampling_rate
        time = np.linspace(0, duration_seconds, len(self.audio_data))
        plt.figure(figsize=(10, 4))
        plt.plot(time, self.audio_data)
        plt.title("Audio Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()


class AudioFileProcessor(AudioDataProcessor):
    def __init__(self, audio_file_path: Path, sampling_rate: int = 48000):
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.sampling_rate = sampling_rate
        self.audio_format = self.audio_file_path.suffix[1:]
        self.audio_file_path_str = str(self.audio_file_path.resolve())
        self._load_audio()

    def _load_audio(self):
        if self.audio_file_path.exists():
            self.audio_data, self.sampling_rate = sf.read(self.audio_file_path)

    def _process_audio(self, callback, *args, **kwargs):
        if self.audio_data is None:
            self._load_audio()
        if self.audio_data is not None:
            return callback(*args, **kwargs)
        else:
            print(f"{self.audio_file_path} does not exist.")
            return None

    def record(self, duration_seconds=5):
        frames = int(duration_seconds * self.sampling_rate)
        self.audio_data = sd.rec(frames, self.sampling_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        if self.audio_format == "wav":
            wavio.write(self.audio_file_path_str, self.audio_data, self.sampling_rate, sampwidth=2)
            return self.audio_file_path
        else:
            return self.convert_format(output_format=self.audio_format)

    def convert_format(self, output_format='mp3'):
        output_file_path = self.audio_file_path.with_suffix(f'.{output_format}')
        self._process_audio(sf.write, output_file_path, self.audio_data, self.sampling_rate)
        return output_file_path

    def plot_mel_spectrogram(self):
        """Load the audio file and plot its mel spectrogram using Whisper."""
        audio = whisper.load_audio(self.audio_file_path)
        mel = whisper.log_mel_spectrogram(audio)
        plt.figure(figsize=(10, 6))
        plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log Mel Spectrogram')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency Bins')
        plt.tight_layout()
        plt.show()



class SpeechToText:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file_path):
        return self.model.transcribe(str(audio_file_path))


class TextTranslator:

    def __init__(self, target_language='zh'):
        self.translator = Translator(to_lang=target_language)

    def translate(self, text):
        return self.translator.translate(text)


# Avoid duplicate OpenMP runtime conflict.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Specify the data_folder
data_folder = Path.home() / "Data"
if not data_folder.exists():
    raise EnvironmentError(f"{data_folder} does not exist.")

# Load API keys
with open(data_folder / "api/api_keys_for_ai.yaml", "r") as keys_file:
    keys = yaml.safe_load(keys_file)

client = OpenAI(api_key=keys["openai_api_key"])

# OpenAI api pricing
with open("interpreter/openai_api_pricing.yaml", "r") as file:
    openai_api_pricing = yaml.safe_load(file)

# %% List all openai models
if False:
    models_all = client.models.list()
    models_all_id = sorted([i.id for i in models_all])
    openai_api_pricing


# %% test AudioProcessor
if False:
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    self = AudioFileProcessor(audio_file_path, sampling_rate=16000)
    self.record(duration_seconds=5)
    self.play()
    self.plot_waveform()
    self.plot_mel_spectrogram()
    # self.convert_format(output_format='mp3')
    # self = AudioDataProcessor(audio_data=audio_sample["array"], sampling_rate=16000)


# %% test local models
if False:
    # Record audio and save to file
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    self = AudioFileProcessor(audio_file_path, sampling_rate=16000)
    self.record(duration_seconds=5)
    self.play()
    self.plot_waveform()
    self.plot_mel_spectrogram()

    # Transcribe recorded audio using Whisper local model
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    stt = SpeechToText(model_name="small")  # tiny, base, small, medium, large, turbo
    transcription = stt.transcribe(audio_file_path)
    text = transcription["text"]
    print(text)

    # Translate the transcribed text to Chinese
    translator = TextTranslator(target_language='zh')
    translation = translator.translate(text)
    print(translation)

    # Convert the translated text to speech and play it
    local_speech_file_path = data_folder / "interpreter" / "local_speech_audio.mp3"
    language = 'zh'
    speech = gTTS(text=translation, lang=language, slow=False)
    speech.save(local_speech_file_path)
    self = AudioFileProcessor(local_speech_file_path, sampling_rate=16000)
    self.play()
    self.plot_waveform()
