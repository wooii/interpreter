"""
Created on Wed Jul 17 11:59:45 2024
@author: Chenfeng Chen
"""

import whisper
import wavio
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from gtts import gTTS
from translate import Translator
from pathlib import Path
from interpreter import client, data_folder


class AudioDataProcessor:
    def __init__(self, audio_data: np.array, sampling_rate: int):
        self.audio_data = audio_data
        self.sampling_rate = sampling_rate

    def play(self):
        return sd.play(data=self.audio_data, samplerate=self.sampling_rate)

    def plot(self):
        duration_seconds = len(self.audio_data) / self.sampling_rate
        time = np.linspace(0, duration_seconds, len(self.audio_data))
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.audio_data)
        plt.title("Audio Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
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


class SpeechToText:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_file_path):
        return self.model.transcribe(str(audio_file_path))


class TextTranslator:

    def __init__(self, target_language='zh'):
        self.translator = Translator(to_lang=target_language)

    def translate(self, text):
        return self.translator.translate(text)


def openai_whisper(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return response


def openai_tts(text, speech_file_path):
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
    response.stream_to_file(speech_file_path)
    return print(f"Saved speech to {speech_file_path}.")


def record_and_transcribe(audio_file_path):
    print("Starting transcription, please speak...")

    while True:
        audio_processor = AudioFileProcessor(audio_file_path, sampling_rate=16000)
        audio = audio_processor.record(duration_seconds=3)
        transcription = openai_whisper(audio_file_path)
        print(f"{transcription.text}")


if __name__ == "__main__":
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    openai_speech_file_path = data_folder / "interpreter" / "openai_speech_audio.mp3"
    local_speech_file_path = data_folder / "interpreter" / "local_speech_audio.mp3"


# %% test AudioProcessor
if False:

    self = AudioFileProcessor(audio_file_path, sampling_rate=16000)
    self.record(duration_seconds=3)
    # self.convert_format(output_format='mp3')
    self.play()
    self.plot()
    # self = AudioDataProcessor(audio_data=audio_sample["array"], sampling_rate=16000)


# %% test openai api
if False:

    transcription = openai_whisper(audio_file_path)
    text = transcription.text
    print(text)

    self = TextTranslator(target_language='zh')
    translation = self.translate(text)
    print(translation)

    openai_tts(translation, openai_speech_file_path)

    self = AudioFileProcessor(openai_speech_file_path, sampling_rate=16000)
    self.play()
    self.plot()


# %% test local models
if False:

    self = SpeechToText(model_name="base")  # tiny, base, small, medium, large
    transcription = self.transcribe(audio_file_path)

    text = transcription["text"]
    print(text)
    self = TextTranslator(target_language='zh')
    translation = self.translate(text)
    print(translation)

    language = 'zh'
    speech = gTTS(text=translation, lang=language, slow=False)
    speech.save(local_speech_file_path)

    self = AudioFileProcessor(local_speech_file_path, sampling_rate=16000)
    self.play()
    self.plot()


# %% test record_and_transcribe
if False:
    record_and_transcribe(audio_file_path)