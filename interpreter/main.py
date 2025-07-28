"""
Created on Wed Jul 17 11:59:45 2024
@author: Chenfeng Chen
"""

import whisper
from gtts import gTTS
from translate import Translator
from interpreter import AudioFileProcessor, data_folder
import warnings
warnings.filterwarnings(
    action="ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
    )


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


def record_and_transcribe_using_local_model(audio_file_path):
    print("Starting transcription, please speak...")
    stt = SpeechToText(model_name="small")
    while True:
        audio_processor = AudioFileProcessor(audio_file_path, sampling_rate=16000)
        audio_processor.record(duration_seconds=3)

        transcription = stt.transcribe(audio_file_path)
        print(f"{transcription["text"]}")


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


# %% test record_and_transcribe
if False:
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    # test local model
    record_and_transcribe_using_local_model(audio_file_path)
