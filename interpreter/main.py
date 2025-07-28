"""
Created on Wed Jul 17 11:59:45 2024
@author: Chenfeng Chen
"""

import whisper
import queue
import threading
import sounddevice as sd
import soundfile as sf
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


class RealTimeTranscribe:
    def __init__(self, audio_file_path, model_size="small", segment_duration=3, sample_rate=16000):
        """
        Initialize the real-time transcriber.
        """
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.frame_size = int(self.sample_rate * self.segment_duration)
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stt = whisper.load_model(self.model_size)
        self.recorder_thread = None

    def audio_recorder(self):
        """Background thread function to record audio continuously"""
        while self.recording:
            audio_data = sd.rec(self.frame_size, samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()
            audio_data = audio_data.flatten()
            self.audio_queue.put(audio_data)

    def start(self):
        """Start real-time transcription."""
        print(f"Starting real-time transcription (Model: Whisper {self.model_size})... (Ctrl+C to stop)")

        self.recording = True
        self.recorder_thread = threading.Thread(target=self.audio_recorder)
        self.recorder_thread.start()
        try:
            while True:
                audio_data = self.audio_queue.get()
                temp_file = self.audio_file_path.with_name("temp_segment.wav")
                sf.write(temp_file, audio_data, self.sample_rate)
                result = self.stt.transcribe(str(temp_file))
                if result['text'].strip():
                    print(f"{result['text']}")
                temp_file.unlink(missing_ok=True)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.recording = False
            self.recorder_thread.join()


# Test the improved class-based function
if __name__ == "__main__":
    audio_file_path = data_folder / "interpreter" / "streaming_audio.wav"
    self = RealTimeTranscribe(audio_file_path)
    self.start()


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
