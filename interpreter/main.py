"""
Created on Wed Jul 17 11:59:45 2024
@author: Chenfeng Chen
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import warnings
from pathlib import Path
import whisper
from jiwer import wer  # pip install jiwer
from interpreter import data_folder
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
    def __init__(self, audio_file_path, model_size="small", sample_rate=16000, segment_duration=3):
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.model = whisper.load_model(model_size)
        self.segment_samples = int(sample_rate * segment_duration)
        self.audio_buffer = np.zeros(0, dtype='float32')
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.transcriber_thread = None
        self.running = False
        self.transcript = []  # Store all transcribed segments

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        with self.lock:
            self.audio_buffer = np.append(self.audio_buffer, indata.flatten())
            while len(self.audio_buffer) >= self.segment_samples:
                segment = self.audio_buffer[:self.segment_samples]
                self.audio_buffer = self.audio_buffer[self.segment_samples:]
                self.q.put(segment.copy())

    def transcriber(self):
        while True:
            segment = self.q.get()
            if segment is None:
                break
            temp_file = self.audio_file_path.with_name("temp_segment.wav")
            sf.write(temp_file, segment, self.sample_rate)
            result = self.model.transcribe(str(temp_file))
            temp_file.unlink(missing_ok=True)
            text = result["text"].strip()
            if text:
                self.transcript.append(text)
                print(text)

    def run(self):
        print(f"Starting real-time transcription (Model: Whisper {self.model_size})... (Ctrl+C to stop)")
        self.running = True
        self.transcriber_thread = threading.Thread(target=self.transcriber, daemon=True)
        self.transcriber_thread.start()
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=self.audio_callback, blocksize=4000):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()

    def stop(self):
        self.running = False
        self.q.put(None)
        if self.transcriber_thread is not None:
            self.transcriber_thread.join()

    def evaluate(self, reference_file: Path):
        with open(reference_file, "r", encoding="utf-8") as f:
            reference_text = f.read().strip()
        hypothesis_text = " ".join(self.transcript)
        error = wer(reference_text, hypothesis_text)
        print(f"Word Error Rate (WER): {error:.2%}")
        return error


# Test the improved class-based function
if __name__ == "__main__":
    audio_file_path = data_folder / "interpreter" / "streaming_audio.wav"
    reference_file = data_folder / "interpreter" / "reference.txt"  # your ground truth transcript
    self = RealTimeTranscribe(audio_file_path, model_size="small", sample_rate=16000, segment_duration=3)
    self.run()
    self.evaluate(reference_file)


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
