"""
Model Comparison Script for Local Speech Recognition

This script compares the performance of various local speech
recognition models (Whisper, Faster-Whisper, Whisper.cpp via pywhispercpp and whispercpp).

Tested models include:
- OpenAI Whisper (tiny, base, small, etc.)
- Faster-Whisper (CTranslate2 backend)
- Whisper.cpp via pywhispercpp
- Whisper.cpp via whispercpp

As of 2025-08-20, Whisper.cpp via pywhispercpp seems to be the fastest on Mac M4 16 GB.
"""

import time
import whisper
import faster_whisper
import pywhispercpp
import whispercpp
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


class WhisperModel:
    """Class to handle loading and transcription for OpenAI Whisper models"""

    def __init__(self, model_size):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the OpenAI Whisper model"""
        try:
            self.model = whisper.load_model(self.model_size)
            print(f"✓ Whisper {self.model_size} model loaded")
        except Exception as e:
            print(f"✗ Failed to load Whisper {self.model_size} model: {e}")
            self.model = None

    def transcribe(self, audio_file_path):
        """Transcribe audio using OpenAI Whisper model"""
        if self.model is None:
            return f"Whisper {self.model_size}: Error - Model not loaded"

        try:
            print(f"Transcribing with Whisper {self.model_size}...")
            start_time = time.time()
            result = self.model.transcribe(str(audio_file_path),
                                          temperature=0.0,
                                          beam_size=1)
            text = result["text"]
            execution_time = time.time() - start_time
            return f"Whisper {self.model_size}: {text} (Time: {execution_time:.2f}s)"
        except Exception as e:
            return f"Whisper {self.model_size}: Error - {str(e)}"


class FasterWhisperModel:
    """Class to handle loading and transcription for Faster-Whisper models"""

    def __init__(self, model_size):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the Faster-Whisper model"""
        try:
            self.model = faster_whisper.WhisperModel(self.model_size, device="auto", compute_type="int8")
            print(f"✓ Faster-Whisper {self.model_size} model loaded")
        except Exception as e:
            print(f"✗ Failed to load Faster-Whisper {self.model_size} model: {e}")
            self.model = None

    def transcribe(self, audio_file_path):
        """Transcribe audio using Faster-Whisper model"""
        if self.model is None:
            return f"Faster-Whisper {self.model_size}: Error - Model not loaded"

        try:
            print(f"Transcribing with Faster-Whisper {self.model_size}...")
            start_time = time.time()
            segments, info = self.model.transcribe(str(audio_file_path),
                                                   beam_size=1)
            text = "".join([segment.text for segment in segments])
            execution_time = time.time() - start_time
            return f"Faster-Whisper {self.model_size}: {text} (Time: {execution_time:.2f}s)"
        except Exception as e:
            return f"Faster-Whisper {self.model_size}: Error - {str(e)}"


class PyWhisperCppModel:
    """Class to handle loading and transcription for pywhispercpp models"""

    def __init__(self, model_size):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pywhispercpp model"""
        try:
            self.model = pywhispercpp.model.Model(
                self.model_size,
                n_threads=2,
                token_timestamps=True,
                max_len=1,
                split_on_word=True
            )
            print(f"✓ pywhispercpp {self.model_size} model loaded")
        except Exception as e:
            print(f"✗ Failed to load pywhispercpp {self.model_size} model: {e}")
            self.model = None

    def transcribe(self, audio_file_path):
        """Transcribe audio using pywhispercpp model"""
        if self.model is None:
            return f"pywhispercpp {self.model_size}: Error - Model not loaded"

        try:
            print(f"Transcribing with pywhispercpp {self.model_size}...")
            start_time = time.time()
            segments = self.model.transcribe(str(audio_file_path))
            text = " ".join(seg.text.strip() for seg in segments)
            execution_time = time.time() - start_time
            return f"pywhispercpp {self.model_size}: {text} (Time: {execution_time:.2f}s)"
        except Exception as e:
            return f"pywhispercpp {self.model_size}: Error - {str(e)}"


class WhisperCppModel:
    """Class to handle loading and transcription for whispercpp.py models"""

    def __init__(self, model_size):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the whispercpp.py model"""
        try:
            self.model = whispercpp.Whisper(self.model_size)
            print(f"✓ whispercpp.py {self.model_size} model loaded")
        except Exception as e:
            print(f"✗ Failed to load whispercpp.py {self.model_size} model: {e}")
            self.model = None

    def transcribe(self, audio_file_path):
        """Transcribe audio using whispercpp.py model"""
        if self.model is None:
            return f"whispercpp.py {self.model_size}: Error - Model not loaded"

        try:
            print(f"Transcribing with whispercpp.py {self.model_size}...")
            start_time = time.time()
            result = self.model.transcribe(str(audio_file_path))
            # Use the extract_text method to get the actual transcription
            text = " ".join(self.model.extract_text(result))
            execution_time = time.time() - start_time
            return f"whispercpp.py {self.model_size}: {text} (Time: {execution_time:.2f}s)"
        except Exception as e:
            return f"whispercpp.py {self.model_size}: Error - {str(e)}"


class CompareModels:
    def __init__(self, whisper_model_sizes=["tiny", "base", "small"],
                 faster_whisper_model_sizes=["base", "small"],
                 pywhispercpp_model_sizes=["base", "small"],
                 whispercpp_model_sizes=["base", "small"],
                 data_folder=Path.home() / "Data",
                 audio_file_name="comparison_recording.wav",
                 comparison_results_file_name="comparison_results.txt"):
        self.whisper_model_sizes = whisper_model_sizes
        self.faster_whisper_model_sizes = faster_whisper_model_sizes
        self.pywhispercpp_model_sizes = pywhispercpp_model_sizes
        self.whispercpp_model_sizes = whispercpp_model_sizes
        self.sampling_rate = 16000

        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.audio_file_path = self.data_folder / audio_file_name
        self.comparison_results_path = self.data_folder / comparison_results_file_name

        self.initialize_models()

    def initialize_models(self):
        """Initialize all available speech recognition models"""
        print("Initializing models...")

        # Initialize Whisper models
        self.whisper_models = []
        for model_size in self.whisper_model_sizes:
            model = WhisperModel(model_size)
            self.whisper_models.append(model)

        # Initialize Faster-Whisper models
        self.faster_whisper_models = []
        for model_size in self.faster_whisper_model_sizes:
            model = FasterWhisperModel(model_size)
            self.faster_whisper_models.append(model)

        # Initialize pywhispercpp models
        self.pywhispercpp_models = []
        for model_size in self.pywhispercpp_model_sizes:
            model = PyWhisperCppModel(model_size)
            self.pywhispercpp_models.append(model)

        # Initialize whispercpp.py models
        self.whispercpp_models = []
        for model_size in self.whispercpp_model_sizes:
            model = WhisperCppModel(model_size)
            self.whispercpp_models.append(model)

    def record_audio(self, duration=5):
        """Record audio for a specified duration"""
        print(f"Recording for {duration} seconds... Please speak now.")

        # Record audio
        frames = int(duration * self.sampling_rate)
        audio_data = sd.rec(frames, self.sampling_rate, channels=1)
        sd.wait()  # Wait until recording is finished

        # Save audio to file
        sf.write(str(self.audio_file_path), audio_data, self.sampling_rate)
        print(f"Audio saved to {self.audio_file_path}")

        return audio_data

    def compare_models(self):
        """Compare all available models with the same audio input"""
        print("=" * 50)
        print("Speech Recognition Model Comparison")
        print("=" * 50)

        results = []

        for model in self.whisper_models:
            result = model.transcribe(self.audio_file_path)
            results.append(result)
            print(result)

        for model in self.faster_whisper_models:
            result = model.transcribe(self.audio_file_path)
            results.append(result)
            print(result)

        for model in self.pywhispercpp_models:
            result = model.transcribe(self.audio_file_path)
            results.append(result)
            print(result)

        for model in self.whispercpp_models:
            result = model.transcribe(self.audio_file_path)
            results.append(result)
            print(result)

        with open(self.comparison_results_path, "w", encoding="utf-8") as f:
            f.write("Speech Recognition Model Comparison Results\n")
            f.write("=" * 50 + "\n")
            for result in results:
                f.write(result + "\n")

        print(f"\nResults saved to: {self.comparison_results_path}")
        return results


if __name__ == "__main__":
    self = CompareModels(
        whisper_model_sizes=["tiny", "base", "small"],
        faster_whisper_model_sizes=["base", "small"],
        pywhispercpp_model_sizes=["base", "small"],
        whispercpp_model_sizes=["base", "small"],
        data_folder=Path.home() / "Data",
        audio_file_name="comparison_recording.wav",
        comparison_results_file_name="comparison_results.txt",
    )

    self.record_audio(duration=10)
    self.compare_models()
