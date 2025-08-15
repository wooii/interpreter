"""
Model Comparison Script for Local Speech Recognition

This script allows you to compare the performance of various local speech
recognition models by recording audio and transcribing it with multiple
local models including Whisper and Faster-Whisper.

I have tested orther models like Google Speech Recognition, Sphinx using
speech_recoganision, vosk and others.

As of 2025-07-27, Whisper and Faster-Whisper models consistently deliver the
highest accuracy for local speech recognition, which is why this script
focuses on these models.

However, it seems that the Whisper models are faster than Faster-Whisper
models running on MacOS, and the Whisper small model is quite accurate.

"""

import time
import whisper
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel as FasterWhisperModel
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class CompareModels:
    def __init__(self, whisper_model_sizes=["tiny", "base", "small"],
                 faster_whisper_model_sizes=["base", "small"],
                 data_folder=Path.home() / "Data",
                 audio_file_name="comparison_recording.wav",
                 comparison_results_file_name="comparison_results.txt",):
        self.whisper_model_sizes = whisper_model_sizes
        self.faster_whisper_model_sizes = faster_whisper_model_sizes
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
        self.whisper_models = {}
        for model_size in self.whisper_model_sizes:
            try:
                model = whisper.load_model(model_size)
                self.whisper_models[model_size] = model
                print(f"✓ Whisper {model_size} model loaded")
            except Exception as e:
                print(f"✗ Failed to load Whisper {model_size} model: {e}")

        # Initialize Faster-Whisper models
        self.faster_whisper_models = {}
        for model_size in self.faster_whisper_model_sizes:
            try:
                model = FasterWhisperModel(model_size, device="auto", compute_type="int8")
                self.faster_whisper_models[model_size] = model
                print(f"✓ Faster-Whisper {model_size} model loaded")
            except Exception as e:
                print(f"✗ Failed to load Faster-Whisper {model_size} model: {e}")

    def transcribe_with_whisper(self, model, model_name, language=None):
        """Transcribe audio using a Whisper model"""
        try:
            print(f"Transcribing with {model_name}...")
            start_time = time.time()
            if language:
                result = model.transcribe(str(self.audio_file_path), language=language)
            else:
                result = model.transcribe(str(self.audio_file_path))
            end_time = time.time()
            execution_time = end_time - start_time
            return f"{model_name}: {result['text']} (Time: {execution_time:.2f}s)"
        except Exception as e:
            return f"{model_name}: Error - {str(e)}"

    def transcribe_with_faster_whisper(self, model, model_name, language=None):
        """Transcribe audio using Faster-Whisper model"""
        try:
            print(f"Transcribing with {model_name}...")
            start_time = time.time()
            if language:
                segments, info = model.transcribe(str(self.audio_file_path), language=language, beam_size=1)
            else:
                segments, info = model.transcribe(str(self.audio_file_path), beam_size=1)

            # Join all segments to get the full transcription
            text = "".join([segment.text for segment in segments])
            end_time = time.time()
            execution_time = end_time - start_time
            return f"{model_name}: {text} (Time: {execution_time:.2f}s)"
        except Exception as e:
            return f"{model_name}: Error - {str(e)}"

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

        # Test all models
        results = []

        # Whisper models
        for model_size, model in self.whisper_models.items():
            result = self.transcribe_with_whisper(model, f"Whisper {model_size}")
            results.append(result)
            print(result)

        # Faster-Whisper models
        for model_size, model in self.faster_whisper_models.items():
            result = self.transcribe_with_faster_whisper(model, f"Faster-Whisper {model_size}")
            results.append(result)
            print(result)

        # Save results to file
        with open(self.comparison_results_path, "w", encoding="utf-8") as f:
            f.write("Speech Recognition Model Comparison Results\n")
            f.write("=" * 50 + "\n")
            for result in results:
                f.write(result + "\n")

        print(f"\nResults saved to: {self.comparison_results_path}")
        return results


if __name__ == "__main__":
    self = CompareModels(
        whisper_model_sizes=["tiny", "base", "small", "medium", "turbo"],
        faster_whisper_model_sizes=["base", "small"],
        data_folder=Path.home() / "Data",
        audio_file_name="comparison_recording.wav",
        comparison_results_file_name="comparison_results.txt",
        )
    self.record_audio(duration=10)
    self.compare_models()

