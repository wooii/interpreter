from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import wavio
import whisper

from interpreter import DATA_DIR


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
            wavio.write(
                self.audio_file_path_str,
                self.audio_data,
                self.sampling_rate,
                sampwidth=2,
            )
            return self.audio_file_path
        else:
            return self.convert_format(output_format=self.audio_format)

    def convert_format(self, output_format="mp3"):
        output_file_path = self.audio_file_path.with_suffix(f".{output_format}")
        self._process_audio(
            sf.write, output_file_path, self.audio_data, self.sampling_rate
        )
        return output_file_path

    def plot_mel_spectrogram(self):
        """Load the audio file and plot its mel spectrogram using Whisper."""
        audio = whisper.load_audio(str(self.audio_file_path))
        mel = whisper.log_mel_spectrogram(audio)
        plt.figure(figsize=(10, 6))
        plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Log Mel Spectrogram")
        plt.title("Mel Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # test AudioProcessor
    audio_file_path = DATA_DIR / "audio.mp3"
    self = AudioFileProcessor(audio_file_path, sampling_rate=16000)
    self.record(duration_seconds=5)
    self.play()
    self.plot_waveform()
    self.plot_mel_spectrogram()
    # self.convert_format(output_format='mp3')
    # self = AudioDataProcessor(audio_data=audio_sample["array"], sampling_rate=16000)
