from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

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
        sf.write(self.audio_file_path, self.audio_data, self.sampling_rate)
        return self.audio_file_path

    def convert_format(self, output_format="mp3"):
        output_file_path = self.audio_file_path.with_suffix(f".{output_format}")
        self._process_audio(
            sf.write, output_file_path, self.audio_data, self.sampling_rate
        )
        return output_file_path

    def plot_mel_spectrogram(self):
        """Load the audio file and plot its log-mel spectrogram (numpy +
        matplotlib only).

        Mirrors openai-whisper's `log_mel_spectrogram` look: 80 Slaney mel
        bins, power->mel->log10, viridis, origin lower.
        """
        if self.audio_data is None:
            self._load_audio()
        if self.audio_data is None:
            print(f"{self.audio_file_path} does not exist.")
            return
        audio = self.audio_data
        if audio.ndim > 1:  # downmix multi-channel (stereo) to mono
            audio = audio.mean(axis=1)
        n_fft, hop_length = 400, 160
        n_mels = 80
        win = np.hanning(n_fft)
        n_frames = max(0, 1 + (len(audio) - n_fft) // hop_length)
        if n_frames < 1:
            print(
                f"{self.audio_file_path}: too short for a mel spectrogram (< {n_fft} samples)."
            )
            return
        frames = np.array(
            [
                audio[i * hop_length : i * hop_length + n_fft] * win
                for i in range(n_frames)
            ]
        )
        power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # (frames, bins)
        filters = self._mel_filters(n_fft, n_mels, self.sampling_rate)  # (mels, bins)
        log_mel = np.log10(np.maximum(filters @ power.T, 1e-10))  # (mels, frames)
        # Same clamp + rescale as openai-whisper's log_mel_spectrogram
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        log_mel = np.asarray(log_mel, dtype=float).squeeze()
        if log_mel.ndim != 2:
            raise ValueError(
                f"unexpected mel spectrogram shape {log_mel.shape} (expected 2D)"
            )
        plt.figure(figsize=(10, 6))
        plt.imshow(log_mel, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Log Mel Magnitude")
        plt.title("Mel Spectrogram")
        plt.xlabel("Frame")
        plt.ylabel("Mel Band")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _mel_filters(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
        """Slaney-style mel filterbank (n_mels, n_fft//2+1) matching
        librosa's defaults as used by openai-whisper (htk=False, norm="slaney")."""
        n_bins = n_fft // 2 + 1
        freqs = np.linspace(0.0, sr / 2.0, n_bins)
        mel_min, mel_max = (
            AudioFileProcessor._hz_to_mel(0.0),
            AudioFileProcessor._hz_to_mel(sr / 2.0),
        )
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = AudioFileProcessor._mel_to_hz(mel_pts)
        filters = np.zeros((n_mels, n_bins))
        for m in range(n_mels):
            lo, cen, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            weights = np.minimum((freqs - lo) / (cen - lo), (hi - freqs) / (hi - cen))
            filters[m] = np.maximum(0.0, weights)
        # Slaney normalisation: unit area per band (librosa norm="slaney")
        filters *= (2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels]))[:, np.newaxis]
        return filters

    @staticmethod
    def _hz_to_mel(hz) -> np.ndarray:
        """Slaney (librosa/whisper) hz -> mel scale."""
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        hz = np.asarray(hz, dtype=float)
        mels = hz / f_sp
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mels = min_log_mel + np.log(hz / min_log_hz) / logstep
        return np.where(hz >= min_log_hz, log_mels, mels)

    @staticmethod
    def _mel_to_hz(mels) -> np.ndarray:
        """Slaney (librosa/whisper) mel -> hz scale."""
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        mels = np.asarray(mels, dtype=float)
        freqs = f_sp * mels
        return np.where(
            mels >= min_log_mel,
            min_log_hz * np.exp(logstep * (mels - min_log_mel)),
            freqs,
        )


if __name__ == "__main__":
    # test AudioProcessor
    audio_file_path = DATA_DIR / "audio.wav"
    self = AudioFileProcessor(audio_file_path, sampling_rate=16000)
    self.record(duration_seconds=5)
    self.play()
    self.plot_waveform()
    self.plot_mel_spectrogram()
    # self.convert_format(output_format='mp3')
