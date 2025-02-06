"""


Created on Fri Sep 20 13:46:07 2024
@author: Chenfeng Chen
"""

import whisper
import pandas as pd
import matplotlib.pyplot as plt
from interpreter import client, data_folder

audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"


audio = whisper.load_audio(audio_file_path)

mel = whisper.log_mel_spectrogram(audio)

plt.figure(figsize=(10, 6))
plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Log Mel Spectrogram')
plt.title('Mel Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency Bins')
plt.tight_layout()
plt.show()


model = whisper.load_model("small")

result = model.transcribe(audio)
transcription = result["text"]


print(transcription)



import librosa

array, sampling_rate = librosa.load(librosa.ex("trumpet"))

import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
import numpy as np

dft_input = array[1000:24096]

# calculate the DFT
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)

# get the amplitude spectrum in decibels
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# get the frequency bins
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

plt.figure().set_figwidth(12)
plt.plot(frequency, amplitude_db[1600,:])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")









