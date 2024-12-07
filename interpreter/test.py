"""


Created on Fri Sep 20 13:46:07 2024
@author: Chenfeng Chen
"""

import whisper
import pandas as pd
from interpreter.config import client, data_folder

audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"


audio = whisper.load_audio(audio_file_path)

mel = whisper.log_mel_spectrogram(audio)

pd.plot(mel)

model = whisper.load_model("small")

result = model.transcribe(audio)
transcription = result["text"]


print(transcription)

