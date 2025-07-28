from interpreter import AudioFileProcessor, client, data_folder
from translate import Translator


def openai_whisper(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return response


def openai_tts(text, speech_file_path):
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
    response.stream_to_file(speech_file_path)
    return print(f"Saved speech to {speech_file_path}.")


def record_and_transcribe_using_openai_model(audio_file_path):
    print("Starting transcription, please speak...")

    while True:
        audio_processor = AudioFileProcessor(audio_file_path, sampling_rate=16000)
        audio = audio_processor.record(duration_seconds=3)
        transcription = openai_whisper(audio_file_path)
        print(f"{transcription.text}")


# %% test openai api
if __name__ == "__main__":
    # Define file paths for input audio and output speech
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    openai_speech_file_path = data_folder / "interpreter" / "openai_speech_audio.mp3"

    # Transcribe audio using OpenAI Whisper
    transcription = openai_whisper(audio_file_path)
    text = transcription.text
    print(text)

    # Translate the transcribed text to Chinese
    translator = Translator(to_lang='zh')
    translation = translator(text)
    print(translation)

    # Generate speech from the translated text using OpenAI TTS
    openai_tts(translation, openai_speech_file_path)

    # Play and plot the generated speech audio
    self = AudioFileProcessor(openai_speech_file_path, sampling_rate=16000)
    self.play()
    self.plot_waveform()


# %% test record_and_transcribe
if False:
    audio_file_path = data_folder / "interpreter" / "recorded_audio.mp3"
    # test OpenAI model
    record_and_transcribe_using_openai_model(audio_file_path)