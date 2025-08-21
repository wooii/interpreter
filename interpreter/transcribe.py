import threading
import queue
import time
import datetime
import collections
import warnings
import torch
import ollama
import re
import fire
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
from jiwer import wer, cer
from interpreter import data_folder

warnings.filterwarnings(action="ignore", message="FP16 is not supported on CPU; using FP32 instead")


class SileroVAD:
    def __init__(self, frame_size=512, sample_rate=16000):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                       model='silero_vad',
                                       force_reload=False,
                                       onnx=False)
    def is_speech(self, frame):
        if len(frame) != self.frame_size:
            return False
        frame_tensor = torch.from_numpy(frame).float()
        speech_prob = self.model(frame_tensor, self.sample_rate).item()
        return speech_prob > 0.4


class Translator:
    def __init__(self, model="qwen3:0.6b", target_lang="Chinese"):
        self.model = model
        self.target_lang = target_lang
    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        prompt = f"Translate to {self.target_lang}:\n{text} /no_think"
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            translated = response["response"].strip()
            translated = re.sub(r'<think>.*?</think>', '', translated, flags=re.DOTALL)
            return translated.strip()
        except Exception as e:
            return f"[Translation error: {e}]"


def process_audio_segment(full_segment, sample_rate):
    if np.sqrt(np.mean(full_segment**2)) < 0.001:
        return None
    max_val = np.max(np.abs(full_segment)) + 1e-8
    full_segment = full_segment / max_val
    full_segment = nr.reduce_noise(y=full_segment, sr=sample_rate)
    return full_segment



class SpeechToText:
    def transcribe(self, audio: np.ndarray):
        raise NotImplementedError

    def transcribe_file(self, file_path: str):
        raise NotImplementedError

    def extract_text(self, result):
        raise NotImplementedError

    def format_and_display_transcription(self, all_words, transcript, start_time, translator=None, translate_to=None):
        raise NotImplementedError

    def _color_word(self, word, prob):
        prob = max(0.0, min(1.0, prob))
        if prob < 0.5:
            r = 255
            g = int(2 * prob * 255)
        else:
            r = int((1 - 2 * (prob - 0.5)) * 255)
            g = 255
        b = 0
        return f"\033[38;2;{r};{g};{b}m{word}\033[0m"


class WhisperBackend(SpeechToText):
    def __init__(self, model_size):
        import whisper
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio: np.ndarray):
        return self.model.transcribe(
            audio=audio.astype(np.float32),
            word_timestamps=True,
            temperature=0.0,
            beam_size=1,
            no_speech_threshold=0.5,
            logprob_threshold=-0.3,
            condition_on_previous_text=True,
            fp16=False
        )

    def transcribe_file(self, file_path: str):
        return self.model.transcribe(str(file_path))

    def extract_text(self, result):
        return result["text"].strip()

    def format_and_display_transcription(self, result, transcript, start_time, translator=None, translate_to=None):
        all_words = []
        if isinstance(result, dict) and "segments" in result:
            for seg in result["segments"]:
                if "words" in seg:
                    all_words.extend(seg["words"])

        if not all_words:
            return
        sentence = self.extract_text(result)
        text = " ".join(self._color_word(w["word"].strip(), w.get("probability", 1.0)) for w in all_words if w["word"].strip()).strip()
        if text:
            transcript.append(sentence)
            end = time.time()
            elapsed = datetime.timedelta(seconds=end - start_time)
            dt = (datetime.datetime.min + elapsed)
            time_str = dt.strftime("%M:%S.%f")[:-3]
            if translate_to and translator:
                translated = translator.translate(sentence)
                output = f"[{time_str}] {text} → {translated}"
            else:
                output = f"[{time_str}] {text}"
            print(output)


class PyWhisperCppBackend(SpeechToText):
    def __init__(self, model_size):
        from interpreter.whispercpp import WhisperCppModel
        self.model = WhisperCppModel(model_size,
                                     token_timestamps=True,
                                     max_len=1,
                                     split_on_word=True,
                                     print_progress=False)

    def transcribe(self, audio: np.ndarray):
        return self.model.transcribe(media=audio.astype(np.float32), temperature=0.0)

    def transcribe_file(self, file_path: str):
        return self.model.transcribe(str(file_path))

    def extract_text(self, result):
        return " ".join([i.text for i in result]).strip()

    def format_and_display_transcription(self, result, transcript, start_time, translator=None, translate_to=None):
        if isinstance(result, list):
            all_words = [i.text for i in result]

        if not all_words:
            return

        # Concatenate text for transcript
        sentence = self.extract_text(result)
        # Color words based on probability
        text = " ".join(self._color_word(i.text.strip(), i.probability) for i in result).strip()

        if text:
            transcript.append(sentence)
            end = time.time()
            elapsed = datetime.timedelta(seconds=end - start_time)
            dt = datetime.datetime.min + elapsed
            time_str = dt.strftime("%M:%S.%f")[:-3]

            if translate_to and translator:
                translated = translator.translate(sentence)
                output = f"[{time_str}] {text} → {translated}"
            else:
                output = f"[{time_str}] {text}"
            print(output)


class RealTimeTranscribe:
    def __init__(self, audio_file_path=None, model_size="small", translate_to="Chinese", use_whispercpp=True):
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.translate_to = translate_to
        self.use_whispercpp = use_whispercpp
        self.sample_rate = 16000
        self.frame_size = 512
        self.vad = SileroVAD(self.frame_size, self.sample_rate)
        self.translator = Translator(target_lang=translate_to) if translate_to else None
        if self.use_whispercpp:
            self.stt = PyWhisperCppBackend(model_size)
        else:
            self.stt = WhisperBackend(model_size)
        self._initialize_state()

    def _initialize_state(self):
        self.ring_buffer = collections.deque(maxlen=20)
        self.triggered = False
        self.recorded_frames = []
        self.prev_tail_audio = np.zeros(0, dtype='float32')
        self.q_raw = queue.Queue()
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.transcriber_thread = None
        self.vad_thread = None
        self.running = False
        self.transcript = []
        self.full_recording_list = []
        self.start_time = time.time()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        audio_data = indata.flatten()
        if self.audio_file_path:
            self.full_recording_list.append(audio_data)
        while len(audio_data) >= self.frame_size:
            frame = audio_data[:self.frame_size]
            audio_data = audio_data[self.frame_size:]
            self.q_raw.put(frame)

    def _vad_worker(self):
        while self.running:
            frame = self.q_raw.get()
            if frame is None:
                break
            is_speech = self.vad.is_speech(frame)
            self.ring_buffer.append((frame, is_speech))
            if not self.triggered:
                if sum(s for _, s in self.ring_buffer) > 0.5 * self.ring_buffer.maxlen:
                    self.triggered = True
                    for f, _ in self.ring_buffer:
                        self.recorded_frames.append(f)
                    self.ring_buffer.clear()
            else:
                self.recorded_frames.append(frame)
                if sum(1 for _, s in self.ring_buffer if not s) > 0.9 * self.ring_buffer.maxlen:
                    if self.recorded_frames:
                        segment = np.concatenate(self.recorded_frames)
                        self.q.put(segment.copy())
                    self.triggered = False
                    self.recorded_frames.clear()
                    self.ring_buffer.clear()

    def _transcribe(self):
        while True:
            segment = self.q.get()
            if segment is None:
                break
            full_segment = np.concatenate([self.prev_tail_audio, segment])
            processed_segment = process_audio_segment(full_segment, self.sample_rate)
            if processed_segment is None:
                continue
            result = self.stt.transcribe(processed_segment)
            output = self.stt.format_and_display_transcription(
                result, self.transcript, self.start_time, self.translator, self.translate_to,
            )
            if output:
                print(output)

    def _stop(self):
        self.running = False
        self.q.put(None)
        self.q_raw.put(None)
        if self.transcriber_thread is not None:
            self.transcriber_thread.join()
        if self.vad_thread is not None:
            self.vad_thread.join()
        if self.audio_file_path and self.full_recording_list:
            full_audio = np.concatenate(self.full_recording_list)
            sf.write(self.audio_file_path, full_audio, self.sample_rate)
            print(f"Audio saved to {self.audio_file_path}")

    def run(self):
        print(f"Real-time transcribe (Model: {self.stt.__class__.__name__} {self.model_size})... (Ctrl+C to stop)")
        self.running = True
        self.start_time = time.time()
        self.transcriber_thread = threading.Thread(target=self._transcribe, daemon=True)
        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.transcriber_thread.start()
        self.vad_thread.start()
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1,
                                dtype='float32',
                                callback=self._audio_callback,
                                blocksize=self.frame_size):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self._stop()

    def evaluate(self):
        if self.audio_file_path is None:
            print("No audio_file_path provided for evaluation.")
            return None
        result = self.stt.transcribe_file(str(self.audio_file_path))
        self.reference_transcript = self.stt.extract_text(result)
        self.realtime_transcript = " ".join(self.transcript).strip()
        wer_error = wer(self.reference_transcript.lower(), self.realtime_transcript.lower())
        cer_error = cer(self.reference_transcript.lower(), self.realtime_transcript.lower())
        print(f"Word Error Rate (WER): {wer_error:.2%}")
        print(f"Character Error Rate (CER): {cer_error:.2%}")
        print(f"Reference Transcript: {self.reference_transcript}")
        print(f"Realtime Transcript: {self.realtime_transcript}")
        return {"WER": wer_error, "CER": cer_error}


if __name__ == "__main__":
    self = RealTimeTranscribe(audio_file_path=data_folder / "interpreter" / "streaming_audio.wav",
                              model_size="small.en",
                              translate_to="Chinese",
                              use_whispercpp=False,
                              )

    self.run()

    self.evaluate()
