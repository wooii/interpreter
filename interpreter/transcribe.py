import threading
import queue
import time
import datetime
import collections
import torch
import ollama
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
from jiwer import wer, cer
from interpreter.whispercpp import WhisperCppModel
from interpreter import data_folder


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
    def __init__(self, model_size):
        self.model = WhisperCppModel(model_size,
                                     token_timestamps=True,
                                     max_len=1,
                                     split_on_word=True,
                                     #translate=False,
                                     #language='chinese',
                                     print_progress=False)

    def transcribe(self, audio: np.ndarray):
        return self.model.transcribe(media=audio.astype(np.float32),
                                     temperature=0.0,
                                     )

    def transcribe_file(self, file_path: str):
        return self.model.transcribe(str(file_path))

    def extract_text(self, result):
        return " ".join([i.text for i in result]).strip()


class RealTimeTranscribe:
    def __init__(self, audio_file_path=None, model_size="small", translate_to="Chinese"):
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.translate_to = translate_to
        self.sample_rate = 16000
        self.frame_size = 512
        self.vad = SileroVAD(self.frame_size, self.sample_rate)
        self.translator = Translator(target_lang=translate_to) if translate_to else None
        self.stt = SpeechToText(model_size)
        self._initialize_state()

    def _initialize_state(self):
        self.ring_buffer = collections.deque(maxlen=20)
        self.triggered = False
        self.recorded_frames = []
        self.prev_tail_audio = np.zeros(0, dtype='float32')
        self.q_frame_size_audio = queue.Queue()
        self.q_segmented_audio = queue.Queue()
        self.q_transcript = queue.Queue()
        self.lock = threading.Lock()
        self.transcribe_thread = None
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
            self.q_frame_size_audio.put(frame)

    def _vad_worker(self):
        while self.running:
            frame = self.q_frame_size_audio.get()
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
                        self.q_segmented_audio.put(segment.copy())
                    self.triggered = False
                    self.recorded_frames.clear()
                    self.ring_buffer.clear()

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

    def format_and_display_transcription(self, result):
        if not (isinstance(result, list) and result):
            return
        sentence = self.stt.extract_text(result)
        text = self._format_output(result)
        time_str = self._get_time_str()
        output = f"[{time_str}] {text}"
        self.transcript.append(sentence)
        self._display(output, sentence)

    def _format_output(self, result):
        # Returns colored text for the transcription
        return " ".join(self._color_word(i.text.strip(), i.probability) for i in result).strip()

    def _get_time_str(self):
        end = time.time()
        elapsed = datetime.timedelta(seconds=end - self.start_time)
        dt = datetime.datetime.min + elapsed
        return dt.strftime("%M:%S.%f")[:-3]

    def _display(self, output, sentence):
        if self.translate_to and self.translator:
            print(output, end=" ", flush=True)
            threading.Thread(target=self._translate, args=(sentence,), daemon=True).start()
        else:
            print(output)

    def _translate(self, sentence):
        translate_start = time.time()
        translated = self.translator.translate(sentence)
        time.sleep(5)
        translate_time = time.time() - translate_start
        print(f"→ {translated} ({translate_time:.2f}s)")

    def _transcribe(self):
        while True:
            segment = self.q_segmented_audio.get()
            if segment is None:
                break
            full_segment = np.concatenate([self.prev_tail_audio, segment])
            processed_segment = process_audio_segment(full_segment, self.sample_rate)
            if processed_segment is None:
                continue
            result = self.stt.transcribe(processed_segment)
            self.format_and_display_transcription(result)

    def _stop(self):
        self.running = False
        self.q_segmented_audio.put(None)
        self.q_frame_size_audio.put(None)
        if self.transcribe_thread is not None:
            self.transcribe_thread.join()
        if self.vad_thread is not None:
            self.vad_thread.join()
        if self.audio_file_path and self.full_recording_list:
            full_audio = np.concatenate(self.full_recording_list)
            sf.write(self.audio_file_path, full_audio, self.sample_rate)
            print(f"Audio saved to {self.audio_file_path}")

    def run(self):
        print(f"Real-time transcribe ({self.stt.model.__class__.__name__}: {self.model_size})... (Ctrl+C to stop)")
        self.running = True
        self.start_time = time.time()
        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.vad_thread.start()
        self.transcribe_thread = threading.Thread(target=self._transcribe, daemon=True)
        self.transcribe_thread.start()

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
                              model_size="large-v3-turbo-q5_0",
                              translate_to="Chinese",
                              )

    self.run()

    self.evaluate()
