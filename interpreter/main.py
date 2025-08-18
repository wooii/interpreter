import whisper
import threading
import queue
import time
import warnings
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from jiwer import wer, cer
from interpreter import data_folder
from transformers import MarianMTModel, MarianTokenizer


class OfflineTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-zh"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        batch = self.tokenizer([text], return_tensors="pt", padding=True)
        gen = self.model.generate(**batch)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


warnings.filterwarnings(
    action="ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)


def color_word_gradient(word, prob):
    """
    Map probability (0.0-1.0) to RGB color from red -> yellow -> green.
    """
    prob = max(0.0, min(1.0, prob))  # clamp to [0,1]

    if prob < 0.5:
        # Red to Yellow
        r = 255
        g = int(2 * prob * 255)
    else:
        # Yellow to Green
        r = int((1 - 2 * (prob - 0.5)) * 255)
        g = 255
    b = 0

    return f"\033[38;2;{r};{g};{b}m{word}\033[0m"


class RealTimeTranscribe:
    def __init__(self,
                 audio_file_path=None,
                 model_size="small",
                 sample_rate=16000,
                 min_segment_duration=3,
                 blocksize=2000,
                 word_prob_threshold=0.85,
                 translate_enabled=True):
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.blocksize = blocksize
        self.segment_duration = min_segment_duration
        self.word_prob_threshold = word_prob_threshold
        self.translate_enabled = translate_enabled  # store flag
        self.model = whisper.load_model(model_size)
        self.segment_samples = self._duration_to_samples(self.segment_duration)
        self.audio_buffer = np.zeros(0, dtype='float32')
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.transcriber_thread = None
        self.running = False
        self.transcript = []
        self.prev_tail_audio = np.zeros(0, dtype='float32')
        self.full_recording = np.zeros(0, dtype='float32')
        self.segments_processed = 0
        self.translator = OfflineTranslator()

    def _duration_to_samples(self, duration):
        samples = int(duration * self.sample_rate)
        # Round up to nearest multiple of blocksize
        remainder = samples % self.blocksize
        if remainder > 0:
            samples += self.blocksize - remainder
        return samples

    def transcriber(self):
        cut_overlap = 0.1
        start = time.time()
        while True:
            segment = self.q.get()
            if segment is None:
                break

            full_segment = np.concatenate([self.prev_tail_audio, segment])
            if np.sqrt(np.mean(full_segment**2)) < 0.001:
                continue

            t0 = time.time()
            result = self.model.transcribe(
                audio=full_segment.astype(np.float32),
                word_timestamps=True,
                temperature=0.0,
                no_speech_threshold=0.9,
                logprob_threshold=-0.3,
                condition_on_previous_text=True,
            )
            t1 = time.time()
            transcribe_time = t1 - t0

            # Start adjusting after first segment
            if self.segments_processed > 0:
                # Make duration proportional to transcribe_time
                new_duration = max(self.min_segment_duration, transcribe_time)
                self.segment_duration = new_duration
                self.segment_samples = self._duration_to_samples(self.segment_duration)

            self.segments_processed += 1

            # Process words as before
            if not result.get("segments"):
                continue
            all_words = []
            for seg in result["segments"]:
                if "words" not in seg:
                    continue
                if seg.get("no_speech_prob", 0) > 0.9 and seg.get("avg_logprob", -1) < -0.3:
                    continue
                all_words.extend(seg["words"])
            if not all_words:
                continue

            if all_words[-1]["probability"] >= self.word_prob_threshold:
                last_end_time = all_words[-1]["end"]
            elif len(all_words) > 1:
                last_end_time = all_words[-2]["end"]
                all_words = all_words[:-1]
            else:
                last_end_time = all_words[-1]["end"]

            cut_sample_idx = int(max(0, last_end_time - cut_overlap) * self.sample_rate)
            self.prev_tail_audio = full_segment[cut_sample_idx:]

            text = " ".join(color_word_gradient(w["word"].strip(), w["probability"])
                            for w in all_words if w["word"].strip()).strip()
            if text:
                # Save raw English transcript
                sentence = " ".join(w["word"].strip() for w in all_words if w["word"].strip())
                self.transcript.append(sentence)

                # Offline translation
                translated = self.translator.translate(sentence)

                end = time.time()
                if self.translate_enabled:  # <-- conditional
                    translated = self.translator.translate(sentence)
                    print(f"[{end - start:.3f}s] {text} → {translated}")
                else:
                    print(f"[{end - start:.3f}s] {text}")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        with self.lock:
            audio_data = indata.flatten()
            # Save all recorded audio if path is provided
            if self.audio_file_path:
                self.full_recording = np.append(self.full_recording, audio_data)
            self.audio_buffer = np.append(self.audio_buffer, audio_data)
            while len(self.audio_buffer) >= self.segment_samples:
                segment = self.audio_buffer[:self.segment_samples]
                self.audio_buffer = self.audio_buffer[self.segment_samples:]
                self.q.put(segment.copy())

    def run(self):
        print(f"Real-time transcribe (Model: Whisper {self.model_size})... (Ctrl+C to stop)")
        self.running = True
        self.transcriber_thread = threading.Thread(target=self.transcriber, daemon=True)
        self.transcriber_thread.start()
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1,
                                dtype='float32',
                                callback=self.audio_callback,
                                blocksize=self.blocksize):
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
        # Save full recording if path was provided
        if self.audio_file_path and len(self.full_recording) > 0:
            sf.write(self.audio_file_path, self.full_recording, self.sample_rate)
            print(f"Audio saved to {self.audio_file_path}")

    def evaluate(self):
        """Transcribe audio using Whisper model and compare with real-time transcript."""
        if self.audio_file_path is None:
            print("No audio_file_path provided for evaluation.")
            return None

        # Load and transcribe reference audio
        result = self.model.transcribe(str(self.audio_file_path))
        self.reference_transcript = result["text"].strip()

        # Get hypothesis text from real-time transcription
        self.realtime_transcript = " ".join(self.transcript).strip()

        # Calculate WER and CER
        wer_error = wer(self.reference_transcript, self.realtime_transcript)
        cer_error = cer(self.reference_transcript, self.realtime_transcript)

        # Display results
        print(f"Word Error Rate (WER): {wer_error:.2%}")
        print(f"Character Error Rate (CER): {cer_error:.2%}")
        print(f"Reference: {self.reference_transcript}")
        print(f"Realtime Transcript: {self.realtime_transcript}")

        return {"WER": wer_error, "CER": cer_error}


if __name__ == "__main__":
    self = RealTimeTranscribe(audio_file_path=data_folder / "interpreter" / "streaming_audio.wav",
                              model_size="small",
                              sample_rate=16000,
                              min_segment_duration=3,
                              translate_enabled=True)
    self.run()

    self.evaluate()
