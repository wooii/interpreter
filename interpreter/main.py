import whisper
import threading
import queue
import time
import datetime
import webrtcvad
import collections
import warnings
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
from jiwer import wer, cer
from transformers import MarianMTModel, MarianTokenizer
from interpreter import data_folder


warnings.filterwarnings(
    action="ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)


class OfflineTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-zh"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.translator = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        batch = self.tokenizer([text], return_tensors="pt", padding=True)
        gen = self.translator.generate(**batch)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


class RealTimeTranscribe:
    """
    Real-time transcription and translation.
    For degraded audio (e.g., playback to mic), try vad_aggressiveness=2 or 3.
    """
    def __init__(self,
                 audio_file_path=None,
                 model_size="small",
                 translate_enabled=True,
                 word_prob_threshold=0.85,
                 vad_aggressiveness=1):
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.translate_enabled = translate_enabled
        self.word_prob_threshold = word_prob_threshold
        self.vad_aggressiveness = vad_aggressiveness
        self._initialize_models()
        self._initialize_audio_params()
        self._initialize_state()

    def _initialize_models(self):
        """Initialize Whisper model and optional translator."""
        self.stt_model = whisper.load_model(self.model_size)
        if self.translate_enabled:
            self.translator = OfflineTranslator()

    def _initialize_audio_params(self):
        """Initialize audio processing parameters."""
        self.sample_rate = 16000
        self.frame_duration_ms = 20  # 20ms frames as required by VAD
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

    def _initialize_state(self):
        """Initialize state management variables."""
        # Ring buffer to store audio frames
        self.ring_buffer = collections.deque(maxlen=10)
        self.triggered = False  # State to track if we're in a speech segment

        # Storage for recorded frames and previous audio
        self.recorded_frames = []
        self.prev_tail_audio = np.zeros(0, dtype='float32')

        # Queues for threading
        self.q_raw = queue.Queue()  # raw frames from callback
        self.q = queue.Queue()  # speech segments for transcription

        # Threading and synchronization
        self.lock = threading.Lock()
        self.transcriber_thread = None
        self.vad_thread = None
        self.running = False

        # Transcript storage
        self.transcript = []
        self.full_recording_list = []  # more efficient than np.append

        # Timing
        self.start_time = time.time()

    def _float_to_pcm16(self, audio):
        """Convert float32 audio to int16 PCM format required by VAD"""
        return (audio * 32767).astype(np.int16).tobytes()

    def _is_speech(self, frame):
        """Check if a frame contains speech using VAD"""
        if len(frame) != self.frame_size:
            return False
        return self.vad.is_speech(self._float_to_pcm16(frame), self.sample_rate)

    def _audio_callback(self, indata, frames, time_info, status):
        """Minimal audio callback: just slice frames and enqueue."""
        if status:
            print(status)
        audio_data = indata.flatten()

        # Collect audio if saving
        if self.audio_file_path:
            self.full_recording_list.append(audio_data)

        # Slice into fixed-size frames and enqueue
        while len(audio_data) >= self.frame_size:
            frame = audio_data[:self.frame_size]
            audio_data = audio_data[self.frame_size:]
            self.q_raw.put(frame)

    def _vad_worker(self):
        """Background worker: process frames with VAD state machine."""
        while self.running:
            frame = self.q_raw.get()
            if frame is None:
                break
            is_speech = self._is_speech(frame)
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

    def _color_word(self, word, prob):
        """Map probability (0.0-1.0) to RGB color from red -> yellow -> green."""
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


    def _process_audio_segment(self, full_segment):
        """Process an audio segment with noise reduction and normalization."""
        # Skip if too quiet (likely just noise)
        if np.sqrt(np.mean(full_segment**2)) < 0.001:
            return None

        # Normalize audio to [-1, 1] for consistent loudness
        max_val = np.max(np.abs(full_segment)) + 1e-8
        full_segment = full_segment / max_val

        # Apply noise reduction
        full_segment = nr.reduce_noise(y=full_segment, sr=self.sample_rate)
        return full_segment

    def _extract_words(self, result):
        """Extract words from transcription result with filtering."""
        if not result.get("segments"):
            return []

        all_words = []
        for seg in result["segments"]:
            if "words" not in seg:
                continue
            if seg.get("no_speech_prob", 0) > 0.9 and seg.get("avg_logprob", -1) < -0.5:
                continue
            all_words.extend(seg["words"])

        return all_words

    def _format_and_display_transcription(self, all_words):
        """Format the transcription output and display it."""
        if not all_words:
            return

        # Save raw English transcript
        sentence = " ".join(w["word"].strip() for w in all_words if w["word"].strip())

        # Create colored text for display
        text = " ".join(self._color_word(w["word"].strip(), w["probability"])
                        for w in all_words if w["word"].strip()).strip()

        # Display the transcription with timestamp and optional translation
        if text:
            self.transcript.append(sentence)

            end = time.time()
            elapsed = datetime.timedelta(seconds=end - self.start_time)
            # Convert timedelta to datetime object (relative to 0)
            dt = (datetime.datetime.min + elapsed)
            time_str = dt.strftime("%M:%S.%f")[:-3]  # MM:SS.mmm

            if self.translate_enabled:
                translated = self.translator.translate(sentence)
                print(f"[{time_str}] {text} → {translated}")
            else:
                print(f"[{time_str}] {text}")

    def _update_audio_context(self, segment, all_words, max_overlap_sec=0.5):
        """Keep the tail of the last word as overlap for next transcription."""
        if not all_words:
            self.prev_tail_audio = np.zeros(0, dtype='float32')
            return

        # Get last word's end timestamp
        last_word = all_words[-1]
        end_time = last_word.get("end", None)  # in seconds
        if end_time is None:
            self.prev_tail_audio = np.zeros(0, dtype='float32')
            return
        start_sample = int(end_time * self.sample_rate)
        # Convert to sample index
        max_overlap_samples = int(max_overlap_sec * self.sample_rate)
        start_sample = max(len(segment) - max_overlap_samples, start_sample)
        self.prev_tail_audio = segment[start_sample:]

    def _transcribe(self):
        """Transcription thread that processes audio segments"""
        while True:
            segment = self.q.get()
            if segment is None:
                break

            # Combine with previous audio tail
            full_segment = np.concatenate([self.prev_tail_audio, segment])

            # Process audio segment
            processed_segment = self._process_audio_segment(full_segment)
            if processed_segment is None:
                continue

            # Transcribe with Whisper (tweaked params for degraded audio)
            result = self.stt_model.transcribe(
                audio=processed_segment.astype(np.float32),
                word_timestamps=True,
                temperature=0.0,
                no_speech_threshold=0.5,  # less likely to skip quiet speech
                logprob_threshold=-0.3,   # accept more uncertain words
                condition_on_previous_text=True,
                fp16=False  # Explicitly disable FP16 for CPU compatibility
            )

            # Extract words from result
            all_words = self._extract_words(result)
            if not all_words:
                continue

            # Format and display transcription with audio context update
            self._format_and_display_transcription(all_words)
            if all_words and segment is not None:
                self._update_audio_context(segment, all_words)

    def _stop(self):
        """Stop the transcription process"""
        self.running = False
        self.q.put(None)
        self.q_raw.put(None)
        if self.transcriber_thread is not None:
            self.transcriber_thread.join()
        if self.vad_thread is not None:
            self.vad_thread.join()
        # Save full recording if path was provided
        if self.audio_file_path and self.full_recording_list:
            full_audio = np.concatenate(self.full_recording_list)
            sf.write(self.audio_file_path, full_audio, self.sample_rate)
            print(f"Audio saved to {self.audio_file_path}")

    def run(self):
        """Start the real-time transcription process"""
        print(f"Real-time transcribe (Model: Whisper {self.model_size})... (Ctrl+C to stop)")
        self.running = True
        self.start_time = time.time()

        # Start VAD and transcription threads
        self.transcriber_thread = threading.Thread(target=self._transcribe, daemon=True)
        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.transcriber_thread.start()
        self.vad_thread.start()

        try:
            # Start audio input stream
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1,
                                dtype='float32',
                                callback=self._audio_callback,
                                blocksize=self.frame_size):  # Use frame_size for better VAD performance
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self._stop()

    def evaluate(self):
        """Transcribe saved audio using Whisper model and compare with real-time transcript."""
        if self.audio_file_path is None:
            print("No audio_file_path provided for evaluation.")
            return None
        result = self.stt_model.transcribe(str(self.audio_file_path))
        self.reference_transcript = result["text"].strip()
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
                              translate_enabled=True)
    self.run()

    self.evaluate()