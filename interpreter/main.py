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
import webrtcvad
import collections
import datetime

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
                 blocksize=1024,
                 word_prob_threshold=0.85,
                 translate_enabled=True,
                 vad_aggressiveness=2):
        self.audio_file_path = audio_file_path
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.word_prob_threshold = word_prob_threshold
        self.translate_enabled = translate_enabled
        self.vad_aggressiveness = vad_aggressiveness

        # Initialize Whisper model
        self.model = whisper.load_model(model_size)
        self.translator = OfflineTranslator()

        # Audio processing parameters
        self.frame_duration_ms = 30  # 30ms frames as required by VAD
        self.frame_size = int(sample_rate * (self.frame_duration_ms / 1000.0))
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # Rolling window parameters for speech detection
        self.padding_duration_ms = 300  # Duration of padding for silence detection
        self.padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)

        # Ring buffer to store audio frames
        self.ring_buffer = collections.deque(maxlen=self.padding_frames)
        self.triggered = False  # State to track if we're in a speech segment

        # Storage for recorded frames and previous audio
        self.recorded_frames = []
        self.prev_tail_audio = np.zeros(0, dtype='float32')

        # Threading and synchronization
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.transcriber_thread = None
        self.running = False

        # Transcript storage
        self.transcript = []
        self.full_recording = np.zeros(0, dtype='float32')

        # Timing
        self.start_time = time.time()

    def float_to_pcm16(self, audio):
        """Convert float32 audio to int16 PCM format required by VAD"""
        return (audio * 32767).astype(np.int16).tobytes()

    def is_speech(self, frame):
        """Check if a frame contains speech using VAD"""
        if len(frame) != self.frame_size:
            return False
        return self.vad.is_speech(self.float_to_pcm16(frame), self.sample_rate)

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback function for sounddevice"""
        if status:
            print(status)
        with self.lock:
            audio_data = indata.flatten()
            # Save all recorded audio if path is provided
            if self.audio_file_path:
                self.full_recording = np.append(self.full_recording, audio_data)

            # Process audio in VAD frames
            while len(audio_data) >= self.frame_size:
                frame = audio_data[:self.frame_size]
                audio_data = audio_data[self.frame_size:]

                is_speech = self.is_speech(frame)

                # State machine for speech detection
                if not self.triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])

                    # If we're in a speech segment
                    if num_voiced > 0.5 * self.ring_buffer.maxlen:
                        self.triggered = True
                        # Add all buffered frames to recorded frames
                        for f, s in self.ring_buffer:
                            self.recorded_frames.append(f)
                        self.ring_buffer.clear()
                else:
                    # We're in a speech segment, just add the frame
                    self.recorded_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])

                    # End of speech segment
                    if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                        # Put the recorded segment in the queue for transcription
                        if len(self.recorded_frames) > 0:
                            segment = np.concatenate(self.recorded_frames)
                            self.q.put(segment.copy())

                        self.triggered = False
                        self.recorded_frames = []
                        self.ring_buffer.clear()

    def transcriber(self):
        """Transcription thread that processes audio segments"""
        while True:
            segment = self.q.get()
            if segment is None:
                break

            # Combine with previous audio tail
            full_segment = np.concatenate([self.prev_tail_audio, segment])

            # Skip if too quiet (likely just noise)
            if np.sqrt(np.mean(full_segment**2)) < 0.001:
                continue

            # Transcribe with Whisper
            t0 = time.time()
            result = self.model.transcribe(
                audio=full_segment.astype(np.float32),
                word_timestamps=True,
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-0.5,
                condition_on_previous_text=True,
                fp16=False  # Explicitly disable FP16 for CPU compatibility
            )
            t1 = time.time()

            # Process transcription results
            if not result.get("segments"):
                continue

            all_words = []
            for seg in result["segments"]:
                if "words" not in seg:
                    continue
                if seg.get("no_speech_prob", 0) > 0.9 and seg.get("avg_logprob", -1) < -0.5:
                    continue
                all_words.extend(seg["words"])

            if not all_words:
                continue

            # Determine where to cut the audio for overlap
            if len(all_words) > 0:
                # Save raw English transcript
                sentence = " ".join(w["word"].strip() for w in all_words if w["word"].strip())
                self.transcript.append(sentence)

                # Create colored text for display
                text = " ".join(color_word_gradient(w["word"].strip(), w["probability"])
                                for w in all_words if w["word"].strip()).strip()

                if text:
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

                    # Keep a small overlap for context in next transcription
                    # Keep last 0.2 seconds for context
                    overlap_samples = int(0.2 * self.sample_rate)
                    if len(full_segment) > overlap_samples:
                        self.prev_tail_audio = full_segment[-overlap_samples:]
                    else:
                        self.prev_tail_audio = full_segment

    def run(self):
        """Start the real-time transcription process"""
        print(f"Improved real-time transcribe (Model: Whisper {self.model_size})... (Ctrl+C to stop)")
        self.running = True
        self.start_time = time.time()

        # Start transcription thread
        self.transcriber_thread = threading.Thread(target=self.transcriber, daemon=True)
        self.transcriber_thread.start()

        try:
            # Start audio input stream
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1,
                                dtype='float32',
                                callback=self.audio_callback,
                                blocksize=self.frame_size):  # Use frame_size for better VAD performance
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()

    def stop(self):
        """Stop the transcription process"""
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
        print(f"Reference Transcript: {self.reference_transcript}")
        print(f"Realtime Transcript: {self.realtime_transcript}")

        return {"WER": wer_error, "CER": cer_error}


if __name__ == "__main__":
    # Use the improved real-time transcription
    self = RealTimeTranscribe(audio_file_path=data_folder / "interpreter" / "streaming_audio.wav",
                              model_size="small",
                              translate_enabled=True)
    self.run()

    self.evaluate()