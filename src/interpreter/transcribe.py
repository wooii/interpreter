"""Real-time speech-to-text with optional translation.

Quick start (defaults: SenseVoice STT + opus-mt-en-zh translation):

    from interpreter.transcribe import RealTimeTranscribe

    rtt = RealTimeTranscribe(translate_to="Chinese")
    rtt.run()                       # live mic; Ctrl+C to stop
    rtt.evaluate()                  # WER/CER vs full-file reference (needs audio_file_path)

Model selection:

    RealTimeTranscribe(stt_model="sensevoice")             # default; zh<->en code-switching
    RealTimeTranscribe(stt_model="parakeet-tdt-0.6b-v2")   # best en-only accuracy

    RealTimeTranscribe(translate_model="opus-mt-en-zh")    # default; dedicated NMT, en->zh
    RealTimeTranscribe(translate_to=None)                  # no translation

Record a session for later evaluation:

    rtt = RealTimeTranscribe(audio_file_path="session.wav", translate_to=None)
    rtt.run()                       # writes session.wav on stop
    rtt.evaluate()                  # WER/CER of the live transcript vs offline re-transcribe

Standalone use (no mic): transcribe a 16 kHz mono file or array directly

    from interpreter.transcribe import SpeechToText, Translator

    stt = SpeechToText("sensevoice")            # or "parakeet-tdt-0.6b-v2"
    text = stt.extract_text(stt.transcribe_file("clip.wav"))
    print(Translator().translate(text))         # opus-mt-en-zh -> Chinese

CLI: `python -m interpreter.transcribe` runs live dictation with the defaults
(STT = sensevoice, translate = opus-mt-en-zh, output to data/transcribe_<ts>.wav).
Model picks follow the Phase 1 conclusion (docs/benchmark.md); weights download
anonymously from HF on first use.
"""

from __future__ import annotations

import collections
import datetime
import queue
import threading
import time
from typing import Any

import noisereduce as nr
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from jiwer import cer, wer

from interpreter import DATA_DIR


class VAD:
    """Voice activity detection"""

    def __init__(self, frame_size=512, sample_rate=16000, speech_threshold=0.4):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

    def is_speech(self, frame):
        if len(frame) != self.frame_size:
            return False
        frame_tensor = torch.from_numpy(frame).float()
        speech_prob = self.model(frame_tensor, self.sample_rate).item()
        return speech_prob > self.speech_threshold


class Translator:
    """en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend.

    Dedicated NMT: deterministic, best BLEU on the benchmark corpus (33.59,
    docs/benchmark.md), ~1.2 s/sentence; single pair en->zh. The qwen3.5 LLM
    quality mode was dropped 2026-08-24 — live dictation showed hallucinated
    content and a meaning-reversed error (PLAN.md, docs/benchmark.md).
    """

    def __init__(self, model="opus-mt-en-zh", target_lang="Chinese"):
        self.model = model
        self.target_lang = target_lang
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = f"Helsinki-NLP/{model}"
        self._nmt_tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
        self._nmt_model: Any = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        inputs = self._nmt_tokenizer(text, return_tensors="pt", truncation=True)
        out = self._nmt_model.generate(**inputs, max_length=256)
        return self._nmt_tokenizer.decode(out[0], skip_special_tokens=True).strip()


def process_audio_segment(full_segment, sample_rate):
    # start = time.time()
    if np.sqrt(np.mean(full_segment**2)) < 0.001:
        return None
    max_val = np.max(np.abs(full_segment)) + 1e-8
    full_segment = full_segment / max_val
    full_segment = nr.reduce_noise(y=full_segment, sr=sample_rate)
    # print(f"    [Audio processing time: {time.time() - start:.4f}s]", flush=True)
    return full_segment


class SpeechToText:
    """STT backend dispatch — Phase 1 model-selection winners (docs/benchmark.md).
    Sherpa-onnx int8 only since 2026-08-24 (whisper.cpp and Moonshine both
    dropped — see docs/benchmark.md for the reasons).

    model_name:
      - "sensevoice"            product default (sherpa int8; dictate/multilingual winner)
      - "parakeet-tdt-0.6b-v2"  en-only / listen option (sherpa int8 transducer)

    Per-word confidence coloring: the sherpa transducer (parakeet) exposes
    per-token log-probs, grouped into word probs in sherpa_stt.py.
    SenseVoice exposes no per-token scores in sherpa-onnx 1.13.0 — its
    output is uncolored (uniform), a known limitation (docs/benchmark.md).
    """

    def __init__(self, model_name):
        from interpreter.sherpa_stt import SherpaStt

        self.model_name = model_name
        self.model: Any = SherpaStt(model_name)

    def transcribe(self, audio: np.ndarray):
        audio = audio.astype(np.float32)
        return self.model.transcribe(audio)

    def transcribe_file(self, file_path: str):
        return self.model.transcribe_file(str(file_path))

    def extract_text(self, result):
        return " ".join([i.text for i in result]).strip()


class RealTimeTranscribe:
    def __init__(
        self,
        audio_file_path=None,
        stt_model="sensevoice",
        translate_model="opus-mt-en-zh",
        translate_to="Chinese",
        max_segment_duration=5.0,
    ):
        self.audio_file_path = audio_file_path
        self.stt_model = stt_model
        self.translate_to = translate_to
        self.max_segment_duration = max_segment_duration
        self.sample_rate = 16000
        self.frame_size = 512
        self.vad = VAD(self.frame_size, self.sample_rate)
        self.translator = (
            Translator(model=translate_model, target_lang=translate_to)
            if translate_to
            else None
        )
        self.stt = SpeechToText(stt_model)
        self.stt_model_name = self.stt.model_name
        self._initialize_state()
        # Calculate max frames based on the configurable duration
        self.max_segment_frames = int(
            self.max_segment_duration * self.sample_rate / self.frame_size
        )

    def _initialize_state(self):
        self.ring_buffer_maxlen = 20
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_maxlen)
        self.triggered = False
        self.recorded_frames = []
        self.recorded_frames_count = 0  # Track number of frames recorded
        self.prev_tail_audio = np.zeros(0, dtype="float32")
        self.q_for_vad = queue.Queue()
        self.q_for_transcription = queue.Queue()
        self.q_for_translation = queue.Queue()
        self.lock = threading.Lock()
        self.transcription_thread = None
        self.vad_thread = None
        self.translation_thread = None
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
            frame = audio_data[: self.frame_size]
            audio_data = audio_data[self.frame_size :]
            self.q_for_vad.put(frame)

    def _vad_worker(self):
        while self.running:
            try:
                frame = self.q_for_vad.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            is_speech = self.vad.is_speech(frame)
            self.ring_buffer.append((frame, is_speech))
            if not self.triggered:
                if sum(s for _, s in self.ring_buffer) > 0.4 * self.ring_buffer_maxlen:
                    self.triggered = True
                    for f, _ in self.ring_buffer:
                        self.recorded_frames.append(f)
                    self.recorded_frames_count = len(self.recorded_frames)
                    self.ring_buffer.clear()
            else:
                self.recorded_frames.append(frame)
                self.recorded_frames_count += 1

                # Check if we've reached maximum segment duration
                max_duration_reached = (
                    self.recorded_frames_count >= self.max_segment_frames
                )

                # Check for silence (end of speech)
                silence_detected = (
                    sum(1 for _, s in self.ring_buffer if not s)
                    > 0.8 * self.ring_buffer_maxlen
                )

                # If either maximum duration reached or silence detected, process the segment
                if max_duration_reached or silence_detected:
                    if self.recorded_frames:
                        segment = np.concatenate(self.recorded_frames)
                        self.q_for_transcription.put(segment.copy())
                    self.triggered = False
                    self.recorded_frames.clear()
                    self.recorded_frames_count = 0
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

    def _format_and_display_transcription(self, result, transcription_time=None):
        if not (isinstance(result, list) and result):
            return
        transcript = self.stt.extract_text(result)
        formated_transcript = self._format_transcript(result)
        time_str = self._get_time_str()
        self.transcript.append(transcript)
        duration = (
            f" ({transcription_time:.4f}s)" if transcription_time is not None else ""
        )
        print(f"[{time_str}] {formated_transcript}{duration}", flush=True)
        if self.translator:
            self.q_for_translation.put(transcript)

    def _format_transcript(self, result):
        # Returns colored text for the transcription. Per-word confidence
        # from the sherpa transducer's word_probs (when present) — skips
        # empty segments/words.
        parts = []
        for seg in result:
            word_probs = getattr(seg, "word_probs", None)
            if word_probs:
                parts.append(" ".join(self._color_word(w, p) for w, p in word_probs))
            elif seg.text.strip():
                parts.append(self._color_word(seg.text.strip(), seg.probability))
        return " ".join(parts).strip()

    def _get_time_str(self):
        elapsed = time.time() - self.start_time
        return f"{int(elapsed // 60):02d}:{elapsed % 60:06.3f}"

    def _transcription_worker(self):
        while self.running:
            try:
                segment = self.q_for_transcription.get(timeout=0.1)
            except queue.Empty:
                continue
            if segment is None:
                break
            full_segment = np.concatenate([self.prev_tail_audio, segment])
            processed_segment = process_audio_segment(full_segment, self.sample_rate)
            if processed_segment is None:
                continue
            start_time = time.time()
            result = self.stt.transcribe(processed_segment)
            transcription_time = time.time() - start_time
            self._format_and_display_transcription(result, transcription_time)

    def _translation_worker(self):
        if self.translator is None:
            return
        while self.running:
            try:
                transcript = self.q_for_translation.get(timeout=0.1)
            except queue.Empty:
                continue
            if transcript is None:
                break
            translate_start = time.time()
            translated = self.translator.translate(transcript)
            translate_time = time.time() - translate_start
            # Print translation on a new indented line below the transcript
            print(f"    → {translated} ({translate_time:.4f}s)", flush=True)

    def _stop(self):
        self.running = False
        self.q_for_transcription.put(None)
        self.q_for_vad.put(None)
        self.q_for_translation.put(None)
        if self.transcription_thread is not None:
            self.transcription_thread.join()
        if self.vad_thread is not None:
            self.vad_thread.join()
        if self.translation_thread is not None:
            self.translation_thread.join()
        if self.audio_file_path and self.full_recording_list:
            full_audio = np.concatenate(self.full_recording_list)
            sf.write(self.audio_file_path, full_audio, self.sample_rate)
            print(f"Audio saved to {self.audio_file_path}")

    def run(self):
        print("Real-time transcribe... (Ctrl+C to stop)")
        print(f"Speech-to-text model: {self.stt_model_name}")
        if self.translator:
            print(
                f"Translation model: {self.translator.model} → {self.translator.target_lang}"
            )
        self.running = True
        self.start_time = time.time()
        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.vad_thread.start()
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self.transcription_thread.start()
        if self.translator:
            self.translation_thread = threading.Thread(
                target=self._translation_worker, daemon=True
            )
            self.translation_thread.start()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=self.frame_size,
            ):
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
        wer_error = wer(
            self.reference_transcript.lower(), self.realtime_transcript.lower()
        )
        cer_error = cer(
            self.reference_transcript.lower(), self.realtime_transcript.lower()
        )
        print(f"Word Error Rate (WER): {wer_error:.2%}")
        print(f"Character Error Rate (CER): {cer_error:.2%}")
        print(f"Reference Transcript: {self.reference_transcript}")
        print(f"Realtime Transcript: {self.realtime_transcript}")
        return {"WER": wer_error, "CER": cer_error}


if __name__ == "__main__":
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")

    # Phase 1 conclusion defaults (docs/benchmark.md, 2026-08-24):
    #   per-mode target: listen -> parakeet-tdt-0.6b-v2, dictate -> sensevoice;
    #   streaming (Moonshine) dropped 2026-08-24 — sherpa online recognizer
    #   probe is the future streaming path (Phase 2). Until Phase 2 wires
    #   per-mode config, the product default is sensevoice (best code-switcher,
    #   fastest+lightest)
    #   + opus-mt-en-zh (dedicated NMT, best BLEU; qwen LLM quality mode dropped
    #   2026-08-24 — see PLAN.md).
    self = RealTimeTranscribe(
        audio_file_path=DATA_DIR / f"transcribe_{timestamp}.wav",
        stt_model="sensevoice",
        translate_model="opus-mt-en-zh",
        translate_to="Chinese",
        max_segment_duration=10.0,
    )

    self.run()

    self.evaluate()
