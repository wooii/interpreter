"""Real-time speech-to-text with optional translation.

Quick start (defaults: SenseVoice STT + opus-mt-en-zh translation):

    from interpreter.transcribe import RealTimeTranscribe

    rtt = RealTimeTranscribe(translate_to="Chinese")
    rtt.run()                       # live mic; Ctrl+C to stop
    rtt.evaluate()                  # WER/CER vs full-file reference (needs audio_file_path)

Model selection (the CLI picks models internally per task — see __main__.py;
the library still accepts explicit names):

    RealTimeTranscribe(stt_model="sensevoice")             # default; zh<->en code-switching
    RealTimeTranscribe(stt_model="parakeet-tdt-0.6b-v2")   # best en-only accuracy

    RealTimeTranscribe(translate_model="opus-mt-en-zh")    # default; dedicated NMT, en->zh
    RealTimeTranscribe(translate_to=None)                  # no translation

Record a session for later evaluation:

    rtt = RealTimeTranscribe(audio_file_path="session.wav", translate_to=None)
    rtt.run()                       # writes session.wav on stop
    rtt.evaluate()                  # WER/CER of the live transcript vs offline re-transcribe

Standalone use (no mic): transcribe a 16 kHz mono file or array directly

    from interpreter.transcribe import SpeechToText
    from interpreter.translate import Translator

    stt = SpeechToText("sensevoice")            # or "parakeet-tdt-0.6b-v2"
    text = stt.extract_text(stt.transcribe_file("clip.wav"))
    print(Translator().translate(text))         # opus-mt-en-zh -> Chinese

CLI: `uv run python -m interpreter listen|dictate` runs live dictation (see
__main__.py). Model picks follow the Phase 1 conclusion (docs/benchmark.md);
weights download anonymously from HF on first use.
"""

from __future__ import annotations

import collections
import importlib.util
import math
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import noisereduce as nr
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from jiwer import cer, wer

from interpreter import DATA_DIR
from interpreter.translate import Translator

STT_MODEL_EN_ONLY = "parakeet-tdt-0.6b-v2"
STT_MODEL_MIXED = "sensevoice"


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _sentence_case(text: str) -> str:
    """Capitalize the first letter found in a segment; leave the rest lowercase."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def _normalize_sensevoice_case(text: str) -> str:
    """SenseVoice emits English in ALL CAPS (a model artifact; parakeet and the
    Chinese portions are unaffected) — lowercase and sentence-case each segment."""
    return _sentence_case(text.lower())

MODELS_DIR = DATA_DIR / "benchmark" / "transcribe" / "models"

MODEL_SPECS: dict[str, dict] = {
    "parakeet-tdt-0.6b-v2": {
        "repo": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        "files": {
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        "factory": "from_transducer",
        "kwargs": {"model_type": "nemo_transducer"},
    },
    "sensevoice": {
        "repo": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
        "files": {
            "model": "model.int8.onnx",
            "tokens": "tokens.txt",
        },
        "factory": "from_sense_voice",
        "kwargs": {"use_itn": True},
    },
}


def _download_model_files(name: str) -> None:
    spec = MODEL_SPECS[name]
    dest = MODELS_DIR / name
    if (dest / ".complete").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for rel in spec["files"].values():
        hf_hub_download(spec["repo"], rel, repo_type="model", local_dir=dest)
    (dest / ".complete").touch()


def _ensure_onnxruntime_dylib() -> None:
    """macOS only: sherpa-onnx wheels don't bundle onnxruntime — dlopen of
    `@rpath/libonnxruntime.<ver>.dylib` fails (docs/benchmark.md). Copy the
    dylibs from the installed onnxruntime package into the sherpa package's
    lib dir — the first @rpath search location. dyld reads DYLD_* at exec
    time, so a runtime env tweak can't fix this."""
    if sys.platform != "darwin":
        return
    try:
        import onnxruntime

        spec = importlib.util.find_spec("sherpa_onnx")
        if spec is None or not spec.submodule_search_locations:
            return
        sherpa_lib = Path(spec.submodule_search_locations[0]) / "lib"
        sherpa_lib.mkdir(parents=True, exist_ok=True)
        capi = Path(onnxruntime.__file__).parent / "capi"
        for src in capi.glob("libonnxruntime*.dylib"):
            dest = sherpa_lib / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
    except Exception:  # noqa: S110, BLE001 - best-effort fix; surface the real import error
        pass


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
            trust_repo=True,
        )

    def is_speech(self, frame):
        if len(frame) != self.frame_size:
            return False
        frame_tensor = torch.from_numpy(frame).float()
        speech_prob = self.model(frame_tensor, self.sample_rate).item()
        return speech_prob > self.speech_threshold


class SherpaSegment:
    """Minimal segment object for the display layer (`.text`, `.probability`,
    `.t0`, `.t1`). Sherpa exposes no per-token log-probs for SenseVoice, so
    its `probability` is 1.0; transducer models (parakeet) expose `tokens` +
    `ys_log_probs`, which `_word_probs_from_result` groups into `word_probs` —
    (word, prob) pairs the real-time display colors per word."""

    def __init__(
        self,
        text: str,
        probability: float = 1.0,
        t0: float = 0.0,
        t1: float = 0.0,
        word_probs: list[tuple[str, float]] | None = None,
    ) -> None:
        self.text = text
        self.probability = probability
        self.t0 = t0
        self.t1 = t1
        self.word_probs = word_probs


def _word_probs_from_result(result: Any) -> list[tuple[str, float]] | None:
    """Per-word confidence from sherpa's per-token data, or None when the
    model exposes none (SenseVoice). Transducer tokens may or may not carry
    leading spaces (observed both on the same model), so words are grouped by
    cumulative character length against the words of `result.text`; a word's
    probability is exp(mean of its token log-probs)."""
    tokens = list(getattr(result, "tokens", None) or [])
    scores = list(getattr(result, "ys_log_probs", None) or [])
    if not tokens or len(scores) != len(tokens):
        return None
    text_words = [w for w in (result.text or "").split() if w]
    if not text_words:
        return None
    flat = "".join(t.strip() for t in tokens)
    if flat != "".join(text_words):
        return None
    words: list[tuple[str, float]] = []
    cur: list[str] = []
    cur_scores: list[float] = []
    target_len = len(text_words[0])
    acc = 0
    ti = 0
    for tok, score in zip(tokens, scores):
        stripped = tok.strip()
        if not stripped:
            continue
        cur.append(stripped)
        cur_scores.append(score)
        acc += len(stripped)
        if acc >= target_len:
            words.append(("".join(cur), math.exp(sum(cur_scores) / len(cur_scores))))
            cur, cur_scores = [], []
            acc = 0
            ti += 1
            target_len = len(text_words[ti]) if ti < len(text_words) else 0
    return words or None


class SherpaStt:
    """Offline sherpa-onnx recognizer over raw 16 kHz mono float32 audio.

    Model choice (docs/benchmark.md, 2026-08-24): en-only ->
    parakeet-tdt-0.6b-v2, mixed -> sensevoice; both sherpa-onnx int8
    (whisper.cpp/Moonshine dropped). Weights download anonymously from HF.
    """

    def __init__(self, model_name: str, num_threads: int = 4) -> None:
        spec = MODEL_SPECS[model_name]
        self.model_name = model_name
        _download_model_files(model_name)
        _ensure_onnxruntime_dylib()
        import sherpa_onnx

        kwargs: dict[str, object] = {
            key: str(MODELS_DIR / model_name / rel)
            for key, rel in spec["files"].items()
        }
        kwargs.update(spec.get("kwargs", {}))
        kwargs["num_threads"] = num_threads
        self.recognizer: Any = getattr(sherpa_onnx.OfflineRecognizer, spec["factory"])(
            **kwargs
        )

    def transcribe(self, audio: np.ndarray) -> list[SherpaSegment]:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio.astype(np.float32))
        self.recognizer.decode_stream(stream)
        result = stream.result
        text = result.text.strip()
        if not text:
            return []
        return [SherpaSegment(text, word_probs=_word_probs_from_result(result))]

    def transcribe_file(self, path: str | Path) -> list[SherpaSegment]:
        audio, sr = sf.read(str(path), dtype="float32")
        assert sr == 16000, f"expected 16 kHz audio, got {sr}"
        return self.transcribe(audio)


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
    per-token log-probs, grouped into word probs in _word_probs_from_result.
    SenseVoice exposes no per-token scores in sherpa-onnx 1.13.0 — its
    output is uncolored (uniform), a known limitation (docs/benchmark.md).
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model: Any = SherpaStt(model_name)

    def transcribe(self, audio: np.ndarray):
        audio = audio.astype(np.float32)
        return self.model.transcribe(audio)

    def transcribe_file(self, file_path: str):
        return self.model.transcribe_file(str(file_path))

    def extract_text(self, result):
        text = " ".join([i.text for i in result]).strip()
        if self.model_name == "sensevoice":
            text = _normalize_sensevoice_case(text)
        return text


class RealTimeTranscribe:
    def __init__(
        self,
        audio_file_path=None,
        stt_model=STT_MODEL_MIXED,
        translate_model="opus-mt-en-zh",
        translate_to="Chinese",
        max_segment_duration=5.0,
        plain_output=False,
    ):
        self.audio_file_path = audio_file_path
        self.stt_model = stt_model
        self.translate_to = translate_to
        self.plain_output = plain_output
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
        time_str = self._get_time_str()
        self.transcript.append(transcript)
        if self.plain_output:
            print(transcript, flush=True)
        else:
            formated_transcript = self._format_transcript(result)
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

    def _print_clean_transcript(self):
        if not self.transcript:
            return
        print("\nTranscript:")
        print("\n".join(self.transcript))

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
            if self.plain_output:
                self._print_clean_transcript()

    def evaluate(self):
        if self.audio_file_path is None:
            print("No audio_file_path provided for evaluation.")
            return None
        result = self.stt.transcribe_file(str(self.audio_file_path))
        self.reference_transcript = self.stt.extract_text(result)
        self.realtime_transcript = " ".join(self.transcript).strip()
        cer_error = cer(
            self.reference_transcript.lower(), self.realtime_transcript.lower()
        )
        print(f"Character Error Rate (CER): {cer_error:.2%}")
        metrics = {"CER": cer_error}
        # WER only makes sense on whitespace-segmented text (English). On
        # unsegmented Chinese/mixed it compares chars vs segments and explodes
        # (docs/benchmark.md scores zh with CER for this reason), so skip it.
        if not (
            _contains_cjk(self.reference_transcript)
            or _contains_cjk(self.realtime_transcript)
        ):
            wer_error = wer(
                self.reference_transcript.lower(), self.realtime_transcript.lower()
            )
            print(f"Word Error Rate (WER): {wer_error:.2%}")
            metrics["WER"] = wer_error
        else:
            print("Word Error Rate (WER): N/A (Chinese/mixed — use CER)")
        print(f"Reference Transcript: {self.reference_transcript}")
        print(f"Realtime Transcript: {self.realtime_transcript}")
        return metrics
