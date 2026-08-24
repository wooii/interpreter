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
__main__.py). Model picks follow the Phase 1 conclusion (_archive/benchmark-2026-08-24.md);
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


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _join_text_parts(parts, force_space=False) -> str:
    """Join transcript chunks. By default a space is inserted only between two
    non-CJK chars — English words need one, but no space between CJK (or
    CJK/latin) chars. `force_space=True` (display/copy output) separates every
    chunk with a space — dictation has no punctuation, so segment boundaries
    need at least that. Metrics keep the unspaced join (spaces would inflate
    CER against the unspaced zh reference)."""
    out = ""
    for p in parts:
        if not p:
            continue
        if out:
            if force_space:
                out += " "
            else:
                lc = out[-1]
                fc = p[0]
                if (
                    not lc.isspace()
                    and not fc.isspace()
                    and not _is_cjk_char(lc)
                    and not _is_cjk_char(fc)
                ):
                    out += " "
        out += p
    return out


def _norm(text: str) -> str:
    """Whitespace-collapsed, lowercased view for stability comparison."""
    return "".join(ch for ch in text.lower() if not ch.isspace())


def _stable_prefix(full: str, prefix: str) -> bool:
    """True when `prefix` is a stable prefix of `full` — i.e. appending the
    new segment did not change the earlier text. For whitespace-segmented text
    (English) compare at word boundaries so a mid-word re-segmentation like
    abc|def -> abcdef keeps the window open; for CJK (no word boundaries) a
    character-prefix is enough because committing a char prefix + tail
    reconstructs the full text on concatenation."""
    if not _contains_cjk(prefix):
        pw = prefix.split()
        fw = full.split()
        return (
            bool(pw)
            and len(fw) >= len(pw)
            and [w.lower() for w in fw[: len(pw)]] == [w.lower() for w in pw]
        )
    pn = _norm(prefix)
    return bool(pn) and _norm(full).startswith(pn)


def _strip_prefix(full: str, prefix: str) -> str:
    """Suffix of `full` after `prefix` (used only when `_stable_prefix`
    holds)."""
    if not _contains_cjk(prefix):
        return " ".join(full.split()[len(prefix.split()) :])
    pn = _norm(prefix)
    consumed = 0
    i = 0
    n = len(full)
    while consumed < len(pn) and i < n:
        ch = full[i]
        if ch.isspace():
            i += 1
            continue
        consumed += 1
        i += 1
    return full[i:].lstrip()


def _chunk_punct(text: str) -> str:
    """Sentence-ending punctuation for dictation. SenseVoice emits none.
    English chunks get a period; Chinese chunks get none — the space in the
    join separates zh sentences (user preference: zh reads better space-only).
    Empty when the chunk already ends in punctuation (parakeet en-only)."""
    if not text or text[-1] in "。，,.!?！？:：;；":
        return ""
    if _contains_cjk(text):
        return ""
    return "."


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
    `@rpath/libonnxruntime.<ver>.dylib` fails (_archive/benchmark-2026-08-24.md). Copy the
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


def _load_silero_vad():
    """Silero VAD weights. `torch.hub.load` clones the GitHub repo and imports
    its utils even when cached (slow, needs git, pulls in torchaudio) — load
    the cached TorchScript file directly when present, falling back to
    torch.hub for the first fetch."""
    hub_dir = Path(torch.hub.get_dir())
    for repo_dir in sorted(hub_dir.glob("snakers4_silero-vad*")):
        pt = repo_dir / "silero_vad.pt"
        if pt.is_file():
            model = torch.jit.load(str(pt))
            model.eval()
            return model
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model


class VAD:
    """Voice activity detection"""

    def __init__(self, frame_size=512, sample_rate=16000, speech_threshold=0.4):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.model = _load_silero_vad()

    def is_speech(self, frame):
        if len(frame) != self.frame_size:
            return False
        frame_tensor = torch.from_numpy(frame).float()
        speech_prob = self.model(frame_tensor, self.sample_rate).item()
        return speech_prob > self.speech_threshold


class SherpaSegment:
    """Minimal segment object for the display layer (`.text`, `.probability`,
    `.word_probs`). Sherpa exposes no per-token log-probs for SenseVoice, so
    its `probability` is 1.0; transducer models (parakeet) expose `tokens` +
    `ys_log_probs`, which `_word_probs_from_result` groups into `word_probs` —
    (word, prob) pairs the real-time display colors per word."""

    def __init__(
        self,
        text: str,
        probability: float = 1.0,
        word_probs: list[tuple[str, float]] | None = None,
    ) -> None:
        self.text = text
        self.probability = probability
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

    Model choice (_archive/benchmark-2026-08-24.md, 2026-08-24): en-only ->
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
    if np.sqrt(np.mean(full_segment**2)) < 0.001:
        return None
    max_val = np.max(np.abs(full_segment)) + 1e-8
    full_segment = full_segment / max_val
    full_segment = nr.reduce_noise(y=full_segment, sr=sample_rate)
    return full_segment


class SpeechToText:
    """STT backend dispatch — Phase 1 model-selection winners (_archive/benchmark-2026-08-24.md).
    Sherpa-onnx int8 only since 2026-08-24 (whisper.cpp and Moonshine both
    dropped — see _archive/benchmark-2026-08-24.md for the reasons).

    model_name:
      - "sensevoice"            product default (sherpa int8; dictate/multilingual winner)
      - "parakeet-tdt-0.6b-v2"  en-only / listen option (sherpa int8 transducer)

    Per-word confidence coloring: the sherpa transducer (parakeet) exposes
    per-token log-probs, grouped into word probs in _word_probs_from_result.
    SenseVoice exposes no per-token scores in sherpa-onnx 1.13.0 — its
    output is uncolored (uniform), a known limitation (_archive/benchmark-2026-08-24.md).
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
        clean=False,
        max_window_seconds=60.0,
    ):
        self.audio_file_path = audio_file_path
        self.clean = clean
        self.max_window_seconds = max_window_seconds
        self.max_segment_duration = max_segment_duration
        self.sample_rate = 16000
        self.frame_size = 512
        self.vad, self.translator, self.stt = self._load_models(
            stt_model, translate_model, translate_to
        )
        self.stt_model_name = self.stt.model_name
        self._initialize_state()
        # Calculate max frames based on the configurable duration
        self.max_segment_frames = int(
            self.max_segment_duration * self.sample_rate / self.frame_size
        )

    def _load_models(self, stt_model, translate_model, translate_to):
        """VAD / Translator / STT are independent model stacks — load them
        concurrently so startup waits for the slowest load, not their sum.
        Failures surface in the main thread after the workers join."""
        loaded: dict[str, Any] = {}
        errors: list[BaseException] = []

        def _load(name: str, fn) -> None:
            try:
                loaded[name] = fn()
            except Exception as exc:  # noqa: BLE001 - re-raised in the main thread below
                errors.append(exc)

        threads = [
            threading.Thread(
                target=_load,
                args=("vad", lambda: VAD(self.frame_size, self.sample_rate)),
            ),
            threading.Thread(
                target=_load,
                args=(
                    "translator",
                    lambda: (
                        Translator(model=translate_model, target_lang=translate_to)
                        if translate_to
                        else None
                    ),
                ),
            ),
            threading.Thread(
                target=_load,
                args=("stt", lambda: SpeechToText(stt_model)),
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return loaded["vad"], loaded.get("translator"), loaded["stt"]

    def _initialize_state(self):
        self.ring_buffer_maxlen = 20
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_maxlen)
        self.triggered = False
        self.recorded_frames = []
        self.recorded_frames_count = 0  # Track number of frames recorded
        self.q_for_vad = queue.Queue()
        self.q_for_transcription = queue.Queue()
        self.q_for_translation = queue.Queue()
        self.display_lock = threading.Lock()
        self.transcription_thread = None
        self.vad_thread = None
        self.translation_thread = None
        self.running = False
        self.transcript = []
        self.full_recording_list = []
        self.start_time = time.time()
        # Adaptive stability-window state: `committed_*` hold locked (stable)
        # chunks; `window_*` is the active re-decode window.
        self.committed_chunks = []
        self.committed_styled = []
        self.committed_ts = []
        self.committed_compute = []
        self.committed_translations = []
        self.window_segments = []
        self.window_audio_duration = 0.0
        self.window_plain = None
        self.window_styled = ""
        self.window_ts = None
        self.window_compute = None
        self.window_translation = None
        self._window_seq = 0
        self._last_window_translation_text = ""
        self.final_translation = None
        self._tty = False
        self._header_lines: list[str] = []

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
        parts = [c + _chunk_punct(c) for c in self.committed_chunks if c]
        if self.window_plain:
            parts.append(self.window_plain + _chunk_punct(self.window_plain))
        text = _join_text_parts(parts, force_space=True)
        if not text:
            return
        print("\nTranscript:")
        print(text)

    def _ingest_segment(self, processed_segment):
        """Adaptive stability-window re-decode (growing-buffer re-decode):
        append the segment's audio, re-decode the whole window so the newest
        utterance gets predecessor context, then commit the window once its
        text stops changing (stability check) and slide it forward."""
        self.window_segments.append(processed_segment)
        self.window_audio_duration += len(processed_segment) / self.sample_rate
        start_time = time.time()
        result = self.stt.transcribe(np.concatenate(self.window_segments))
        compute = time.time() - start_time
        if not (isinstance(result, list) and result):
            return
        plain = self.stt.extract_text(result)
        if not plain:
            return
        styled = self._format_transcript(result)
        ts = self._get_time_str()
        prev = self.window_plain
        stable = prev is not None and _stable_prefix(plain, prev)
        forced = self.window_audio_duration > self.max_window_seconds

        if stable or forced:
            if prev is not None:
                self._commit_chunk(
                    prev,
                    self.window_styled,
                    self.window_ts,
                    self.window_compute,
                    self.window_translation,
                )
            if forced and prev is None:
                self._commit_chunk(plain, styled, ts, compute)
                self._reset_window()
                self._redraw()
                return
            # Slide the window to the newest segment and re-decode it alone so
            # the partial is colored and consistent with the next comparison
            # baseline (the corrected-tail version is kept only if the
            # standalone decode comes back empty).
            self.window_translation = None
            self.window_segments = [processed_segment]
            self.window_audio_duration = len(processed_segment) / self.sample_rate
            start_time = time.time()
            result2 = self.stt.transcribe(processed_segment)
            compute2 = time.time() - start_time
            plain2 = (
                self.stt.extract_text(result2)
                if (isinstance(result2, list) and result2)
                else ""
            )
            if plain2:
                self.window_plain = plain2
                self.window_styled = self._format_transcript(result2)
                self.window_ts = ts
                self.window_compute = compute2
                self._enqueue_window_translation(plain2)
            else:
                tail = _strip_prefix(plain, prev) if prev is not None else plain
                if not tail.strip():
                    self._reset_window()
                    self._redraw()
                    return
                self.window_plain = tail
                self.window_styled = tail
                self.window_ts = ts
                self.window_compute = compute
                self._enqueue_window_translation(tail)
        else:
            self.window_plain = plain
            self.window_styled = styled
            self.window_ts = ts
            self.window_compute = compute
            self._enqueue_window_translation(plain)
        self._redraw()

    def _reset_window(self):
        self.window_segments = []
        self.window_audio_duration = 0.0
        self.window_plain = None
        self.window_styled = ""
        self.window_ts = None
        self.window_compute = None
        self.window_translation = None

    def _enqueue_window_translation(self, text):
        """Live translation of the current window, deduplicated on unchanged
        text. Seq numbers let the worker drop stale decodes (latest wins) so
        translation keeps up with the transcript instead of waiting for the
        window to commit."""
        if not self.translator or not text or not text.strip():
            return
        if text == self._last_window_translation_text:
            return
        self._last_window_translation_text = text
        seq = self._window_seq
        self._window_seq += 1
        self.q_for_translation.put(("window", seq, text))

    def _commit_chunk(self, plain, styled, ts, compute, translation=None):
        # Append the parallel lists atomically under the display lock: a
        # translation-thread redraw that reads them mid-append would hit an
        # IndexError and silently kill the worker (translations stopped after
        # the first sentence — see PLAN.md gotcha).
        with self.display_lock:
            self.committed_chunks.append(plain)
            self.committed_styled.append(styled or plain)
            self.committed_ts.append(ts)
            self.committed_compute.append(compute)
            self.committed_translations.append(translation)
        if self.translator and plain.strip() and translation is None:
            self.q_for_translation.put(
                ("commit", len(self.committed_chunks) - 1, plain)
            )

    def _fmt_meta_line(self, ts, styled, compute):
        suffix = f" ({compute:.4f}s)" if compute is not None else ""
        return f"[{ts}] {styled}{suffix}"

    def _render_lines(self):
        if self.clean:
            parts = [c + _chunk_punct(c) for c in self.committed_chunks if c]
            if self.window_plain:
                parts.append(self.window_plain)
            text = _join_text_parts(parts, force_space=True)
            return [text] if text else []
        lines = []
        for i in range(len(self.committed_chunks)):
            lines.append(
                self._fmt_meta_line(
                    self.committed_ts[i],
                    self.committed_styled[i],
                    self.committed_compute[i],
                )
            )
            tr = self.committed_translations[i]
            if tr:
                lines.append(f"    → {tr}")
        if self.window_plain:
            lines.append(
                self._fmt_meta_line(
                    self.window_ts, self.window_styled, self.window_compute
                )
            )
            if self.window_translation:
                lines.append(f"    → {self.window_translation}")
        return lines

    def _redraw(self):
        """Redraw the running transcript. In a tty the redraw runs in the
        alternate screen buffer (vim-style): clear everything and reprint the
        header + transcript, so no cursor-up/erase-line bookkeeping is needed
        and no escape sequences leak into scrollback. Non-tty output prints
        append-only (no ANSI)."""
        with self.display_lock:
            if self._tty:
                sys.stdout.write("\x1b[2J\x1b[H")
            for line in [*self._header_lines, *self._render_lines()]:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _finalize(self):
        parts = list(self.committed_chunks)
        if self.window_plain:
            parts.append(self.window_plain)
        self.transcript = [_join_text_parts(parts)]

    def _transcription_worker(self):
        while self.running:
            try:
                segment = self.q_for_transcription.get(timeout=0.1)
            except queue.Empty:
                continue
            if segment is None:
                break
            processed_segment = process_audio_segment(segment, self.sample_rate)
            if processed_segment is None:
                continue
            self._ingest_segment(processed_segment)

    def _translation_worker(self):
        if self.translator is None:
            return
        while self.running:
            try:
                item = self.q_for_translation.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            kind, *rest = item
            if kind == "commit":
                idx, transcript = rest
            else:
                seq, transcript = rest
                if seq < self._window_seq - 1:
                    continue  # stale window decode — a newer one is queued
            if transcript is None:
                break
            try:
                translate_start = time.time()
                translated = self.translator.translate(transcript)
                translate_time = time.time() - translate_start
                with self.display_lock:
                    if kind == "commit":
                        if idx < len(self.committed_translations):
                            self.committed_translations[idx] = (
                                f"{translated} ({translate_time:.4f}s)"
                            )
                    else:
                        self.window_translation = (
                            f"{translated} ({translate_time:.4f}s)"
                        )
                self._redraw()
            except Exception:  # noqa: S112, BLE001 - best-effort: a failing sentence must not kill the worker and drop every later translation
                continue

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
        if self.translator and self.window_plain and self.window_plain.strip():
            if self.window_translation:
                self.final_translation = self.window_translation
            else:
                translate_start = time.time()
                translated = self.translator.translate(self.window_plain)
                self.final_translation = (
                    f"{translated} ({time.time() - translate_start:.4f}s)"
                )
        self._finalize()
        if self.audio_file_path and self.full_recording_list:
            full_audio = np.concatenate(self.full_recording_list)
            sf.write(self.audio_file_path, full_audio, self.sample_rate)
            print(f"Audio saved to {self.audio_file_path}")

    def run(self):
        self._tty = sys.stdout.isatty()
        self._header_lines = [
            "Real-time transcribe... (Ctrl+C to stop)",
            f"Speech-to-text model: {self.stt_model_name}",
        ]
        if self.translator:
            self._header_lines.append(
                f"Translation model: {self.translator.model} → {self.translator.target_lang}"
            )
        if self._tty:
            sys.stdout.write("\x1b[?1049h")
        self._redraw()
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
            pass
        finally:
            if self._tty:
                sys.stdout.write("\x1b[?1049l")
                sys.stdout.flush()
        print("\nStopping...")
        self._stop()
        if not self.clean and self.translator and self.final_translation:
            print(f"    → {self.final_translation}")
        if self.clean:
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
        # (_archive/benchmark-2026-08-24.md scores zh with CER for this reason), so skip it.
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
