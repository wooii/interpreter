"""Real-time speech-to-text with adaptive stability-window.

Product models (single source; see PLAN.md): ``parakeet-unified-en-0.6b``
(en-only/offline, listen) and ``sensevoicesmall`` (mixed zh/en, dictate);
translate ``opus-mt-en-zh``. CLI picks models internally
(``python -m interpreter listen|dictate``); library accepts explicit names.
Supports live mic (``RealTimeTranscribe``), replay (``input_path``) and
standalone (``SpeechToText``/``Translator``). Weights under
``data/models/`` via HF; full selection record in ``_archive/``.
"""

from __future__ import annotations

import collections
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import noisereduce as nr
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from jiwer import cer, wer

from interpreter import TRANSCRIBE_MODELS_DIR
from interpreter.common import _contains_cjk, _is_cjk_char, ensure_onnxruntime
from interpreter.speaker import UNCERTAIN, SpeakerAssigner
from interpreter.translate import Translator

STT_MODEL_EN_ONLY = "parakeet-unified-en-0.6b"
STT_MODEL_MIXED = "sensevoicesmall"


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


def _strip_timing(text: str) -> str:
    """Drop the "(0.1234s)" timing suffix the translation worker appends."""
    if text.endswith("s)") and " (" in text:
        return text.rsplit(" (", 1)[0]
    return text


def _flowing_text(chunk_plains: list[str], window_plain: str | None) -> str:
    """The clean (dictate) rendering: every committed chunk with punctuation
    appended, plus the live window, joined as one flowing string."""
    parts = [c + _chunk_punct(c) for c in chunk_plains if c]
    if window_plain:
        parts.append(window_plain)
    return _join_text_parts(parts, force_space=True)


def _flowing_styled_parts(
    chunk_plains: list[str],
    chunk_styleds: list[str],
    window_plain: str | None,
    window_styled: str | None,
) -> list[str]:
    """Styled counterpart of `_flowing_text`'s parts: the per-word colored
    twin of each committed chunk (punctuation appended after the color reset)
    plus the live window's styled twin. Used only when the snapshot actually
    carries confidence colors (en-only/parakeet — see `_has_confidence_colors`);
    the renderer maps the ANSI colors to its own format and joins with spaces."""
    parts: list[str] = []
    for plain, styled in zip(chunk_plains, chunk_styleds):
        if not plain:
            continue
        parts.append((styled or plain) + _chunk_punct(plain))
    if window_plain:
        parts.append(window_styled or window_plain)
    return parts


def _has_confidence_colors(snapshot: TranscriptSnapshot) -> bool:
    """True when the snapshot's styled twins carry real per-word confidence
    colors (`\x1b[38;2;…m`). Only transducer models (parakeet en-only) expose
    per-token log-probs; SenseVoice has none, so its styled stays plain and
    mixed dictate renders uncolored."""
    return any("\x1b[38;2;" in c.styled for c in snapshot.chunks) or (
        snapshot.window is not None and "\x1b[38;2;" in snapshot.window.styled
    )


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


def _strip_styled_prefix(styled: str, n_visible: int) -> str:
    """Colored suffix of a colored string after the first `n_visible` visible
    (non-space) characters, preserving the ANSI per-word coloring. `styled`'s
    visible characters match the plain text's, so the same cut applies to
    both (the slide baseline's tail must stay colored — the "color coding is
    broken" report)."""
    i = 0
    seen = 0
    n = len(styled)
    while i < n and seen < n_visible:
        if styled.startswith("\x1b[", i):
            j = styled.index("m", i)
            i = j + 1
        elif styled[i].isspace():
            i += 1
        else:
            seen += 1
            i += 1
    while i < n and styled[i].isspace():
        i += 1
    return styled[i:]


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


MODEL_SPECS: dict[str, dict] = {
    "parakeet-unified-en-0.6b": {
        "repo": "csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
        "files": {
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        "factory": "from_transducer",
        "kwargs": {"model_type": "nemo_transducer"},
    },
    "sensevoicesmall": {
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
    dest = TRANSCRIBE_MODELS_DIR / name
    if (dest / ".complete").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for rel in spec["files"].values():
        hf_hub_download(spec["repo"], rel, repo_type="model", local_dir=dest)
    (dest / ".complete").touch()


def _ensure_onnxruntime_dylib() -> None:
    """Deprecated alias for :func:`interpreter.common.ensure_onnxruntime`."""
    ensure_onnxruntime()


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

    Model choice (PLAN.md 2026-08-25): en-only -> parakeet-unified-en-0.6b
    (offline mode of the unified model — beats parakeet-tdt-0.6b-v2 at every
    segment cap on the Phase 3 probe; true streaming is not real-time on the
    container), mixed -> sensevoicesmall; both sherpa-onnx int8
    (whisper.cpp/Moonshine dropped). Weights download anonymously from HF.
    """

    def __init__(self, model_name: str, num_threads: int = 4) -> None:
        spec = MODEL_SPECS[model_name]
        self.model_name = model_name
        _download_model_files(model_name)
        ensure_onnxruntime()
        import sherpa_onnx

        kwargs: dict[str, object] = {
            key: str(TRANSCRIBE_MODELS_DIR / model_name / rel)
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
    """STT backend dispatch — Phase 1 model-selection winners (_archive/benchmark_transcribe.md).
    Sherpa-onnx int8 only since 2026-08-24 (whisper.cpp and Moonshine both
    dropped — see _archive/benchmark_transcribe.md for the reasons).

    model_name:
      - "sensevoicesmall"              product default (sherpa int8; dictate/multilingual winner)
      - "parakeet-unified-en-0.6b"    en-only / listen default (sherpa int8 transducer; offline mode —
                                      streaming not real-time on the container, PLAN.md 2026-08-25)

    Per-word confidence coloring: the sherpa transducer (parakeet) exposes
    per-token log-probs, grouped into word probs in _word_probs_from_result.
    SenseVoice exposes no per-token scores in sherpa-onnx 1.13.0 — its
    output is uncolored (uniform), a known limitation (_archive/benchmark_transcribe.md).
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
        if self.model_name == "sensevoicesmall":
            text = _normalize_sensevoice_case(text)
        return text


@dataclass(frozen=True)
class ChunkSnapshot:
    plain: str
    styled: str
    ts: str
    compute: float | None
    translation: str | None
    speaker: str | None


@dataclass(frozen=True)
class WindowSnapshot:
    plain: str
    styled: str
    ts: str | None
    compute: float | None
    translation: str | None
    speaker: str | None


@dataclass(frozen=True)
class TranscriptSnapshot:
    """Immutable view of the live transcript state, built under display_lock.
    Renderers (terminal, GUI) consume this and never touch engine state."""

    stt_model: str
    translator_model: str | None
    target_lang: str | None
    speaker_names: tuple[str, ...] | None  # None = speaker ID off
    chunks: tuple[ChunkSnapshot, ...]
    window: WindowSnapshot | None


class BaseRenderer:
    """Rendering hook: receives immutable snapshots of the live state. The
    terminal renderer is the default; a GUI renderer (see app.py) consumes the
    same snapshots. All methods are called from engine worker threads."""

    def begin(self) -> None:
        """Called once when run() starts."""

    def render(self, snapshot: TranscriptSnapshot) -> None:
        """Called on every state change (window re-decode, commit, translation)."""

    def end(self) -> None:
        """Called once after run() stops (session already finalized)."""


class TerminalRenderer(BaseRenderer):
    """Reproduce the CLI's live terminal output exactly from snapshots. The
    alternate-screen enter/exit and the dim "[?]" styling depend on `tty`."""

    def __init__(self, clean: bool = False) -> None:
        self.clean = clean

    def begin(self) -> None:
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[?1049h")

    def end(self) -> None:
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[?1049l")
            sys.stdout.flush()

    def render(self, snapshot: TranscriptSnapshot) -> None:
        tty = sys.stdout.isatty()
        if tty:
            sys.stdout.write("\x1b[2J\x1b[H")
        for line in [*self._header_lines(snapshot), *self._lines(snapshot, tty)]:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _lines(self, snapshot: TranscriptSnapshot, tty: bool) -> list[str]:
        if self.clean:
            if _has_confidence_colors(snapshot):
                text = _join_text_parts(
                    _flowing_styled_parts(
                        [c.plain for c in snapshot.chunks],
                        [c.styled for c in snapshot.chunks],
                        snapshot.window.plain if snapshot.window else None,
                        snapshot.window.styled if snapshot.window else None,
                    ),
                    force_space=True,
                )
            else:
                text = _flowing_text(
                    [c.plain for c in snapshot.chunks],
                    snapshot.window.plain if snapshot.window else None,
                )
            return [text] if text else []
        lines: list[str] = []
        for c in snapshot.chunks:
            lines.append(self._fmt_meta_line(c.ts, c.styled, c.compute, c.speaker, tty))
            if c.translation:
                lines.append(f"    → {c.translation}")
        if snapshot.window is not None:
            lines.append(
                self._fmt_meta_line(
                    snapshot.window.ts,
                    snapshot.window.styled,
                    snapshot.window.compute,
                    snapshot.window.speaker,
                    tty,
                )
            )
            if snapshot.window.translation:
                lines.append(f"    → {snapshot.window.translation}")
        return lines

    @staticmethod
    def _fmt_meta_line(ts, styled, compute, speaker, tty) -> str:
        prefix = f"[{speaker}] " if speaker else ""
        if speaker == UNCERTAIN and tty:
            prefix = "\033[2m[?]\033[0m "
        suffix = f" ({compute:.4f}s)" if compute is not None else ""
        return f"{prefix}[{ts}] {styled}{suffix}"

    @staticmethod
    def _header_lines(snapshot: TranscriptSnapshot) -> list[str]:
        lines = [
            "Real-time transcribe... (Ctrl+C to stop)",
            f"Speech-to-text model: {snapshot.stt_model}",
        ]
        if snapshot.speaker_names is not None:
            names = ", ".join(snapshot.speaker_names) or "waiting for first voice"
            lines.append(f"Speaker ID: WeSpeaker en (auto-assign: {names})")
        if snapshot.translator_model:
            lines.append(
                f"Translation model: {snapshot.translator_model} → "
                f"{snapshot.target_lang}"
            )
        return lines


class RealTimeTranscribe:
    def __init__(
        self,
        audio_file_path=None,
        stt_model=STT_MODEL_MIXED,
        translate_model="opus-mt-en-zh",
        translate_to="Chinese",
        max_segment_duration=4.0,
        clean=False,
        max_window_seconds=8.0,
        max_window_segments=3,
        speaker_id=False,
        input_path=None,
        resume_from=None,
        renderer: BaseRenderer | None = None,
        quiet: bool = False,
    ):
        self.audio_file_path = audio_file_path
        self.input_path = input_path
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.clean = clean
        self.quiet = quiet
        self.renderer = (
            renderer if renderer is not None else TerminalRenderer(clean=clean)
        )
        self.max_window_seconds = max_window_seconds
        self.max_window_segments = max_window_segments
        self.max_segment_duration = max_segment_duration
        self.sample_rate = 16000
        self.frame_size = 512
        self.vad, self.translator, self.stt, self.speaker = self._load_models(
            stt_model, translate_model, translate_to, speaker_id
        )
        self.stt_model_name = self.stt.model_name
        self._initialize_state()
        # Calculate max frames based on the configurable duration
        self.max_segment_frames = int(
            self.max_segment_duration * self.sample_rate / self.frame_size
        )

    def _load_models(self, stt_model, translate_model, translate_to, speaker_id):
        """VAD / Translator / STT / Speaker are independent model stacks — load
        them concurrently so startup waits for the slowest load, not their sum.
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
            threading.Thread(
                target=_load,
                args=("speaker", lambda: SpeakerAssigner() if speaker_id else None),
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return (
            loaded["vad"],
            loaded.get("translator"),
            loaded["stt"],
            loaded.get("speaker"),
        )

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
        self.committed_speakers = []
        self._segment_speaker = None
        self._window_speakers: list[str | None] = []
        self._window_pending_ids: list[int | None] = []
        self._pending_chunk: dict[int, int] = {}
        self._chunk_assignments: dict[int, list[str | None]] = {}
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

    def _audio_callback(self, indata, frames, time_info, status):
        if status and not self.quiet:
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
        # Colored text for the transcription. Per-word confidence from the
        # sherpa transducer's word_probs (when present) — skips empty
        # segments/words. Segments WITHOUT per-token scores (SenseVoice)
        # stay plain: coloring them with the uniform 1.0 probability was
        # fake confidence, not an absence of color.
        parts = []
        for seg in result:
            word_probs = getattr(seg, "word_probs", None)
            if word_probs:
                parts.append(" ".join(self._color_word(w, p) for w, p in word_probs))
            elif seg.text.strip():
                parts.append(seg.text.strip())
        return " ".join(parts).strip()

    def _get_time_str(self):
        elapsed = time.time() - self.start_time
        return f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}"

    def _print_clean_transcript(self):
        parts = [c + _chunk_punct(c) for c in self.committed_chunks if c]
        if self.window_plain:
            parts.append(self.window_plain + _chunk_punct(self.window_plain))
        text = _join_text_parts(parts, force_space=True)
        if not text:
            return
        print("\nTranscript:")
        print(text)

    def _ingest_segment(
        self,
        processed_segment,
        speaker_emb=None,
        speaker_duration_s=None,
        speaker_rms=None,
    ):
        """Adaptive stability-window re-decode (growing-buffer re-decode):
        append the segment's audio, re-decode the whole window so the newest
        utterance gets predecessor context, then commit the window once its
        text stops changing (stability check) and slide it forward."""
        self.window_segments.append(processed_segment)
        self._window_speakers.append(None)
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
        # Bound the window even when the text never stabilizes: SenseVoice
        # re-decodes a growing buffer with drifting first words ("OKY" ->
        # "OKAY" -> "ALL RIGHT"), so prefix stability can fail forever and the
        # whole session would render as one never-committing line. A small
        # segment cap commits ~2-segment chunks while keeping re-decode
        # context (accuracy).
        window_capped = len(self.window_segments) >= self.max_window_segments

        new_speaker = (
            self.speaker.assign_embedding(
                speaker_emb,
                duration_s=speaker_duration_s,
                rms=speaker_rms,
            )
            if self.speaker is not None and speaker_emb is not None
            else None
        )
        self._window_speakers[-1] = new_speaker
        # Per-speaker segmentation: when the newest segment belongs to a
        # different speaker than the current window, close the window now —
        # a line must only ever append the RIGHT speaker's speech (the "my
        # voice gets appended to other" report: the live window accumulated
        # mixed speakers and its tag followed the newest segment). The slide
        # baseline stays context-rich (windowed tail, not a standalone
        # decode), so the per-speaker commits don't cost accuracy.
        speaker_changed = (
            prev is not None
            and new_speaker is not None
            and new_speaker != self._segment_speaker
        )
        # Pending id of the newest segment (None unless it was flagged "?").
        # Committed chunks are mapped back to the pending embeddings that
        # produced their speaker tag, so a promotion can retroactively relabel
        # exactly the right chunks.
        pid_now = (
            self.speaker.pending_count() - 1
            if new_speaker == UNCERTAIN and self.speaker is not None
            else None
        )
        self._window_pending_ids.append(pid_now)

        if stable or forced or window_capped or speaker_changed:
            if prev is not None:
                # The committed chunk covers the window before the newest
                # segment; tag it by its segments' assignments — all the same
                # -> that speaker, mixed -> UNCERTAIN (a mixed window can't be
                # cleanly labeled, and per-segment windows would sacrifice the
                # re-decode context that drives transcription accuracy).
                self._commit_chunk(
                    prev,
                    self.window_styled,
                    self.window_ts,
                    self.window_compute,
                    self.window_translation,
                    speaker=self._window_chunk_speaker(include_newest=False),
                )
                idx = len(self.committed_chunks) - 1
                for pid in self._window_pending_ids[:-1]:
                    if pid is not None:
                        self._pending_chunk[pid] = idx
                self._chunk_assignments[idx] = self._window_speakers[:-1]
            if (forced or window_capped) and prev is None:
                self._commit_chunk(
                    plain,
                    styled,
                    ts,
                    compute,
                    speaker=self._window_chunk_speaker(include_newest=True),
                )
                idx = len(self.committed_chunks) - 1
                for pid in self._window_pending_ids:
                    if pid is not None:
                        self._pending_chunk[pid] = idx
                self._chunk_assignments[idx] = self._window_speakers
                self._reset_window()
                self._relabel_promoted()
                self._redraw()
                return
            # Slide the window to the newest segment. Its baseline is its
            # portion of the WINDOWED decode (the tail after the committed
            # `prev` — context-rich and complete, unlike a standalone decode,
            # which truncates tails e.g. "...WHICH HAS A PU" -> "cut off in a
            # not so nice point"), and its styled twin keeps the per-word
            # coloring. The standalone decode is only a fallback (unstable
            # window where the prefix strip is unsafe, or an empty tail). An
            # empty baseline is kept, never reset away — the segment's text
            # must not vanish ("some of my speech don't append").
            self.window_translation = None
            self.window_segments = [processed_segment]
            self._window_speakers = [new_speaker]
            self._window_pending_ids = [pid_now]
            self.window_audio_duration = len(processed_segment) / self.sample_rate
            if prev is not None and _stable_prefix(plain, prev):
                baseline = _strip_prefix(plain, prev)
                baseline_styled = _strip_styled_prefix(styled, len(_norm(prev)))
            else:
                baseline = ""
                baseline_styled = ""
            if baseline.strip():
                self.window_compute = compute
            else:
                start_time = time.time()
                result2 = self.stt.transcribe(processed_segment)
                self.window_compute = time.time() - start_time
                baseline = (
                    self.stt.extract_text(result2)
                    if (isinstance(result2, list) and result2)
                    else ""
                )
                baseline_styled = (
                    self._format_transcript(result2)
                    if (isinstance(result2, list) and result2)
                    else ""
                )
            self.window_plain = baseline
            self.window_styled = baseline_styled
            self.window_ts = ts
            self._enqueue_window_translation(baseline)
        else:
            self.window_plain = plain
            self.window_styled = styled
            self.window_ts = ts
            self.window_compute = compute
            self._enqueue_window_translation(plain)
        self._relabel_promoted()
        self._segment_speaker = new_speaker
        self._redraw()

    def _relabel_promoted(self):
        """A pending-cluster promotion retroactively labels the earlier "?"
        chunks whose segments joined the promoted cluster — the ambiguous
        voice turned out to be a real, distinct speaker, so the display
        shouldn't keep it as "?". Runs after the commit so a chunk committed
        in this very call is covered too.

        A chunk is relabeled ONLY when it belongs entirely to the promoted
        cluster — every covered segment's assignment is either UNCERTAIN or
        the promoted name, and every pending segment of the chunk is in the
        promoted set. A chunk that also contains another speaker's segments
        (e.g. the primary voice sharing a window with the device audio) stays
        "[?]" — relabeling it would show one speaker's words under another's
        label ("my voice merged with others")."""
        promoted = (
            self.speaker.consume_promotion() if self.speaker is not None else None
        )
        if not promoted:
            return
        name, pids = promoted
        promoted_ids = set(pids)
        with self.display_lock:
            for idx in self._chunk_assignments:
                assigns = self._chunk_assignments[idx]
                if any(a not in (None, UNCERTAIN, name) for a in assigns):
                    continue
                chunk_pids = [
                    pid for pid, cidx in self._pending_chunk.items() if cidx == idx
                ]
                if any(pid not in promoted_ids for pid in chunk_pids):
                    continue
                if (
                    idx < len(self.committed_speakers)
                    and self.committed_speakers[idx] == UNCERTAIN
                ):
                    self.committed_speakers[idx] = name
        self._pending_chunk = {}
        self._chunk_assignments = {}

    def _reset_window(self):
        self.window_segments = []
        self._window_speakers = []
        self._window_pending_ids = []
        self.window_audio_duration = 0.0
        self.window_plain = None
        self.window_styled = ""
        self.window_ts = None
        self.window_compute = None
        self.window_translation = None

    def _window_chunk_speaker(self, include_newest: bool) -> str | None:
        """Tag for the chunk being committed: the assignments of the segments
        it covers. All the same -> that name; mixed -> UNCERTAIN (the window
        spans speakers, so no single label is honest); none -> None."""
        assigns = (
            self._window_speakers if include_newest else self._window_speakers[:-1]
        )
        distinct = {a for a in assigns if a is not None}
        if len(distinct) == 1:
            return distinct.pop()
        return UNCERTAIN if distinct else None

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

    def _commit_chunk(self, plain, styled, ts, compute, translation=None, speaker=None):
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
            self.committed_speakers.append(speaker)
        if self.translator and plain.strip() and translation is None:
            self.q_for_translation.put(
                ("commit", len(self.committed_chunks) - 1, plain)
            )

    def _build_snapshot(self) -> TranscriptSnapshot:
        """Immutable view of the live state for renderers, taken under the
        display lock (the parallel committed_* lists are appended atomically
        there — a renderer reading them mid-append would hit an IndexError)."""
        with self.display_lock:
            chunks = tuple(
                ChunkSnapshot(
                    plain=self.committed_chunks[i],
                    styled=self.committed_styled[i],
                    ts=self.committed_ts[i],
                    compute=self.committed_compute[i],
                    translation=(
                        self.committed_translations[i]
                        if i < len(self.committed_translations)
                        else None
                    ),
                    speaker=(
                        self.committed_speakers[i]
                        if i < len(self.committed_speakers)
                        else None
                    ),
                )
                for i in range(len(self.committed_chunks))
            )
            window = None
            if self.window_plain:
                window = WindowSnapshot(
                    plain=self.window_plain,
                    styled=self.window_styled,
                    ts=self.window_ts,
                    compute=self.window_compute,
                    translation=self.window_translation,
                    speaker=self._window_chunk_speaker(include_newest=True),
                )
            speaker_names = (
                tuple(self.speaker.speaker_names) if self.speaker is not None else None
            )
            return TranscriptSnapshot(
                stt_model=self.stt_model_name,
                translator_model=(
                    self.translator.model if self.translator is not None else None
                ),
                target_lang=(
                    self.translator.target_lang if self.translator is not None else None
                ),
                speaker_names=speaker_names,
                chunks=chunks,
                window=window,
            )

    def _redraw(self):
        """Push the current state to the renderer. The engine holds no display
        concern — the default TerminalRenderer reproduces the CLI's alternate-
        screen redraw from the snapshot; a GUI renderer consumes the same data."""
        self.renderer.render(self._build_snapshot())

    def _finalize(self):
        parts = list(self.committed_chunks)
        if self.window_plain:
            parts.append(self.window_plain)
        self.transcript = [_join_text_parts(parts)]

    def _save_transcript(self):
        """Write the session transcript as plain text next to the audio file,
        plus a `.styled` twin that keeps the per-word ANSI confidence colors
        (the GUI re-renders it with the original color coding). Dictate
        (clean): the single clean text, as printed at the end. Listen: one
        line per chunk with timestamp + speaker tag, translations indented
        under their chunk — no ANSI colors in the plain file; the styled twin
        mirrors the same lines with the colored chunk text."""
        if self.audio_file_path is None:
            return
        lines: list[str] = []
        styled_lines: list[str] = []
        if self.clean:
            parts = [c + _chunk_punct(c) for c in self.committed_chunks if c]
            if self.window_plain:
                parts.append(self.window_plain + _chunk_punct(self.window_plain))
            text = _join_text_parts(parts, force_space=True)
            if text:
                lines.append(text)
                styled_lines.append(
                    " ".join(
                        _flowing_styled_parts(
                            list(self.committed_chunks),
                            list(self.committed_styled),
                            self.window_plain,
                            self.window_styled,
                        )
                    )
                )
        else:
            for i, chunk in enumerate(self.committed_chunks):
                speaker = (
                    self.committed_speakers[i]
                    if i < len(self.committed_speakers)
                    else None
                )
                line = f"[{self.committed_ts[i]}]"
                if speaker:
                    line += f" [{speaker}]"
                lines.append(f"{line} {chunk}")
                styled_chunk = (
                    self.committed_styled[i]
                    if i < len(self.committed_styled)
                    else chunk
                )
                styled_lines.append(f"{line} {styled_chunk}")
                tr = self.committed_translations[i]
                if tr:
                    lines.append(f"    → {_strip_timing(tr)}")
                    styled_lines.append(f"    → {_strip_timing(tr)}")
            if self.window_plain:
                speaker = self._window_chunk_speaker(include_newest=True)
                line = f"[{self.window_ts}]"
                if speaker:
                    line += f" [{speaker}]"
                lines.append(f"{line} {self.window_plain}")
                styled_lines.append(f"{line} {self.window_styled or self.window_plain}")
                if self.final_translation:
                    lines.append(f"    → {_strip_timing(self.final_translation)}")
                    styled_lines.append(
                        f"    → {_strip_timing(self.final_translation)}"
                    )
        if not lines:
            return
        txt_path = self.audio_file_path.with_suffix(".txt")
        styled_path = self.audio_file_path.with_suffix(".styled")
        # Resume: append the new lines to the CURRENT on-disk content (read
        # before the write below), so earlier lines — hand-edits included —
        # survive. Only when the old audio still exists, matching the audio
        # merge in `_stop`.
        old_lines: list[str] = []
        old_styled_lines: list[str] = []
        if self.resume_from is not None and self.resume_from.is_file():
            if txt_path.is_file():
                old_lines = txt_path.read_text(encoding="utf-8").splitlines()
            if styled_path.is_file():
                old_styled_lines = styled_path.read_text(encoding="utf-8").splitlines()
        txt_path.write_text("\n".join(old_lines + lines) + "\n", encoding="utf-8")
        styled_path.write_text(
            "\n".join(old_styled_lines + styled_lines) + "\n", encoding="utf-8"
        )
        if not self.quiet:
            print(f"Transcript saved to {txt_path}")

    def _transcription_worker(self):
        while self.running:
            try:
                segment = self.q_for_transcription.get(timeout=0.1)
            except queue.Empty:
                continue
            if segment is None:
                break
            # Speaker ID embeds the RAW segment: noisereduce + peak-normalization
            # compress the embedding space and merge distinct voices (probe
            # calibration also used raw clips). STT still gets the processed audio.
            speaker_emb = (
                self.speaker.embed(segment, self.sample_rate)
                if self.speaker is not None
                else None
            )
            processed_segment = process_audio_segment(segment, self.sample_rate)
            if processed_segment is None:
                continue
            self._ingest_segment(
                processed_segment,
                speaker_emb,
                speaker_duration_s=len(segment) / self.sample_rate,
                speaker_rms=float(np.sqrt(np.mean(segment**2))),
            )

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
            if self.audio_file_path.suffix.lower() == ".flac":
                full_audio = np.clip(full_audio, -1.0, 1.0)
                full_audio = (full_audio * 32767).astype(np.int16)
            if self.resume_from is not None and self.resume_from.is_file():
                # Resume: the new audio appends to the previous recording (the
                # old file is read BEFORE the merged write below).
                old_audio, old_sr = sf.read(self.resume_from, dtype="int16")
                if old_sr != self.sample_rate:
                    raise ValueError(
                        f"cannot resume: existing recording is {old_sr} Hz, "
                        f"the engine records at {self.sample_rate} Hz "
                        f"({self.resume_from})"
                    )
                full_audio = np.concatenate([old_audio, full_audio])
            sf.write(self.audio_file_path, full_audio, self.sample_rate)
            if not self.quiet:
                print(f"Audio saved to {self.audio_file_path}")
            self._save_transcript()

    def stop(self):
        """Request a clean stop (public wrapper): flips `running` so `run()`'s
        mic loop exits on its own, then run() finalizes and writes the session
        files — no callback can race the file write because the InputStream has
        already closed by then. Safe to call from any thread."""
        self.running = False

    def run(self):
        self.renderer.begin()
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
            if self.input_path is not None:
                self._run_file_source()
            else:
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
            self.renderer.end()
        if not self.quiet:
            print("\nStopping...")
        self._stop()
        if not self.quiet:
            if not self.clean and self.translator and self.final_translation:
                print(f"    → {self.final_translation}")
            if self.clean:
                self._print_clean_transcript()

    def _run_file_source(self):
        """Replay an audio file through the SAME live pipeline (VAD → segments
        → stability-window re-decode → commits) instead of the mic. Frames are
        fed exactly as `_audio_callback` would feed them, then a trailing
        silence flushes the VAD so the last segment closes. The input is saved
        to `audio_file_path` (as in mic sessions) so `evaluate()` can score the
        committed transcript against an offline re-transcribe of the same audio.
        16 kHz mono required (same as the live input)."""
        assert self.input_path is not None  # only called when run() has a file source
        assert self.vad_thread is not None  # started in run() before this is invoked
        assert self.transcription_thread is not None
        data, sr = sf.read(self.input_path, dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(
                f"input_path must be {self.sample_rate} Hz, got {sr} Hz "
                f"({self.input_path})"
            )
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        audio = data.flatten()
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i : i + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            self.q_for_vad.put(frame)
        # Trailing silence closes the last VAD segment (same logic as a pause
        # on the mic: 0.8 * ring maxlen non-speech frames after speech).
        silence = np.zeros(
            self.frame_size * (self.ring_buffer_maxlen + 1), dtype=np.float32
        )
        for i in range(0, len(silence), self.frame_size):
            self.q_for_vad.put(silence[i : i + self.frame_size])
        self.full_recording_list.append(audio)
        self.q_for_vad.put(None)
        # Wait for the VAD worker to drain, then for the transcription worker
        # to finish the last segments before run() tears the threads down.
        while self.vad_thread.is_alive():
            time.sleep(0.05)
        while not self.q_for_transcription.empty():
            time.sleep(0.05)
        time.sleep(0.1)

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
        # (_archive/benchmark_transcribe.md scores zh with CER for this reason), so skip it.
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
        print(f"Reference Transcript: \n {self.reference_transcript}")
        print(f"Realtime Transcript: \n {self.realtime_transcript}")
        return metrics
