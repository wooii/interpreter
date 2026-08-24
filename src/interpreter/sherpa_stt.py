"""Sherpa-onnx STT backends for the product (Phase 1 model-selection winners).

Model choice (docs/benchmark.md, 2026-08-24):
- listen / en-only       -> parakeet-tdt-0.6b-v2 (sherpa int8 transducer)
- dictate / multilingual -> sensevoice (sherpa int8)

Weights live under data/benchmark/transcribe/models/ (gitignored under
`/data/`) and are downloaded anonymously from HF on first use — no
auth-gated downloads (PLAN.md, 2026-08-24).

Note: sherpa-onnx dlopens `libonnxruntime.so`; the container has a symlink
in /usr/local/lib (default loader path) so no LD_LIBRARY_PATH is needed.
On macOS the wheel doesn't bundle onnxruntime at all — `_ensure_onnxruntime_dylib`
copies it from the installed onnxruntime package next to the sherpa lib dir.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from interpreter import DATA_DIR

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
    `@rpath/libonnxruntime.<ver>.dylib` fails (docs/benchmark.md, Phase 2
    concern). Copy the dylibs from the installed onnxruntime package into
    the sherpa package's lib dir — the first @rpath search location.
    dyld reads DYLD_* at exec time, so a runtime env tweak can't fix this."""
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


class SherpaSegment:
    """Minimal segment object for the display layer (`.text`, `.probability`,
    `.t0`, `.t1`). Sherpa exposes no per-token log-probs for SenseVoice, so
    its `probability` is 1.0; transducer models (parakeet) expose `tokens` +
    `ys_log_probs`, which `_word_probs_from_result` groups into `word_probs` —
    (word, prob) pairs the real-time display colors per word (transcribe.py)."""

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
    """Offline sherpa-onnx recognizer over raw 16 kHz mono float32 audio."""

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
