"""en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend.

Dedicated NMT: deterministic, best BLEU on the benchmark corpus (33.59,
_archive/benchmark_translate.md), ~1.2 s/sentence; single pair en->zh. The qwen3.5 LLM
quality mode was dropped 2026-08-24 — live dictation showed hallucinated
content and a meaning-reversed error (PLAN.md, _archive/benchmark_translate.md).

Weights are stored under data/models/translate/<name>/ (e.g. opus-mt-en-zh)
via huggingface_hub snapshot_download with local_dir. This matches the STT
and speaker stores under data/models/{transcribe,speaker} (see interpreter.__init__).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from interpreter import TRANSLATE_MODELS_DIR

TRANSLATE_MODEL = "opus-mt-en-zh"

# short name → HF repo
_HF_IDS: dict[str, str] = {
    "opus-mt-en-zh": "Helsinki-NLP/opus-mt-en-zh",
}


def _ensure_translate_model(name: str) -> Path:
    """Ensure translate model weights are present under data/models/translate/<name>.

    Downloads via snapshot_download on first use (anonymous HF). Idempotent
    via a `.complete` marker.
    """
    repo = _HF_IDS.get(name, f"Helsinki-NLP/{name}")
    dest = TRANSLATE_MODELS_DIR / name
    if (dest / ".complete").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download as _snapshot_download

    _snap: Any = _snapshot_download
    _snap(repo_id=repo, local_dir=str(dest), local_dir_use_symlinks=False)
    (dest / ".complete").touch()
    return dest


class Translator:
    """en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend."""

    def __init__(self, model: str = TRANSLATE_MODEL, target_lang: str = "Chinese"):
        self.model = model
        self.target_lang = target_lang
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        t0 = time.perf_counter()
        dest = TRANSLATE_MODELS_DIR / model
        # fast path: already downloaded under data/models/translate
        if (dest / ".complete").exists():
            try:
                self._nmt_tokenizer: Any = AutoTokenizer.from_pretrained(
                    str(dest), local_files_only=True, trust_remote_code=False
                )
                self._nmt_model: Any = AutoModelForSeq2SeqLM.from_pretrained(
                    str(dest), local_files_only=True, trust_remote_code=False
                )
                self.load_s = time.perf_counter() - t0
                return
            except OSError:
                pass
        # ensure download (first use)
        try:
            _ensure_translate_model(model)
            self._nmt_tokenizer = AutoTokenizer.from_pretrained(
                str(dest), local_files_only=True, trust_remote_code=False
            )
            self._nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(dest), local_files_only=True, trust_remote_code=False
            )
            self.load_s = time.perf_counter() - t0
            return
        except Exception:  # noqa: BLE001, S110 - fallback to HF cache
            pass
        # last resort: HF cache (internet) — mirrors pre-2026-08-26 behavior
        model_id = _HF_IDS.get(model, f"Helsinki-NLP/{model}")
        try:
            self._nmt_tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=True
            )
            self._nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, local_files_only=True
            )
        except OSError:
            self._nmt_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._nmt_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.load_s = time.perf_counter() - t0

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        inputs = self._nmt_tokenizer(text, return_tensors="pt", truncation=True)
        out = self._nmt_model.generate(**inputs, max_length=256)
        return self._nmt_tokenizer.decode(out[0], skip_special_tokens=True).strip()
