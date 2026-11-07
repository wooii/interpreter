"""en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend.

Dedicated NMT: deterministic, best BLEU on the benchmark corpus (33.59,
docs/benchmark_translate.md), ~1.2 s/sentence; single pair en->zh. The qwen3.5 LLM
quality mode was dropped 2026-08-24 — live dictation showed hallucinated
content and a meaning-reversed error (PLAN.md, docs/benchmark_translate.md).

Weights are stored under data/models/translate/<name>/ (e.g. opus-mt-en-zh)
via selected-file ``hf_hub_download`` calls with local_dir (only the Marian
config/tokenizer/PyTorch files; the repo's TensorFlow/Rust/Flax copies are
skipped). This matches the STT and speaker stores under
data/models/{transcribe,speaker} (see interpreter.__init__).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from interpreter import TRANSLATE_MODELS_DIR
from interpreter.common import (
    OperationCancelled,
    ProgressCallback,
    check_cancel,
)
from interpreter.progress import download_guard, hf_tqdm_class

TRANSLATE_MODEL = "opus-mt-en-zh"

# short name → HF repo
_HF_IDS: dict[str, str] = {
    "opus-mt-en-zh": "Helsinki-NLP/opus-mt-en-zh",
}

# Only the files Marian needs. The repo also ships TensorFlow/Rust/Flax
# weights (~1.5 GB); downloading them wasted time and space.
_TRANSLATE_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "source.spm",
    "special_tokens_map.json",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
}


def _ensure_translate_model(
    name: str,
    *,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Ensure translate model weights are present under data/models/translate/<name>.

    Downloads the Marian files only (not the unused TF/Rust/Flax copies) on
    first use, anonymously from HF. Idempotent via a ``.complete`` marker.
    """
    repo = _HF_IDS.get(name, f"Helsinki-NLP/{name}")
    dest = TRANSLATE_MODELS_DIR / name
    if (dest / ".complete").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    with download_guard():
        check_cancel(cancel)
        files = [
            rel
            for rel in HfApi().list_repo_files(repo_id=repo)
            if rel in _TRANSLATE_FILES
        ]
        weights = [
            rel for rel in files if rel in ("pytorch_model.bin", "model.safetensors")
        ]
        if not weights:
            raise RuntimeError(f"{repo}: no PyTorch/safetensors weights found")
        # Small config/tokenizer files first; the big weights file last.
        files.sort(key=lambda rel: rel in weights)

        for index, rel in enumerate(files, start=1):
            check_cancel(cancel)
            label = f"Downloading translation model ({index}/{len(files)})"

            def report(
                text: str,
                fraction: float | None,
                *,
                _index: int = index,
            ) -> None:
                if progress is None:
                    return
                if fraction is None:
                    progress(text, None)
                else:
                    progress(text, (_index - 1 + fraction) / len(files))

            hf_hub_download(
                repo_id=repo,
                filename=rel,
                local_dir=str(dest),
                tqdm_class=(
                    hf_tqdm_class(label, report, cancel)
                    if (progress is not None or cancel is not None)
                    else None
                ),
            )
            check_cancel(cancel)
            if progress is not None:
                progress(label, index / len(files))

        (dest / ".complete").touch()
    return dest


class Translator:
    """en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend."""

    def __init__(
        self,
        model: str = TRANSLATE_MODEL,
        target_lang: str = "Chinese",
        *,
        progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ):
        self.model = model
        self.target_lang = target_lang
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        check_cancel(cancel)
        t0 = time.perf_counter()
        dest = TRANSLATE_MODELS_DIR / model
        # fast path: already downloaded under data/models/translate
        if (dest / ".complete").exists():
            try:
                if progress is not None:
                    progress("Loading translation model", None)
                self._nmt_tokenizer: Any = AutoTokenizer.from_pretrained(
                    str(dest), local_files_only=True, trust_remote_code=False
                )
                self._nmt_model: Any = AutoModelForSeq2SeqLM.from_pretrained(
                    str(dest), local_files_only=True, trust_remote_code=False
                )
                check_cancel(cancel)
                self.load_s = time.perf_counter() - t0
                return
            except OSError:
                pass
        # ensure download (first use)
        try:
            _ensure_translate_model(model, progress=progress, cancel=cancel)
            check_cancel(cancel)
            if progress is not None:
                progress("Loading translation model", None)
            self._nmt_tokenizer = AutoTokenizer.from_pretrained(
                str(dest), local_files_only=True, trust_remote_code=False
            )
            self._nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(dest), local_files_only=True, trust_remote_code=False
            )
            check_cancel(cancel)
            self.load_s = time.perf_counter() - t0
            return
        except OperationCancelled:
            raise
        except Exception:  # noqa: BLE001, S110 - fallback to HF cache
            pass
        # last resort: HF cache (internet) — mirrors pre-2026-08-26 behavior
        check_cancel(cancel)
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
        check_cancel(cancel)
        self.load_s = time.perf_counter() - t0

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        inputs = self._nmt_tokenizer(text, return_tensors="pt", truncation=True)
        out = self._nmt_model.generate(**inputs, max_length=256)
        return self._nmt_tokenizer.decode(out[0], skip_special_tokens=True).strip()
