"""en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend.

Dedicated NMT: deterministic, best BLEU on the benchmark corpus (33.59,
docs/benchmark.md), ~1.2 s/sentence; single pair en->zh. The qwen3.5 LLM
quality mode was dropped 2026-08-24 — live dictation showed hallucinated
content and a meaning-reversed error (PLAN.md, docs/benchmark.md).
"""

from __future__ import annotations

from typing import Any

TRANSLATE_MODEL = "opus-mt-en-zh"


class Translator:
    """en->zh translation — opus-mt-en-zh (Helsinki-NLP seq2seq), the only backend."""

    def __init__(self, model: str = TRANSLATE_MODEL, target_lang: str = "Chinese"):
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
