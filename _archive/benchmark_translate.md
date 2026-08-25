# Translate Benchmark (en → zh)

Record date: **2026-08-24** — 100-sentence corpus (wmt19 `newsdev2019`), run in
the Linux container (8 cores / 8 GB). The 2026-08-23 initial pass is retained as
reference below. Shared harness policies (container filter, ≤1B rule, license /
auth rules) and the STT record live in `benchmark_transcribe.md`; speaker ID in
`benchmark_speaker.md`.

## Translate task (en → zh): text → text

### Corpus & metrics

- **100 sentences** from the wmt19 zh-en validation split (`newsdev2019`) in
  `data/benchmark/translate/manifest.json`.
- **BLEU(zh)** via sacrebleu with `tokenize="zh"`.
- **ms/sentence** — decode wall time averaged over the corpus (LLM RTT dominated by
  ollama server round-trips, not model compute).
- **Peak RSS MB** per model process (the 8 GB budget check; see "Metrics explained"
  — for ollama models this measures the python client; the ollama *server* RSS is
  recorded separately).

### How to run

```bash
uv run python -m interpreter.benchmark --task translate --list
uv run python -m interpreter.benchmark --task translate              # all translate models
uv run python -m interpreter.benchmark --task translate opus-mt-en-zh
```

Same container rule as the STT task: prefix with
`UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync` in the container (AGENTS.md).

Per-model JSONs land in `data/benchmark/translate/results/<model>.json`; the parent
prints the merged table and writes `data/benchmark/translate/results/merged.json`.

### Merged results (2026-08-24 pass)

| model | BLEU(zh) | ms/sent | RSS MB |
|---|---|---|---|
| **opus-mt-en-zh** (Helsinki-NLP) | **33.59** | 1152 | 1172 |
| m2m100-418m (facebook) | 31.03 | 5233 | 4658 |
| **ollama-qwen3.5-0.8b** (translate baseline) | 24.16 | 831 | 72* |

*RSS column = the python benchmark client; the ollama *server* is a separate daemon
(≈20–40 MB idle with the model unloaded after the run). Note: the 2026-08-23 pass
measured opus-mt at 732 ms/sentence on the 4-core container — today's 1152 ms is the
8-core pass; the ranking (opus-mt > m2m100 on quality+speed) is unchanged. The 8-core
m2m100 run is 3× faster than the 4-core one (5233 vs 15963 ms/sentence).

Notes: only the transformers NMT pair ran on 2026-08-23 — the ollama baseline needs an
ollama server, which the container lacked; ollama was installed in the container on
2026-08-24. MADLAD-400 was dropped from the fast-NMT tier: gated on HF and CC-BY-NC
licensed. **Qwen3-MT-0.6B excluded** (2026-08-24): gated on HF — model download
requires authentication, and there is no ModelScope mirror; the product policy
excludes auth-gated model downloads entirely (PLAN.md). Verdict: **opus-mt-en-zh is the
product's only translation backend** (best BLEU at ~1.2 s/sentence, 1172 MB); qwen3.5-0.8b —
the former quality-mode LLM baseline — scores 9.4 BLEU below opus-mt at similar
per-sentence cost and was **dropped 2026-08-24** (live dictation A/B: hallucinated
content + meaning reversal; see PLAN.md). The ollama adapter and `ollama` dependency
were removed from the product and the benchmark harness. Early research (2026-08-22)
motivated the fast-NMT default: dedicated NMT ≈ 50–300 ms/sentence vs ~1–5 s for Ollama
small LLMs; NLLB-200 3.3B quality ≈ 4 BLEU below Qwen3-32B local (NLLB excluded anyway:
CC-BY-NC).

### Merged results — initial pass (2026-08-23, reference)

| model | BLEU(zh) | ms/sent | RSS MB |
|---|---|---|---|
| **opus-mt-en-zh** (Helsinki-NLP) | **33.59** | 732 | 991 |
| m2m100-418m (facebook) | 31.03 | 15963 | 3016 |

## Phase 1 conclusion (translate)

**`opus-mt-en-zh` is the product's only translation backend** (2026-08-24) — best
BLEU (33.59 vs the qwen3.5 baseline's 24.16) at ~1.2 s/sentence / 1172 MB. The
former quality-mode LLM baseline (`ollama-qwen3.5-0.8b`) scores 9.4 BLEU below
opus-mt at similar per-sentence cost and was **dropped 2026-08-24** after a live
en→zh dictation A/B showed hallucinated content and a meaning-reversed error
("choose this model instead" → 换成另一种方式) at higher latency (docs in PLAN.md).
The ollama adapter and `ollama` dependency were removed from the product and the
benchmark harness. Early research (2026-08-22) motivated the fast-NMT default:
dedicated NMT ≈ 50–300 ms/sentence vs ~1–5 s for Ollama small LLMs; NLLB-200 3.3B
quality ≈ 4 BLEU below Qwen3-32B local (NLLB excluded anyway: CC-BY-NC).
Implemented in `translate.py` on 2026-08-24.