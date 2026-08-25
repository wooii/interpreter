# Transcribe Benchmark (STT)

Record date: **2026-08-24** — 60 STT samples per model (20 en-only, 20 zh-only,
20 zh↔en mixed), 15-model registry, run in the Linux container (8 cores / 8 GB);
decisions from container numbers only. The 2026-08-23 30-sample pass is retained
as reference below.

Status: **container pass re-run done (2026-08-24)** — 60 STT samples per model (20 en-only, 20 zh-only, 20 zh↔en mixed), 15-model registry; decisions from container numbers only. The 2026-08-23 30-sample pass is retained as reference below.

## Task

The harness (`src/interpreter/benchmark.py`) selects tasks with `--task`:

- **Transcribe (STT)** — `--task stt` (default): audio → text. Whisper-family, sherpa-onnx ASR, Moonshine. Scored with WER/CER/RTF/RSS. **Historical baseline: `whispercpp-large-v3-turbo-q5_0`** (whisper.cpp via pywhispercpp — dropped from the product 2026-08-24; see the Phase 1 conclusion).

Directory layout:

```
data/benchmark/
└── transcribe/                    # STT assets
    ├── manifest.json              # STT samples (60: 20 en / 20 zh / 20 mixed, gold refs + blocks)
    ├── models/                    # downloaded STT weights
    ├── samples/                   # STT audio
    └── results/
        ├── <model>.json           # per-model STT results
        └── merged.json            # STT merged table
```

## Policy: the container is the model filter

The benchmark runs **only in the Linux container** (8 cores, **8 GB RAM**, upgraded
2026-08-24 from 4 cores/4 GB), one model per subprocess
(load → benchmark → write results → exit → next model). A model that does not fit
(exit −9/137, OOM) is **excluded by design**: too big to fit is obviously too slow
to matter for the product. No host benchmark pass. With the 8 GB budget, OOM
exclusions are rare — size/license policy exclusions (below) now dominate.

## Model scope policy (2026-08-24)

- **≤1B parameter policy** — models larger than 1 billion parameters (published
  size) are out of benchmark scope. Applied retroactively: faster-whisper
`large-v3` (1.55B) removed from the registry; Qwen3-ASR `1.7B` (removed
   2026-08-23) was already out; never-benchmarked candidates excluded by size:
   Cohere Transcribe (2B), FireRedASR-AED/-2-AED (1.1B), GLM-ASR-Nano (1.5B),
   Kimi-Audio (7B).
- **Ruling on ARK-ASR-0.6B**: published size (0.6B decoder) counts; its
  end-to-end parameter count is ~1.2B (0.6B decoder + 0.6B audio encoder) —
  flagged, not excluded. Same class of model as Qwen3-ASR-0.6B.
- **License policy** (product rule: no CC-BY-NC or stricter in the MIT repo):
  excluded — Canary-1b v1 (CC-BY-NC-4.0). CC-BY-4.0 models (Parakeet, Canary v2/flash, Nemotron,
  Moonshine) are fine with attribution. VoxCPM is TTS, not ASR — out of scope.
- **No auth-gated downloads (2026-08-24)** — models that require
  authentication to download are excluded (the container has no credentials).
  Everything benchmarked downloads anonymously.
- **Language scope** (zh↔en product): Canary v2/flash (25 EU langs, no zh),
  omniASR (MMS-derived, CC-BY-NC, low-resource focus), Wren-ASR (no zh) are
  excluded by language/license rather than size.

## Transcribe task (STT): audio → text

### Corpus & metrics

- **en-only** (20): LibriSpeech test-clean clips (speaker 61-70968, utterances
  0000–0008 and 0010–0019) + the project's 60 s test recording (`en_10`) — word-level **WER**.
- **zh-only** (20): AISHELL-1 utterances BAC009S0002W0123–W0142 (official HF
  repo, Apache 2.0) — char-level **CER** (zh has no word boundaries; the
  reference is word-segmented while model output is not).
- **en+zh mixed** (20): concatenations of the above with 0.4 s silence gaps, in
  both orders incl. two-switch mixes — column value = mean of the per-sample
  block scores (en block WER + zh block CER), i.e. the model's
  simultaneous-multilingual capability.
- **RTF** = decode wall time ÷ audio duration (< 1.0 = faster than real time). The
  average is inflated by the short 3–5 s clips (fixed overhead); `en_10` (60 s) is the
  long-audio datapoint. Container RTF is relative (the M4 host is faster; ~13–18× on
  the old 4-core profile) — the ranking is what matters.
- **Peak RSS MB** per model process — the memory budget check.
- Whole-text WER/CER on zh-containing samples is inflated by reference word-spacing —
  block scores are authoritative.

### Metrics explained

| Metric | Definition |
|---|---|
| **WER** | Word Error Rate via jiwer — (insertions + substitutions + deletions) ÷ reference word count. Used for English blocks (en has word boundaries). Punctuation tokens count as words (the harness lowercases but does not strip punctuation) — a model that emits more punctuation (e.g. Parakeet v3's Granary-trained punctuation) accrues insertion errors. |
| **CER** | Character Error Rate via jiwer — same formula at character level. Used for zh blocks: zh has no word boundaries and the reference is word-segmented while model output is not; unspaced/glued zh hypothesis words are attributed to the nearest zh block via word alignment. |
| **en+zh mixed** | Mean of the per-sample block scores on mixed samples (en block WER + zh block CER averaged per sample). >1.0 is possible — zh char CER has no upper bound (insertion-heavy hypotheses). |
| **RTF** | Real-Time Factor = decode wall time ÷ audio duration. < 1.0 means the model processes faster than real time. Container numbers are relative (the M4 host is faster; ~13–18× on the old 4-core profile); the *ranking* is what matters. |
| **RSS MB** | Peak **Resident Set Size** of the model's benchmark process, in MB — the maximum physical RAM the process held during the run, measured per model via `getrusage(ru_maxrss)` (`benchmark.py:197`; reported in KB on Linux, bytes on macOS, normalized to MB). It is the memory-budget check: one model per subprocess, so each measurement is isolated and memory is fully released between models; a model that exceeds the budget is OOM-killed (exit −9/137) and recorded as `excluded: OOM`. Caveats: shared-library pages are counted per process (can double-count across processes). |

### How to run

```bash
uv run python -m interpreter.benchmark --list                        # models + samples
uv run python -m interpreter.benchmark                               # all STT models, one subprocess each
uv run python -m interpreter.benchmark parakeet-tdt-0.6b-v2 sensevoice
uv run python -m interpreter.benchmark --samples sample_a1
```

Benchmarking runs only in the container (policy) — there, prefix every command with
`UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync` (plain `uv run` targets the
host venv; AGENTS.md). The registry is now the 10 product-relevant models —
whisper.cpp and Moonshine adapters were removed 2026-08-24 (re-run those from commit
`59e91d7`).

Per-model STT JSONs land in `data/benchmark/transcribe/results/<model>.json` (gitignored
under `/data/`); the parent prints the merged table and writes
`data/benchmark/transcribe/results/merged.json`.

### Merged results — 60-sample pass (2026-08-24, 20 en / 20 zh / 20 mixed)

| model | en-only WER | zh-only CER | en+zh mixed | RTF avg | RSS MB |
|---|---|---|---|---|---|
| **SenseVoiceSmall** | 0.256 | **0.008** | **0.182** | 0.010 | 564 |
| **Parakeet TDT 0.6B v2** (sherpa int8) | **0.148** | — | — | 0.027 | 1331 |
| Parakeet TDT 0.6B v3 (sherpa int8) | 0.165 | — | — | 0.028 | 1759 |
| Qwen3-ASR `0.6B` (sherpa int8) | 0.238 | 0.107 | 0.384 | 0.099 | 3387 |
| ARK-ASR-0.6B | 0.202 | 0.099 | 0.362 | 0.749 | 6848 |
| faster-whisper `small` (int8) | 0.187 | 0.262 | 0.507 | 0.282 | 1063 |
| faster-whisper `medium` (int8) | 0.221 | 0.199 | 0.502 | 0.777 | 2193 |
| whisper-small (openai, fp32) | 0.196 | 0.290 | 0.809 | 0.630 | 1750 |
| whisper.cpp `small` | 0.208 | 0.967 | 0.565 | 0.540 | 832 |
| whisper.cpp `base` | 0.230 | 0.975 | 0.607 | 0.154 | 365 |
| whisper.cpp `medium.en` | 0.209 | — | — | 2.354 | 1861 |
| whisper.cpp `large-v3-turbo-q5_0` (**baseline**) | 0.191 | 0.644 | 0.526 | 2.587 | 867 |
| Moonshine v2 streaming-medium | 0.204 | — | — | 0.096 | 1251 |
| Fun-ASR-Nano-2512 (sherpa int8) | 0.222 | 0.079 | 0.453 | 0.097 | 2344 |
| Dolphin small (sherpa int8) | 1.000 | 0.016 | 0.754 | 0.026 | 1462 |

Models in this pass (15): whispercpp `large-v3-turbo-q5_0` (baseline), `base`,
`small`, `medium.en`, whisper-small, faster-whisper `small`/`medium`, Moonshine v2
streaming-medium, Parakeet TDT 0.6B v2 **and v3**, Dolphin small, Fun-ASR-Nano-2512,
SenseVoiceSmall, Qwen3-ASR `0.6B` (re-run — fits the 8 GB budget now), ARK-ASR-0.6B.
Excluded by policy (see above): faster-whisper `large-v3` (1.55B), Qwen3-ASR `1.7B`,
Cohere Transcribe (2B), FireRedASR (1.1B), GLM-ASR-Nano (1.5B), Canary-1b (CC-BY-NC).
Not runnable with the pinned
sherpa-onnx 1.13.x (factories unreleased): X-ASR streaming zh-en, Nemotron-3.5
streaming 0.6B — logged as candidates for a future round.

### Merged results — 30-sample pass (2026-08-23, reference)

| model | en-only WER | zh-only CER | en+zh mixed | RTF avg | RSS MB |
|---|---|---|---|---|---|
| **SenseVoiceSmall** | 0.156 | 0.011 | **0.207** | 0.010 | 569 |
| Parakeet TDT 0.6B v2 (sherpa int8) | **0.133** | — | — | 0.027 | 1369 |
| faster-whisper `small` (int8) | 0.143 | 0.305 | 0.688 | 0.386 | 1064 |
| whisper-small (openai, fp32) | 0.154 | 0.352 | 1.015 | 0.715 | 1746 |
| whisper.cpp `small` | 0.161 | 1.000 | 0.597 | 0.664 | 832 |
| whisper.cpp `base` | 0.202 | 1.000 | 0.598 | 0.189 | 368 |
| whisper.cpp `medium.en` | 0.185 | — | — | 2.835 | 1858 |
| faster-whisper `medium` (int8) | 0.195 | 0.233 | 0.609 | 1.011 | 1767 |
| whisper.cpp `large-v3-turbo-q5_0` (**baseline**) | 0.209 | 0.737 | 0.553 | 3.569 | 864 |
| Moonshine v2 streaming-medium | 0.207 | — | — | 0.100 | 1156 |
| Fun-ASR-Nano-2512 (sherpa int8) | 0.259 | 0.083 | 0.450 | 0.099 | 2242 |
| Dolphin small (sherpa int8) | 1.000 | **0.023** | 0.803 | 0.027 | 1459 |
| Qwen3-ASR `0.6B` (sherpa int8) | — | — | — | — | excluded: OOM (4 GB budget) |

Notes: mixed > 1.0 is possible (zh char CER has no upper bound — insertion-heavy
hypotheses). whisper.cpp base/small show zh CER 1.000 — same English-misdetection
behavior as the baseline (`large-v3-turbo-q5_0`). faster-whisper's CTranslate2 runtime detects zh far better
(0.305/0.233) than whisper.cpp with the same checkpoints. faster-whisper `large-v3`
(1.55B) and Qwen3-ASR `1.7B` were removed from this reference table by the ≤1B
policy (2026-08-24).

### Findings

- **en-only (listen mode): Parakeet TDT 0.6B v2** — best en WER (0.148 on the
  20-sample set) at 1/95th of the baseline's (`large-v3-turbo-q5_0`) compute, and it's a
  transducer (streaming-capable weights; probe the sherpa online recognizer in Phase 2).
  v3 (0.165) is close but slightly worse on English — the 25-EU-language checkpoint
  trades a little en accuracy for breadth the product doesn't need. Two causes, from
  per-sample inspection: (a) most of the gap is punctuation — v3 (Granary-trained,
  punctuation-preserving) inserts commas/final periods that the word-level WER counts
  as errors; 13 of 20 samples are word-identical; (b) the one real word-level
  regression is the 60 s `en_10` (0.127 → 0.239: dropped phrase, "two time"/"thirty"
  errors) — v3 is weaker on long-form English than the en-only v2 (human-transcribed
  NeMo ASR Set vs largely pseudo-labeled Granary). The whisper family
  adds nothing new: faster-whisper `small` (0.187) and openai whisper-small (0.196) are
  close to Parakeet but slower (0.282 / 0.630 RTF). The baseline `large-v3-turbo-q5_0`
  (0.191) remains the accuracy-max option; Moonshine v2 is the streaming-latency
  wildcard (real streaming API, keeps real time easily).
- **The whisper family misdetects short Chinese clips as English** and outputs a rough
  translation instead of a transcript — confirmed across whisper.cpp base/small (zh
  CER 0.967/0.975) and the baseline `large-v3-turbo-q5_0` (0.644): e.g. 眼中钉 → *"the government's eye-tune"*,
  显示出了极强的威力 → *"It shows the very powerful power"*. Language misdetection
  triggers the decoder's latent translation; it is not explicit translate mode.
  faster-whisper's CTranslate2 runtime is markedly better at zh detection (0.262 small /
  0.199 medium).
- **multilingual (dictate mode): SenseVoiceSmall** — the only model that handles true
  zh↔en code-switching: zh CER 0.008, mixed error 0.182 (en WER + zh CER combined),
  and it is the fastest and lightest model in the benchmark.
- **The baseline `large-v3-turbo-q5_0` cannot code-switch**: at every language switch it commits to
  one language and silently drops the other block (zh CER 0.644 — mostly dropped zh,
  en WER 0.191 — mostly the en parts kept). Excellent per-language, wrong tool for
  mixed dictation.
- **Dolphin is a zh specialist**: zh CER 0.016 (best after SenseVoice) but en-deaf
  (en WER 1.000) — only relevant for pure-zh configs, not the bilingual product.
- **Fun-ASR-Nano-2512**: second-best zh (0.079) but drops one language in half the
  mixed samples (0.453) and keeps the ~60 s `max_total_len=512` cap.
- **2026-08-24 additions:** Qwen3-ASR `0.6B` ran for real on the 8 GB budget — second-best
  zh (0.107) and second-best mixed (0.384) after SenseVoice, at RTF 0.099; a credible
  SenseVoice alternative for multilingual configs. ARK-ASR-0.6B (Apache-2.0, zh/en)
  lands at 0.202/0.099/0.362 — third-best mixed — but peaks at 6848 MB RSS (~86% of the
  8 GB budget) at RTF 0.749; its paper's SOTA claim doesn't translate to a win on this
  corpus and its memory footprint rules it out as a co-resident model. Parakeet v3
  confirmed as an en-only non-upgrade over v2 on English.
- **Dominated / dropped:** whisper.cpp `medium.en` (worse en WER than Parakeet and
  slower than it and Moonshine); Qwen3-ASR `1.7B` (4.7 GB weights, broken
  transformers path — now also out by the ≤1B policy); faster-whisper `large-v3` out
  by the ≤1B policy.
- **Multilingual verdict** is from synthesized mixed samples; self-recorded dictation
  remains a future validation, not a blocker.

### Recommendation / Phase 1 conclusion (2026-08-24, from the 60-sample pass)

Chosen model set from the merged table, mapped to the config matrix:

| Config cell | Model | Rationale |
|---|---|---|
| listen / en-only | **Parakeet TDT 0.6B v2** (sherpa int8) | best en WER (0.148) at RTF 0.027 — ~95× less compute than the baseline; transducer = streaming-capable weights (probe sherpa's online recognizer in Phase 2 — the future streaming path, see below) |
| dictate / multilingual | **SenseVoiceSmall** (sherpa int8) | only candidate that truly code-switches (zh CER 0.008, mixed 0.182); fastest + lightest model in the benchmark (RTF 0.010, 564 MB); Qwen3-ASR `0.6B` (mixed 0.384, RTF 0.099) is the closest alternative |
| listen / streaming live | ~~Moonshine v2 streaming-medium~~ **dropped 2026-08-24** | was the only one with a real streaming API (partials in ~1 s), but weaker on accuracy/speed than Parakeet; the streaming path is now a sherpa-onnx `OnlineRecognizer` probe of Parakeet (Phase 2) |

**Baseline verdict — superseded:** the product default `whispercpp-large-v3-turbo-q5_0`
loses its role. It is dominated on every axis measured: worse en WER than Parakeet v2
(0.191 vs 0.148) at ~95× the compute, worst-in-class zh (0.644 — whisper English
misdetection), mixed 0.526 vs SenseVoice's 0.182, and the highest RTF in the table
(2.587). No config cell keeps it; it survives only as the historical measurement
anchor. Phase 1 conclusion: **listen → Parakeet v2, dictate → SenseVoiceSmall**.
Implemented in `transcribe.py` on 2026-08-24 (sherpa STT backends; whisper.cpp and
Moonshine dropped the same day); per-mode defaults in
`profiles/` land with the Phase 2
config plumbing. **Interim product default (2026-08-24): `sensevoice`** —
until Phase 2 wires per-mode config, the single STT default is the dictate/multilingual
winner (only true code-switcher, fastest + lightest model in the benchmark; en-only
accuracy takes a small hit vs Parakeet in the interim), with parakeet-tdt-0.6b-v2
selectable via `stt_model="parakeet-tdt-0.6b-v2"`.

**whisper.cpp dropped from the product completely (2026-08-24)** — the results above
stay as the historical record, but whisper.cpp is no longer in `transcribe.py`, the
benchmark harness, or the dependencies (`pywhispercpp` removed). Why:
- **Dominated on every axis** — the baseline `large-v3-turbo-q5_0` is worse en than
  Parakeet (0.191 vs 0.148) at ~95× the compute, worst-class zh (0.644, from whisper's
  English misdetection), no code-switching (mixed 0.526 vs SenseVoice 0.182), and the
  highest RTF (2.587). No config cell keeps it, so it has no product role.
- **The sherpa pair covers everything** — SenseVoiceSmall (dictate/multilingual) +
  Parakeet (listen/en-only) fill all STT roles; whisper.cpp's last edge, per-segment
  probability coloring, was superseded by Parakeet's real per-word coloring (from
  token log-probs), leaving no remaining differentiator.
- **Operational weight** — `pywhispercpp` is a compiled binding with multi-GB ggml
  weight downloads, and its context is not thread-safe (a Phase 2 parallel-worker
  concern); dropping it also let the legacy `model_comparison.py` harness go.
- No separate snapshot commit was made — the pre-drop harness survives at commit
  `59e91d7` (`feat: add benchmark.py`): that tree has `benchmark.py` with the
  whisper.cpp + Moonshine adapters and `--stream` hooks, plus `whispercpp.py`.
  Re-run whisper.cpp benchmarking from there (deps: `pywhispercpp`).

**Moonshine dropped from the product completely (2026-08-24)** — the results above
(and the M4 spot-run) stay as the historical record, but Moonshine is no longer in
`transcribe.py`, the benchmark harness, or the dependencies (`moonshine-voice` removed).
Why:
- **Its only edge couldn't be exercised** — Moonshine's value is live streaming
  partials (~1 s), but the product pipeline is segment-based (VAD → whole segment →
  transcribe), so it was wired up with `transcribe_without_streaming` — the *same*
  interaction as the sherpa backends, where it was never meant to win.
- **Weakest of the candidates on the axis that mattered** — live A/B and the
  benchmark agree: worse en WER (0.204 vs Parakeet 0.148) and slower per segment
  than Parakeet; no product role as a segment-based model.
- **The future streaming path doesn't need it** — Parakeet is a transducer
  (streaming-capable weights); a sherpa-onnx `OnlineRecognizer` probe of Parakeet
  (Phase 2) would deliver live partials *with* the benchmark's best accuracy, which
  Moonshine can't match. The harness's `--stream` hooks stay dormant for that.
- Moonshine was never committed as its own file and the `moonshine-voice` dep was
  never in `pyproject.toml` — its adapter + `--stream` hooks survive in
  `benchmark.py` at `59e91d7`, but the dependency itself is not recoverable from
  git history (would need re-adding to re-run).

**Dropped:** the entire whisper family (whisper.cpp base/small/medium.en, faster-whisper
small/medium, openai whisper-small) — every one dominated (slower than Parakeet,
worse on zh: whisper.cpp zh CER ~0.97 from English misdetection); the baseline
`large-v3-turbo-q5_0` loses its default role (0.191 en WER at ~95× Parakeet's compute);
Dolphin (en-deaf; keep only for a hypothetical pure-zh config); Fun-ASR-Nano-2512
(60 s cap, drops one language in mixed clips); ARK-ASR-0.6B (6848 MB RSS ≈ 86% of the
8 GB budget, no accuracy win — the container filter settles what the ≤1B ruling left
open); Parakeet v3 (en non-upgrade over v2). Faster-whisper `large-v3`, Qwen3-ASR
`1.7B`, Cohere Transcribe, FireRedASR, GLM-ASR-Nano, Canary-1b: out by the
≤1B / license / auth policy (see "Model scope policy").

**Phase 2 implications:** one `SherpaStt` backend, config-driven model choice (both
winners run on sherpa-onnx int8); **Moonshine dropped 2026-08-24** (see above) — the
streaming path becomes a sherpa-onnx `OnlineRecognizer` probe of Parakeet (transducer =
streaming-capable weights); the harness's `--stream` hooks stay dormant for it. The
per-word confidence coloring: **done for the sherpa transducer (parakeet)** — per-token
`ys_log_probs` grouped into word probs in `sherpa_stt.py` (`_word_probs_from_result`,
2026-08-24); **SenseVoice exposes no per-token scores in sherpa-onnx 1.13.0** — its
output stays uniform until a sherpa release surfaces them; `profiles/` defaults:
listen → parakeet, dictate → sensevoice.

### Reference: M4 host spot-run (context only, not the decision basis)

Same harness, run once on the Mac (2026-08-23) before the container-only policy was
set. Confirms the ranking and that every container-passing model is comfortably
real-time on the M4. Sherpa models did not run there (macOS wheel fails to load
`libonnxruntime.1.24.4.dylib` — fixed on the product side 2026-08-24 by
`_ensure_onnxruntime_dylib` in `sherpa_stt.py`).

| model | WER avg | RTF avg |
|---|---|---|
| whisper.cpp `large-v3-turbo-q5_0` (**baseline**) | 0.023 | 0.075 |
| whisper.cpp `medium.en` | 0.197 | 0.095 |
| Moonshine v2 streaming-medium | 0.238 | 0.064 |
