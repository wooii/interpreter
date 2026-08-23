# Interpreter — Modernization Plan

Status: **Phase 1 (gate) — pending (research done 2026-08-22; benchmark next)**.

## Current state

Facts that hold *now*. Issues/gotchas are not listed here — they live in the phase item that resolves them (below).

**Product decisions** (stand until superseded):

- **Local models only (2026-08-21)** — no cloud STT/translation, no API keys, ever. Online paths are a commodity (OpenAI itself ships `gpt-realtime-whisper` / `gpt-transcribe` / `gpt-realtime-translate`), so they add no differentiation and carry per-minute cost. The product's edge is the local bilingual zh↔en pair: privacy, offline, zero cost. Removed: deprecated OpenAI script, `openai_api_pricing.yaml`, `openai` dependency.
- **Language scope is a config, not a mode (2026-08-22).** Both `listen` and `dictate` support `en-only` and `multilingual` (zh/en mixed) variants; STT and translation are independently configurable per mode. **Translation must be fast** — target sub-second per segment: fast dedicated NMT by default, optional LLM quality mode.
- **No CC-BY-NC (or stricter) model weights in the product** (MIT repo): excludes NLLB-200 and Moonshine's non-English models; opus-mt / M2M100 / MADLAD-400 licenses are verified in Phase 1.
- **Entry point stays `transcribe.py` (2026-08-22)** — no rename for now; a `__main__.py` + argparse CLI (subcommands `listen|dictate|benchmark`) is deferred to the Phase 2 CLI item. Current `run()` → `evaluate()` in one command is known-temporary.

**Current architecture** (`src/interpreter/transcribe.py`):

- `RealTimeTranscribe`: Silero VAD (`torch.hub` → `snakers4/silero-vad`) feeds a ring buffer; three daemon threads consume queues for VAD → transcription → translation; `max_segment_duration` caps segment length.
- STT is `WhisperCppModel` (`src/interpreter/whispercpp.py`), a `pywhispercpp.Model` subclass adding per-segment average token probability (`MySegment.probability`); output is printed with per-word color coding from that probability.
- **Benchmark baseline (current defaults):** STT = pywhispercpp `large-v3-turbo-q5_0` (whisper.cpp); translation = Ollama `qwen3.5:0.8b`, `think=False`, `ollama.chat`. Everything in Phase 1 is measured against this pair.
- `RealTimeTranscribe.evaluate()` computes WER/CER (jiwer) vs a full-file reference; only meaningful when `audio_file_path` is set.
- Final product form (Python pkg + local web UI vs native macOS/iOS app) is **undecided** — Phase 3 decides. Recommended default: local FastAPI web UI over the same package; native macOS/iOS later via sherpa-onnx / Moonshine native bindings stays open.

## Phases

### Done (prior work)

- Real-time, continuous audio recording and streaming (mic → VAD → STT pipeline).
- Local transcription integration and initial model comparison (`src/interpreter/model_comparison.py`): whisper.cpp engine in production, Faster-Whisper benchmarked, OpenAI Whisper API path removed (local-only decision — see Current state).

### Phase 1 — Model research & benchmark (GATE)

Status: **PENDING** — research done (2026-08-22); candidate set below is the frozen shortlist.

Decide the best local model combination on real audio, **beating the current baseline** (pywhispercpp `large-v3-turbo-q5_0` + Ollama `qwen3.5:0.8b`) on accuracy (WER/CER incl. code-switching), latency (STT RTF + translation end-to-end), and resource use (target: Mac M4 16 GB).

- Build the benchmark harness (extend `src/interpreter/model_comparison.py`):
  - Mode A samples: English meeting recordings (en-only listen scenario).
  - Mode B samples: self-recorded zh↔en dictation (multilingual dictate scenario).
  - Metrics: WER/CER (jiwer) + per-segment RTF/latency; code-switching WER at language-switch points; translation end-to-end latency + quality (BLEU + human spot-check).
  - Runs in the container on the sample files; latency-sensitive numbers flagged for a host re-run (M4 16 GB) — container RTF is not the product's real latency.
- STT candidates (all open-source, local):
  - **en-only tier** (for `en-only` mode configs): Moonshine v2 streaming-medium (EN, MIT; listen+dictate); whisper.cpp `medium.en` (mature fallback); Parakeet TDT 0.6B v2 via sherpa-onnx int8 (optional; offline-only).
  - **multilingual tier**: Qwen3-ASR `1.7B` + `0.6B` (Apache-2.0; streaming, code-switching); Fun-ASR-Nano-2512 (funasr / sherpa-onnx int8; streaming); SenseVoiceSmall (funasr or GGUF/llama.cpp; non-streaming, fastest zh/en); Dolphin `small.cn.streaming` (0.4B, Apache-2.0); whisper.cpp `large-v3-turbo-q5_0` (**baseline**); faster-whisper `large-v3` (accuracy reference).
  - Excluded (document in `docs/model-selection.md`): Parakeet for multilingual (no zh), Moonshine for multilingual (non-EN models non-commercial), Kitten/Zion (unverifiable / too heavy), FireRedASR2-LLM (needs ≥32 GB VRAM).
- Translation candidates:
  - **fast tier** (default; target <1 s/segment): opus-mt-en-zh (Helsinki-NLP), M2M100-418M, MADLAD-400 (418M or 3B) — verify licenses (Apache-2.0/MIT expected; no CC-BY-NC).
  - **quality tier** (optional, config): qwen3.5 `0.8b` (**baseline**)/`2b`/`4b` via Ollama `think=False`; qwen3 `8b` as quality ceiling.
- Deps (add via `UV_PROJECT_ENVIRONMENT=.venv-container uv add --no-sync`, flag host sync): funasr or sherpa-onnx (Py3.14 wheel risk — GGUF/llama.cpp fallback for SenseVoice / Fun-ASR-Nano), ctranslate2 + transformers for NMT, mlx ports on host.
- Output: `docs/model-selection.md` (decision + numbers vs baseline) + set per-mode defaults in `profiles/` for each cell of the config matrix (listen/dictate × en-only/multilingual × translation on/off).
- **Notes merged into this phase** (resolved when the phase is checked off):
  - whisper.cpp weights (e.g. `large-v3-turbo-q5_0`) auto-download via pywhispercpp on first run (several GB) — verify the download during the benchmark run.
  - Candidate risks to probe in the benchmark: Qwen3-ASR has known bugs — infinite token repetition (Qwen3-ASR#129) and pseudo-streaming boundary repetition — and no streaming timestamps; Fun-ASR-Nano-2512 has no reliable checkpoint-native char timestamps (issue #106).
  - Numbers behind the two-tier translation design (research 2026-08-22): dedicated NMT (opus-mt ~300 MB, M2M100-418M, MADLAD-400) does ~50–300 ms/sentence vs ~1–5 s for Ollama small LLMs; NLLB-200 3.3B quality ≈ 4 BLEU below Qwen3-32B local (NLLB excluded anyway: CC-BY-NC).

### Phase 2 — Architecture restructure

Status: **TODO**.

- Layout: `interpreter/{audio,vad,stt,translate,pipeline,app,eval}/`.
- Interfaces: `SttProvider.transcribe(array) -> segments`, `TranslatorProvider.translate(text) -> str`, provider registry selected via config (STT: en-only vs multilingual; translation: fast NMT vs LLM quality).
- Backends: `WhisperCppStt`, `SenseVoiceStt`, `Qwen3AsrStt`, `FunAsrNanoStt`, `OllamaTranslator`, `NmtTranslator`. Keep per-word colored probability output.
- CLI: `__main__.py` + argparse (subcommands `listen|dictate|benchmark`, entry `uv run python -m interpreter`) — the deferred decision from Current state; port current `run()`/`evaluate()` behind `listen`.
- Port `RealTimeTranscribe` into `pipeline/` (VAD + ring buffer + worker threads) unchanged in behavior; CLI gains `--mode listen|dictate` and language-scope config.
- **Notes merged into this phase** (resolved when the phase is checked off):
  - Threading finding: threads run concurrently (total ≈ max of per-thread times, not the sum). `RealTimeTranscribe` already runs 3 worker threads (VAD → STT → translate), but **transcription is single-threaded**: one `_transcription_worker` serially drains `q_for_transcription`, and whisper.cpp blocks per segment. If STT time > segment duration (e.g. `large-v3-turbo-q5_0` on 10 s segments), the queue backlogs and latency grows — the commented-out timing prints in `transcribe.py` (lines ~62/237/241) were probing this. Candidate refinements to evaluate during the port: (a) instrument first — re-enable timing logs + queue-depth counters to measure the backlog; (b) parallel transcription workers (one `WhisperCppModel` per worker — a pywhispercpp ctx is not thread-safe; each `large-v3-turbo-q5_0` instance costs several GB RAM); (c) `n_processors` (whisper_full_parallel, already in `WhisperCppModel.transcribe` at `whispercpp.py:118`) to parallelize within a segment; (d) split `process_audio_segment` (noisereduce) into its own worker stage.

### Phase 3 — App form

Status: **TODO**.

- Decide app form.

### Phase 4 — Future features

Status: **TODO**.

- Speaker identification: display who is talking.
- Choose to translate only other people's speech.
