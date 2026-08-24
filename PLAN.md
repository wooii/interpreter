# Interpreter — Modernization Plan

Status: **Phase 1 complete (2026-08-24) — Phase 2 in progress.**

## Current state

Facts that hold *now*. Issues/gotchas live in the phase item that resolves them.

**Product decisions** (stand until superseded):

- **Local models only** — no cloud STT/translation, no API keys, ever.
- **Minimal feature scope (2026-08-24)** — models are internal per-task picks, never user-facing. User config = `mode` (`listen`/`dictate`) + `translate` bool + `language` scope (`en-only`/`mixed`) only.
- **≤1B parameter policy; no CC-BY-NC-or-stricter weights; no auth-gated downloads.**
- **Container is the benchmark environment** (Linux, 8 GB/8 cores; M4 host numbers are reference only). Entry point stays `transcribe.py` until the Phase 2 CLI.

**Model verdicts** (2026-08-24, 60-sample container pass; **full record: `docs/benchmark.md`**):

- listen / en-only → **Parakeet TDT 0.6B v2**; dictate / multilingual → **SenseVoiceSmall** (only true zh↔en code-switcher); translate → **opus-mt-en-zh** (only backend; LLM quality mode dropped).
- whisper.cpp / Moonshine **dropped from the product**; the whispercpp baseline is superseded (pre-drop harness at commit `59e91d7`).
- **No true streaming for Parakeet TDT v2** — its "real-time" is simulated streaming (growing-buffer re-decode, sherpa-onnx #2918/#3573); true streaming needs the separate Parakeet Unified model (Phase 3).

**Architecture** (`src/interpreter/transcribe.py`):

- `RealTimeTranscribe`: Silero VAD → ring buffer → 3 worker threads (VAD → STT → translate); `max_segment_duration` caps segments. `plain_output=True` (dictate CLI) prints copy-paste-friendly text — no timestamps/durations/ANSI colors, plus a final clean `Transcript:` block after Ctrl+C; listen keeps the styled per-segment view. `evaluate()` reports WER only for whitespace-segmented (en) text; CJK/mixed reports CER only (WER is meaningless on unsegmented zh — whole-text WER on zh inflates, same reason docs/benchmark.md scores zh with CER).
- STT = sherpa-onnx int8 only: `sensevoice` (default), `parakeet-tdt-0.6b-v2` (en-only/listen). Parakeet colors per word; SenseVoice output stays uncolored (no per-token scores in sherpa-onnx 1.13.0 — documented limitation). **SenseVoice emits English in ALL CAPS (model artifact)** — `SpeechToText.extract_text` sentence-cases it for sensevoice (live lines, final transcript, and evaluate reference all go through it); parakeet already outputs natural case and is untouched.
- Translate = `opus-mt-en-zh` (deterministic, ~1.2 s/sentence, en→zh).
- Weights shared with the benchmark under `data/benchmark/transcribe/models/` (anonymous HF download). App form undecided — **Phase 4** decides.

**Environment**:

- Container runs: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync <cmd>`; never plain `uv sync`/`uv run`. Dep pins: sherpa-onnx 1.13.0 + onnxruntime 1.24.4; **torch + torchaudio from the `pytorch-cpu` index on Linux** (PyPI wheels are CUDA and fail to dlopen on the container's CPU torch). **`sentencepiece` is a direct dep** (opus-mt's Marian tokenizer needs it — silently dropped with funasr's removal, see Phase 2). **matplotlib** is a direct dep again (audio.py plotting; also transitive via noisereduce). sherpa needs `libonnxruntime.so` on the loader path (self-healed in the harness). Container audio: libportaudio2 + pulseaudio for headless `sounddevice`; macOS: `_ensure_onnxruntime_dylib` self-heals the missing bundled onnxruntime.
- Bench assets under `data/benchmark/` (gitignored via `/data/`), grouped by task; `docs/benchmark.md` is the run record — **moves to `_archive/` when Phase 2 completes**.

## Phases

### Phase 1 — Model benchmark (model selection)

Status: **COMPLETE (2026-08-24)** — 60 samples / 15 models; listen → Parakeet TDT 0.6B v2, dictate → SenseVoiceSmall, translate → opus-mt-en-zh; whispercpp baseline superseded. Record: `docs/benchmark.md`.

### Phase 2 — Architecture restructure

Status: **IN PROGRESS** — restructure done 2026-08-24 in a lean 3-file shape (`transcribe.py` + `translate.py` + `__main__.py`, plus `audio.py` and the `benchmark.py` harness); threading-backlog investigation + `docs/benchmark.md` archive still open.

- [x] **Feature scope**: `listen` = en-only STT (Parakeet v2) + translate en→cn, translate **ON** by default; `dictate` = mixed STT (SenseVoice) or `--language en-only` (Parakeet v2), default mixed, translate **OFF** by default. No model names exposed (the CLI resolves them internally).
- [x] **Lean layout** (2026-08-24 — an earlier 7-file concern-split was rejected as over-engineering): `transcribe.py` = the original committed module (VAD, `process_audio_segment`, `SpeechToText` dispatch, `RealTimeTranscribe` with its original `stt_model`/`translate_model`/`translate_to` API) with the sherpa backend folded in (`MODEL_SPECS`, `SherpaStt`, `SherpaSegment`, word-probs, download/dylib helpers); `translate.py` = `Translator` (opus-mt-en-zh, only backend); `__main__.py` = thin CLI; `audio.py` = restored committed class API (play/plot_waveform/record/convert_format/plot_mel_spectrogram); `benchmark.py` = the harness.
- [x] **CLI** (`__main__.py`): `uv run python -m interpreter <listen|dictate|benchmark>`; `listen [--no-translate] [--audio-file]`, `dictate [--translate] [--language en-only|mixed] [--audio-file]`; `benchmark` → passthrough to `interpreter.benchmark`; `run()`/`evaluate()` behind both live modes; README updated.
- [x] **audio.py restored** (2026-08-24): original `AudioDataProcessor`/`AudioFileProcessor` API kept; `plot_mel_spectrogram` now implements whisper's log-mel pipeline in numpy (80 Slaney mel bins, power→mel→log10, whisper's clamp+rescale, viridis) — no whisper dep; `record` writes via soundfile (wavio dropped). Matplotlib re-added as a direct dep.
- [x] **Benchmark trim to sherpa-only**: faster-whisper / openai-whisper / funasr / modelscope / ark-asr adapters + their deps dropped; `--stream` hooks removed (no stream-capable model left; streaming probe moved to Phase 3). Keep sacrebleu/sacremoses (translate-task scoring). Canonical 60-sample results re-run for the two product models after the trim.
- [x] **Dep prune gotchas** (2026-08-24 — removing `funasr` transitively dropped deps the product still needs):
  - **`sentencepiece` re-added as a direct dep** — the opus-mt Marian tokenizer needs it; transformers masks the missing import as a misleading "Unrecognized configuration class" error.
  - **`torchaudio` re-added (pytorch-cpu pin)** — silero-vad's hub script imports it (`from silero_vad.utils_vad import ... torchaudio`); not imported by our src, so the prune missed it.
  - **`torch.hub.load` needs `trust_repo=True`** for silero-vad — the default prompts interactively and fails non-interactive (EOFError).
  - **`__main__.py` must call `main()`** — a missing call exits 0 silently (no-op CLI); caught when `benchmark --list` printed nothing.
- [ ] **Threading backlog** (from the port): transcription is single-threaded — one `_transcription_worker` serially drains `q_for_transcription`, and the sherpa recognizer blocks per segment. If STT time > segment duration, the queue backlogs and latency grows. Candidate refinements to evaluate: (a) instrument first — timing logs + queue-depth counters; (b) parallel transcription workers (one `SherpaStt` per worker); (c) `num_threads` to parallelize within a segment; (d) split `process_audio_segment` (noisereduce) into its own worker stage.
- [ ] **Done**: move `docs/benchmark.md` → `_archive/` as the historical run record; PLAN.md keeps verdicts + links.

### Phase 3 — Future features (Python, product-complete)

Status: **TODO**.

- **Speaker identification**: Parakeet can't do it (ASR only) — add a small WeSpeaker embedding model (CC-BY-4.0; `wespeaker_voxceleb_resnet34.onnx` for en, `wespeaker_zh_cnceleb_resnet34.onnx` for zh) via sherpa-onnx `SpeakerEmbeddingExtractor`/`SpeakerEmbeddingManager` (enrollment-based ID per segment); diarization (pyannote-3.0 segmentation + clustering, sherpa-onnx ≥1.10.28) as the unknown-speaker fallback. Same sherpa-onnx runtime — no Parakeet change.
- **Translate only other people's speech**: ties into speaker ID — suppress translation for enrolled self.
- **Streaming live partials** (deferred from Phase 2, 2026-08-24): benchmark **Parakeet Unified en 0.6B** (true `OnlineRecognizer` streaming, ~240 ms chunks, sherpa-onnx support added 2026) vs TDT v2 at low latency — if it wins on accuracy-at-latency, switch listen to it. TDT v2's "real-time" is simulated streaming only (#2918/#3573).

### Phase 4 — App form

Status: **TODO**.

- Decide app form after the Python product works (Phase 3 done). Recommended default: local FastAPI web UI over the same package; native macOS/iOS later via sherpa-onnx native bindings stays open.
