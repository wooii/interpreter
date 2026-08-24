# Interpreter — Modernization Plan

Status: **Phase 1 + Phase 2 complete (2026-08-24) — Phase 3 open.**

## Current state

Facts that hold *now*. Issues/gotchas live in the phase item that resolves them.

**Product decisions** (stand until superseded):

- **Local models only** — no cloud STT/translation, no API keys, ever.
- **Minimal feature scope (2026-08-24)** — models are internal per-task picks, never user-facing. User config = `mode` (`listen`/`dictate`) + `translate` bool + `language` scope (`en-only`/`mixed`) only.
- **≤1B parameter policy; no CC-BY-NC-or-stricter weights; no auth-gated downloads.**
- **Container is the benchmark environment** (Linux, 8 GB/8 cores; M4 host numbers are reference only). Entry point stays `transcribe.py` until the Phase 2 CLI.

**Startup**: VAD / Translator / STT load **concurrently** (`RealTimeTranscribe._load_models` — independent model stacks, startup waits for the slowest not the sum; worker failures re-raise in the main thread). Silero loads via **direct `torch.jit.load` of the cached `silero_vad.pt`** (skips `torch.hub`'s repo clone/import — also avoids the torchaudio import); transformers uses **`local_files_only=True`** with a download fallback for first use. CLI prints a loading status line.

**Model verdicts** (2026-08-24, 60-sample container pass; **full record: `_archive/benchmark-2026-08-24.md`**):

- listen / en-only → **Parakeet TDT 0.6B v2**; dictate / multilingual → **SenseVoiceSmall** (only true zh↔en code-switcher); translate → **opus-mt-en-zh** (only backend; LLM quality mode dropped).
- whisper.cpp / Moonshine **dropped from the product**; the whispercpp baseline is superseded (pre-drop harness at commit `59e91d7`).
- **No true streaming for Parakeet TDT v2** — its "real-time" is simulated streaming (growing-buffer re-decode, sherpa-onnx #2918/#3573); true streaming needs the separate Parakeet Unified model (Phase 3).

**Architecture** (`src/interpreter/transcribe.py`):

- `RealTimeTranscribe`: Silero VAD → ring buffer → 3 worker threads (VAD → STT → translate); `max_segment_duration` caps segments. **Adaptive stability-window re-decode** (2026-08-24, replaces per-segment transcription): each new segment re-decodes the whole window so the newest utterance gets predecessor context; once the earlier text stops changing (`_stable_prefix` — word-boundary check for en, char-prefix for CJK), the window commits to `committed_chunks` and slides forward (re-decoding the newest segment alone for a colored, consistent baseline); a `max_window_seconds` cap force-commits so cost stays bounded (window is normally tiny). **Live display runs in the terminal alternate screen buffer** (`\x1b[?1049h`, vim-style) with clear-and-reprint per update — cursor-up/erase-line redraw leaked `^[[A` sequences + blank lines into macOS Terminal scrollback. `clean=True` (dictate CLI) renders one flowing plain line with spaces between chunks (zh: space-only separation; en: `.` per chunk — SenseVoice emits no punctuation) + a final `Transcript:` block to copy after exiting the alt screen; listen keeps colored + timestamped + compute-time committed lines, a live partial, and **live translation of the current window** (`window_translation`, latest-wins stale-skip; committed chunks reuse it). `evaluate()` reports WER only for whitespace-segmented (en) text; CJK/mixed reports CER only (WER is meaningless on unsegmented zh — whole-text WER on zh inflates, same reason `_archive/benchmark-2026-08-24.md` scores zh with CER). **SenseVoice emits English in ALL CAPS (model artifact)** — `SpeechToText.extract_text` sentence-cases it for sensevoice; parakeet already outputs natural case and is untouched.
- STT = sherpa-onnx int8 only: `sensevoice` (default), `parakeet-tdt-0.6b-v2` (en-only/listen). Parakeet colors per word; SenseVoice output stays uncolored (no per-token scores in sherpa-onnx 1.13.0 — documented limitation). **SenseVoice emits English in ALL CAPS (model artifact)** — `SpeechToText.extract_text` sentence-cases it for sensevoice (live lines, final transcript, and evaluate reference all go through it); parakeet already outputs natural case and is untouched.
- Translate = `opus-mt-en-zh` (deterministic, ~1.2 s/sentence, en→zh).
- Weights shared with the benchmark under `data/benchmark/transcribe/models/` (anonymous HF download). App form undecided — **Phase 4** decides.

**Environment**:

- Container runs: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync <cmd>`; never plain `uv sync`/`uv run`. Dep pins: sherpa-onnx 1.13.0 + onnxruntime 1.24.4; **torch + torchaudio from the `pytorch-cpu` index on Linux** (PyPI wheels are CUDA and fail to dlopen on the container's CPU torch). **`sentencepiece` is a direct dep** (opus-mt's Marian tokenizer needs it — silently dropped with funasr's removal, see Phase 2). **matplotlib** is a direct dep again (audio.py plotting; also transitive via noisereduce). sherpa needs `libonnxruntime.so` on the loader path (self-healed in the harness). Container audio: libportaudio2 + pulseaudio for headless `sounddevice`; macOS: `_ensure_onnxruntime_dylib` self-heals the missing bundled onnxruntime.
- Bench assets under `data/benchmark/` (gitignored via `/data/`), grouped by task; the Phase 1 run record lives in `_archive/benchmark-2026-08-24.md` (moved there when Phase 2 completed).

## Phases

### Phase 1 — Model benchmark (model selection)

Status: **COMPLETE (2026-08-24)** — 60 samples / 15 models; listen → Parakeet TDT 0.6B v2, dictate → SenseVoiceSmall, translate → opus-mt-en-zh; whispercpp baseline superseded. Record: `_archive/benchmark-2026-08-24.md`.

### Phase 2 — Architecture restructure

Status: **COMPLETE (2026-08-24)** — restructure done in a lean 3-file shape (`transcribe.py` + `translate.py` + `__main__.py`, plus `audio.py` and the `benchmark.py` harness); adaptive-window live engine, threading-backlog closure, and the `docs/benchmark.md` → `_archive/` move all done.

- [x] **Feature scope**: `listen` = en-only STT (Parakeet v2) + translate en→cn, translate **ON** by default; `dictate` = mixed STT (SenseVoice) or `--language en-only` (Parakeet v2), default mixed, translate **OFF** by default. No model names exposed (the CLI resolves them internally).
- [x] **Lean layout** (2026-08-24 — an earlier 7-file concern-split was rejected as over-engineering): `transcribe.py` = the original committed module (VAD, `process_audio_segment`, `SpeechToText` dispatch, `RealTimeTranscribe` with its original `stt_model`/`translate_model`/`translate_to` API) with the sherpa backend folded in (`MODEL_SPECS`, `SherpaStt`, `SherpaSegment`, word-probs, download/dylib helpers); `translate.py` = `Translator` (opus-mt-en-zh, only backend); `__main__.py` = thin CLI; `audio.py` = restored committed class API (play/plot_waveform/record/convert_format/plot_mel_spectrogram); `benchmark.py` = the harness.
- [x] **CLI** (`__main__.py`): `uv run python -m interpreter <listen|dictate|benchmark>`; `listen [--no-translate] [--audio-file]`, `dictate [--translate] [--language en-only|mixed] [--audio-file]`; `benchmark` → passthrough to `interpreter.benchmark`; `run()`/`evaluate()` behind both live modes; README updated.
- [x] **Adaptive stability-window re-decode** (2026-08-24, user-requested): per-segment independent transcription replaced. Each segment re-decodes the whole window (growing-context re-decode — the only streaming option for these models, per sherpa-onnx #2918/#3573); stable earlier text commits and the window slides; `max_window_seconds` (60 s) caps cost. Dictate = clean self-correcting line + final copy block; listen keeps colored/timestamped lines. Gotchas logged: (a) en stability needs a **word-boundary** check (`abc|def` → `abcdef` re-segmentation must NOT commit; char-prefix is fine for CJK because concatenation reconstructs the text); (b) zh chunk joining is **display-dependent** — the live line and final copy block separate chunks with spaces (`_join_text_parts(force_space=True)`, user-requested: no punctuation, so segments need at least a space), while the metrics transcript keeps the unspaced join (spaces would inflate CER vs the unspaced zh reference); (c) cursor-up/erase-line in-place redraw breaks on macOS Terminal (leaks `^[[A` + blank lines into scrollback) → **alternate screen buffer + clear-and-reprint** instead; (d) committed listen chunks lost colors after the first — the stable path must **re-decode the newest segment alone** after a commit so the window's styled text stays colored and consistent (corrected-tail kept only as a fallback when the standalone decode is empty); (e) **translation worker died after the first sentence** — `_commit_chunk` appends its parallel lists non-atomically and a translation-thread redraw could read mid-append (`IndexError` kills the daemon thread silently, dropping all later translations) → commits now append **under `display_lock`**, and the worker wraps each item in try/except so one failing sentence can't kill the queue; (f) **dictation punctuation** — SenseVoice emits none; **zh chunks get no punctuation, separated by spaces only** (user preference), English chunks get `.` (`_chunk_punct`, skipped when the chunk already ends in punctuation for parakeet en-only); metrics stay unpunctuated (CER); (g) **translation is live now, not commit-gated** — the transcription worker enqueues the current window text (`_enqueue_window_translation`, deduped on unchanged text, `seq` for latest-wins stale-skip so the worker never falls behind), the worker stores it in `window_translation` shown under the partial line, and commits **reuse** it (no redundant re-translate); `_stop` reuses it for the final window translation.
- [x] **audio.py restored** (2026-08-24): original `AudioDataProcessor`/`AudioFileProcessor` API kept; `plot_mel_spectrogram` now implements whisper's log-mel pipeline in numpy (80 Slaney mel bins, power→mel→log10, whisper's clamp+rescale, viridis) — no whisper dep; `record` writes via soundfile (wavio dropped). Matplotlib re-added as a direct dep.
- [x] **Benchmark trim to sherpa-only**: faster-whisper / openai-whisper / funasr / modelscope / ark-asr adapters + their deps dropped; `--stream` hooks removed (no stream-capable model left; streaming probe moved to Phase 3). Keep sacrebleu/sacremoses (translate-task scoring). Canonical 60-sample results re-run for the two product models after the trim.
- [x] **Dep prune gotchas** (2026-08-24 — removing `funasr` transitively dropped deps the product still needs):
  - **`sentencepiece` re-added as a direct dep** — the opus-mt Marian tokenizer needs it; transformers masks the missing import as a misleading "Unrecognized configuration class" error.
  - **`torchaudio` re-added (pytorch-cpu pin)** — silero-vad's hub script imports it (`from silero_vad.utils_vad import ... torchaudio`); not imported by our src, so the prune missed it.
  - **`torch.hub.load` needs `trust_repo=True`** for silero-vad — the default prompts interactively and fails non-interactive (EOFError).
  - **`__main__.py` must call `main()`** — a missing call exits 0 silently (no-op CLI); caught when `benchmark --list` printed nothing.
- [x] **Threading backlog** (from the port — **closed 2026-08-24, no change needed**): the original concern — one `_transcription_worker` serially drains the queue and a slow decode backlogs latency — is resolved by measurement + design: (1) the live-mode compute-time display already instruments decodes: SenseVoice ~0.03–0.05 s and parakeet ~0.14–0.23 s per decode vs a segment cadence of seconds (VAD emits a segment only after ~0.5 s of silence), so decode time is 1–2 orders of magnitude below arrival rate; (2) the adaptive window bounds worst-case decode cost (`max_window_seconds` = 60 s → ≤ ~0.6 s at SenseVoice RTF 0.010), and the cap force-commits long continuous speech. The candidate refinements are each rejected: (a) instrumentation — already effectively done via the compute-time display; (b) parallel transcription workers — **breaks the window state machine** (re-decodes are serial by design, each depends on the previous window state); (c) `num_threads` — sherpa recognizers already run 4 threads (`SherpaStt` default); (d) noisereduce on its own stage — per-segment cost is small and not the bottleneck. Revisit only if a future phase moves to frame-level streaming (Phase 3 Parakeet Unified probe).
- [x] **Done** (2026-08-24): `docs/benchmark.md` moved → `_archive/benchmark-2026-08-24.md` as the historical run record; PLAN.md keeps the verdicts + link above. Stale `docs/benchmark.md` references in `transcribe.py`/`translate.py` docstrings updated to the `_archive/` path (2026-08-24).

### Phase 3 — Future features (Python, product-complete)

Status: **TODO**.

- **Speaker identification**: Parakeet can't do it (ASR only) — add a small WeSpeaker embedding model (CC-BY-4.0; `wespeaker_voxceleb_resnet34.onnx` for en, `wespeaker_zh_cnceleb_resnet34.onnx` for zh) via sherpa-onnx `SpeakerEmbeddingExtractor`/`SpeakerEmbeddingManager` (enrollment-based ID per segment); diarization (pyannote-3.0 segmentation + clustering, sherpa-onnx ≥1.10.28) as the unknown-speaker fallback. Same sherpa-onnx runtime — no Parakeet change.
- **Translate only other people's speech**: ties into speaker ID — suppress translation for enrolled self.
- **Streaming live partials** (deferred from Phase 2, 2026-08-24): benchmark **Parakeet Unified en 0.6B** (true `OnlineRecognizer` streaming, ~240 ms chunks, sherpa-onnx support added 2026) vs TDT v2 at low latency — if it wins on accuracy-at-latency, switch listen to it. TDT v2's "real-time" is simulated streaming only (#2918/#3573).

### Phase 4 — App form

Status: **TODO**.

- Decide app form after the Python product works (Phase 3 done). Recommended default: local FastAPI web UI over the same package; native macOS/iOS later via sherpa-onnx native bindings stays open.
