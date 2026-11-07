# Interpreter

A real-time, fully local transcription assistant with optional translation — built for meetings and live conversations where speed, privacy, and accuracy matter. Everything runs on-device: sherpa-onnx speech recognition, opus-mt translation, and WeSpeaker speaker ID. There is no cloud path and no API keys, and models are internal per-task picks rather than user settings.

- **Listen:** English transcription with timestamps, optional Chinese translation, and automatic `[self]` / `[other]` speaker tags (`[?]` when a voice cannot be confidently matched — never a silent mislabel).
- **Dictate:** clean, continuously self-correcting zh/en transcription ready to copy; per-word confidence colours in English-only mode, plain text in mixed mode.
- **Adaptive re-transcription:** every new utterance re-decodes the recent window, so earlier text self-corrects as context grows.
- **Desktop app:** a native PySide6 window over the same engine, with a session sidebar, editable transcripts that autosave, resume, rename, reveal in Finder, drag-out, and delete.

Agent working rules: `AGENTS.md`. Living plan: `PLAN.md`.

## Structure

- `src/interpreter/`: the installed package:
  - `transcribe.py`: Silero VAD → sherpa-onnx STT → optional translation, adaptive stability-window re-decode, and terminal rendering.
  - `translate.py`: deterministic `opus-mt-en-zh` translation.
  - `speaker.py`: online WeSpeaker speaker assignment with the `[?]` uncertainty band and gray-zone promotion.
  - `common.py`: onnxruntime self-heal and CJK helpers.
  - `audio.py`: microphone recording, playback, format conversion, and waveform/mel-spectrogram plotting helpers.
  - `app.py`: PySide6 desktop GUI (`python -m interpreter app`).
  - `benchmark.py`: the STT / translation / speaker benchmark harness.
  - `benchmark_assets.py`: downloads and materialises the gitignored benchmark corpus.
  - `progress.py`: model-download progress reporting and cancellation.
  - `__init__.py`: `DATA_DIR` and model-directory constants.
  - `__main__.py`: CLI entry point.
- `benchmark/`: tracked corpus definitions — `transcribe.json` (59 STT samples), `translate.json` (100 WMT19 sentences), `speaker.json` (LibriSpeech dev-clean split).
- `docs/`: source-controlled benchmark records (`benchmark_transcribe.md`, `benchmark_translate.md`, `benchmark_speaker.md`).
- `scripts/`: `make_app.sh` (macOS `.app` bundle) and `make_icon.py` (icon generation).
- `data/`: gitignored local data — model weights under `models/`, session recordings under `listen/` and `dictate/`, and the benchmark cache under `benchmark/`.

## Setup

Requires macOS 12+ (for the desktop app and `.app` bundle), [uv](https://docs.astral.sh/uv/), and a working microphone. The project is uv-managed; from the repo root:

```bash
uv sync
```

This creates the host `.venv/`. AI agents run in a Linux container with a separate `.venv-container` — see `AGENTS.md`. Model weights download on first use (~1 GB for listen; ~240 MB for mixed dictate), not during `uv sync`; the status line shows loading, and the GUI shows per-file progress (Stop cancels a first-run download).

## Modes

One pipeline (mic → VAD → adaptive-window STT → optional translation), two config-driven modes:

| Mode | Who speaks | What comes out |
|------|------------|----------------|
| **listen** | Others (English) | English transcript + Chinese translation, coloured + timestamped lines, auto speaker tags (`[self]` / `[other]`; `[?]` when a voice can't be confidently matched) |
| **dictate** | Self (zh/en mixed) | Clean dictation, one self-correcting line (per-word confidence colours in English-only mode; plain in mixed) |

## Running

Run from the project root (where `pyproject.toml` is). Models are picked internally per task — no model names to configure.

```bash
uv run python -m interpreter listen                # transcribe English + translate to Chinese (translate on, speaker ID on)
uv run python -m interpreter listen --no-translate # transcribe English only
uv run python -m interpreter dictate               # dictation, zh/en mixed (no translation)
uv run python -m interpreter dictate --en          # dictation, English only
uv run python -m interpreter app                   # local desktop GUI (native window, same engine)
uv run python -m interpreter benchmark --list      # benchmark harness
```

`uv run python -m interpreter app` opens the GUI directly from source — no bundle build needed. The `.app` below is optional; it only adds double-click launch and the app icon.

Sessions record to `data/listen/` or `data/dictate/` as FLAC with a sibling `.txt` transcript and a `.styled` twin that keeps per-word confidence colours. In the GUI, transcripts are editable once idle and autosave ~0.8 s after the last keystroke; a session can be resumed, and its FLAC/`.txt`/`.styled` files rename together.

### macOS app bundle (double-click to launch, with app icon)

Needs macOS 12+ and `iconutil` (built into macOS):

```bash
git clone https://github.com/wooii/interpreter.git
cd interpreter
curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv sync
bash scripts/make_app.sh
```

If `uv` is not found in the current shell after the installer, the `command -v uv || export PATH=...` line adds `~/.local/bin`. `uv sync` creates the native macOS venv; `make_app.sh` builds `dist/Interpreter.app`.

**After building**

1. Launch it with `open dist/Interpreter.app`, or double-click it in Finder.
2. On first use, allow microphone access when macOS asks.
3. Choose Listen or Dictate and press Start. The header progress bar shows one model download/loading step at a time; **Stop** cancels a first-run download and returns to idle.
4. Optional: drag `dist/Interpreter.app` to `/Applications`. It still runs this repo's `.venv`; do not copy the bundle or `.venv` to another Mac.
5. If a Finder launch fails, check `~/Library/Logs/Interpreter.log`.

The bundle is a thin launcher for this repo's `.venv`: code changes are picked up without rebuilding — rerun `make_app.sh` only after moving the repo or changing the icon. The launcher has the repo path baked in at build time (with a walk-up fallback if the repo moved).

## Benchmark

The benchmark harness is separate from the app; normal app use doesn't need it.

Tracked corpus definitions:

- `benchmark/transcribe.json` — 59 STT samples (19 en / 20 zh / 20 mixed) with references, block boundaries, source locators, and mixed-clip recipes.
- `benchmark/translate.json` — 100 WMT19 `newsdev2019` en→zh sentences.
- `benchmark/speaker.json` — LibriSpeech dev-clean enroll/test split (14 speakers, 3 enroll + 10 test clips each).

Audio and result files are a gitignored cache under `data/benchmark/`. Fetch with:

```bash
uv run python -m interpreter benchmark --fetch                 # STT corpus (~380 MB download)
uv run python -m interpreter benchmark --task speaker --fetch  # speaker corpus (~340 MB download)
```

Runs fetch missing assets automatically unless `--no-fetch` is passed, and `--list` works from a fresh clone. Results and merged tables land under `data/benchmark/*/results/`; `docs/` contains the source-controlled historical records and is not regenerated.

Sources/licences: LibriSpeech (CC-BY-4.0, OpenSLR SLR12); AISHELL-1 (Apache-2.0, Hugging Face `AISHELL/AISHELL-1`); WMT19 `newsdev2019` (research use; cite the WMT19 shared-task overview).

## License

MIT
