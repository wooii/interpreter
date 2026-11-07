# Interpreter

A real-time, fully local transcription assistant with optional translation — built for meetings and live conversations where speed, privacy, and accuracy matter.

## Key Features

- Real-time continuous local transcription, with per-word confidence color coding (listen; dictate in English-only mode)
- **Auto speaker ID in listen mode** — each segment is automatically tagged `[self]` / `[other]` as voices appear (no enrollment step; WeSpeaker, fully local); segments that can't be confidently matched are flagged `[?]` instead of being mislabeled, and a close-but-distinct voice is promoted to `[other]` once it forms a consistent cluster
- Adaptive re-transcription: as each utterance arrives, earlier text is re-decoded with more context and self-corrects on the fly (growing-context re-decode — no true streaming in the current models)
- Optional live local translation
- Mixed language (zh/en) dictation
- Dictate mode outputs a clean, continuously-updated transcript that's ready to copy (no timestamps; per-word confidence colors in English-only mode, plain in mixed)
- **Local desktop app** (`app`, PySide6/Qt) — a native window UI over the same engine: live listen/dictate, auto speaker tags, translations, and a session history sidebar with transcript view (editable — edits autosave to the `.txt`), **resume** (record more into the same session's audio + transcript), rename sessions in the sidebar (all related files rename), reveal-in-Finder, drag-out and delete

## Modes

One pipeline (mic → VAD → adaptive-window STT → optional translation), two config-driven modes:

| Mode | Who speaks | What comes out |
|------|------------|----------------|
| **listen** | Others (English) | English transcript + Chinese translation, colored + timestamped lines, auto speaker tags (`[self]` / `[other]`; `[?]` when a voice can't be confidently matched) |
| **dictate** | Self (zh/en mixed) | Clean dictation, one self-correcting line (per-word confidence colors in English-only mode; plain in mixed) |

## Requirements

- macOS 12+ (for the desktop app and `.app` bundle)
- [uv](https://docs.astral.sh/uv/)
- A working microphone

## Setup

The project is uv-managed. Create (or re-sync after dependency changes) the host venv:

```bash
uv sync
```

AI agents run in a Linux container with a separate `.venv-container` — see `AGENTS.md`.

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

**macOS app bundle (double-click to launch, with app icon)** — needs macOS 12+ and `iconutil` (built into macOS):

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
3. Choose Listen or Dictate and press Start. The header progress bar shows one model download/loading step at a time; **Stop** cancels a first-run download and returns to idle. Model weights download on first use (~1 GB for listen; ~240 MB for mixed dictate), not during `uv sync`; the status line shows loading.
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

## Roadmap

See `PLAN.md` for the modernization plan and `docs/` for the benchmark records.

## License
MIT
