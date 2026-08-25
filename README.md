# Interpreter

A real-time, fully local transcription assistant with optional translation — built for meetings and live conversations where speed, privacy, and accuracy matter.

## Key Features

- Real-time continuous local transcription, with per-word confidence color coding (listen)
- **Auto speaker ID in listen mode** — each segment is automatically tagged `[self]` / `[other]` as voices appear (no enrollment step; WeSpeaker, fully local); segments that can't be confidently matched are flagged `[?]` instead of being mislabeled, and a close-but-distinct voice is promoted to `[other]` once it forms a consistent cluster
- Adaptive re-transcription: as each utterance arrives, earlier text is re-decoded with more context and self-corrects on the fly (growing-context re-decode — no true streaming in the current models)
- Optional live local translation
- Mixed language (zh/en) dictation
- Dictate mode outputs a clean, continuously-updated transcript that's ready to copy (no timestamps/colors)

## Modes

One pipeline (mic → VAD → adaptive-window STT → optional translation), two config-driven modes:

| Mode | Who speaks | What comes out |
|------|------------|----------------|
| **listen** | Others (English) | English transcript + Chinese translation, colored + timestamped lines, auto speaker tags (`[self]` / `[other]`; `[?]` when a voice can't be confidently matched) |
| **dictate** | Self (zh/en mixed) | Clean mixed-language dictation, one self-correcting line |

## Requirements

- [uv](https://docs.astral.sh/uv/)
- A working microphone

## Setup

The project is uv-managed. Create (or re-sync after dependency changes) the host venv:

```bash
uv sync
```

AI agents run in a Linux container with a separate `.venv-container` — see `AGENTS.md`.

## Running

Run from the project root (where `pyproject.toml` is). Models are picked
internally per task — no model names to configure.

```bash
uv run python -m interpreter listen                # transcribe English + translate to Chinese (translate on, speaker ID on)
uv run python -m interpreter listen --no-translate # transcribe English only
uv run python -m interpreter dictate               # dictation, zh/en mixed (no translation)
uv run python -m interpreter dictate --en          # dictation, English only
uv run python -m interpreter benchmark --list      # benchmark harness
```

The first run downloads the STT model weights (sherpa-onnx int8: `sensevoice` for mixed, `parakeet-unified-en-0.6b` for en-only) and the en→zh translation model `opus-mt-en-zh` anonymously from Hugging Face.

## Roadmap

See `PLAN.md` for the modernization plan.

## License
MIT
