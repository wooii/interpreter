# Interpreter

A real-time, fully local transcription assistant with optional translation — built for meetings and live conversations where speed, privacy, and accuracy matter.

## Key Features

- Real-time continuous local transcription, with per-word confidence color coding
- Optional live local translation
- Mixed language (zh/en) dictation

## Modes

One pipeline (mic → VAD → STT → optional translation), two config-driven modes:

| Mode | Who speaks | What comes out |
|------|------------|----------------|
| **listen** | Others (English) | English transcript + Chinese translation |
| **dictate** | Self (zh/en mixed) | Clean mixed-language dictation |

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

Run from the project root (where `pyproject.toml` is), e.g. the real-time transcription entrypoint:

```bash
uv run python -m interpreter.transcribe
```

The first run downloads the STT model weights (sherpa-onnx int8, e.g. `sensevoice` / `parakeet-tdt-0.6b-v2`) anonymously from Hugging Face.

## Roadmap

See `PLAN.md` for the modernization plan.

## License
MIT
