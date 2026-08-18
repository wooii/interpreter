# Interpreter

A real-time transcription assistant, with optional translation: records microphone audio, transcribes locally in real time, and optionally translates, with an online fallback — designed for scenarios where speed and accuracy are critical, such as meetings or live conversations.

## Key Features
- Real-time continuous local transcription, with per-word confidence color coding
- Optional local translation
- Online fallback for both transcription and translation

## Requirements
- [uv](https://docs.astral.sh/uv/)
- A working microphone
- API keys for online fallback

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

## Roadmap

See `PLAN.md` for the modernization plan.

## License
MIT
