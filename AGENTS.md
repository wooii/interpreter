# AGENTS.md

A real-time transcription assistant with optional translation — see `README.md` for the product. See `PLAN.md` for the modernization plan, known issues & gotchas, and the current architecture and state notes.

You (the AI agent) run only in the Linux container. The host is macOS and handles anything needing keys.

## Environment

- uv-managed, Python 3.14, `uv_build` backend, src-layout (`src/interpreter/`).
- **Two venvs:** `.venv` (macOS host, plain `uv sync`) and `.venv-container` (Linux container, `UV_PROJECT_ENVIRONMENT=.venv-container uv sync`). **Never** run plain `uv sync`/`uv run` in the container — it overwrites the host venv with Linux binaries.
  - Runs: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync <cmd>`
  - Deps: `UV_PROJECT_ENVIRONMENT=.venv-container uv add --no-sync <pkg>`
  - After dependency changes, `uv sync` is needed on **both** venvs — you run the container side, flag the host side to the human.
- Ruff/ty/pytest are wired as prek commit/push hooks (`prek.toml`); ruff/ty have no dedicated config sections yet. There are no tests and no CI — don't invent them until the plan says so. Run lint/format/typecheck manually with the uv prefix.

## Running (from project root)

- Real entrypoint: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync python -m interpreter.transcribe`
- Benchmark: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync python research/model_comparison.py`
- OpenAI-API path: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync python research/record_and_transcribe_using_openai_api.py`
- Requires a working microphone (sounddevice `InputStream`).
