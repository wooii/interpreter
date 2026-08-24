# AGENTS.md

Environment & setup for AI agents working in this repo.

The product lives in `README.md`; the plan lives in `PLAN.md`.

## Documentation conventions

Docs have fixed roles — keep the right info in the right file.

**`README.md` (product) and `AGENTS.md` (environment & setup) are stable:**
- Edit only when a task changes something they claim — a command, a venv rule, a run instruction, setup steps, or conventions.
- Update the relevant file as part of that same change — don't leave it stale.

**`PLAN.md` is the living doc** with two sections:

- **Current state** — facts that hold *now*: architecture, product decisions, dependency/environment state.
- **Phases** — the work, current and future. Each phase has a `Status:` line and checklist items. Gotchas live inside the item that resolves them — no separate list, no tags. Don't fix silently: log the gotcha in the item, and it resolves when the item is checked off.

**As you work:**
- Finish an item: check it off, update its `Status:` line.
- Hit a gotcha: log it in the phase item that will resolve it.
- Change material state (deps added, files moved): log it under "Current state"; update architecture notes if the layout changed.
- A section (architecture notes, phase detail) grows unwieldy: flag it to the human instead of letting `PLAN.md` sprawl.


## Environment

- AI agent runs only in the Linux container; host is macOS.
- uv-managed, Python 3.14, `uv_build` backend, src-layout.
- **Two venvs:** `.venv` (macOS host, plain `uv sync`) and `.venv-container` (Linux container, `UV_PROJECT_ENVIRONMENT=.venv-container uv sync`).
  - **Never** run plain `uv sync`/`uv run` in the container — it overwrites the host venv with Linux binaries.
  - Runs: `UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync <cmd>`
  - Deps: `UV_PROJECT_ENVIRONMENT=.venv-container uv add --no-sync <pkg>`
  - After dependency changes, sync **both** venvs — you run the container side, flag the host side to the human.
- Ruff/ty/pytest are wired as prek commit/push hooks (`prek.toml`); no dedicated ruff/ty config yet. Run lint/format/typecheck manually with the uv prefix above.
- No tests, no CI — don't scaffold either until `PLAN.md` calls for it.
- Leave `git commit` / `git push` to the human, only suggest a commit title.

## Long-running task protocol

Long benchmark/model runs can hang or fail silently. Interrupt and diagnose like a human would — never commit to one blocking call or blindly re-run.

- **State stop-criteria upfront** when launching a long job: e.g. *"kill if the log shows 3+ consecutive sample FAILEDs, no progress line for N minutes, or a download stalls."*
- **Run long jobs in the background** (`cmd > log 2>&1 &`) with a log file, poll progress every ~60–120 s, and keep the call bounded (`timeout`) so a hang fails fast.
- **On interruption (yours or the human's): diagnose first** — tail the log, find the *first* failure, build a minimal repro, fix it, then resume only the remaining work (e.g. the un-run models), never re-run everything.
- **On failure, check the result artifacts**: an all-FAILED or suspiciously small results file means the run "succeeded" while failing — inspect before trusting it.
- **Stop early on systemic patterns**: if every sample of the first model section fails, don't let the rest of the run continue — kill, fix, restart.

