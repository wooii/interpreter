# AGENTS.md

Working rules for AI agents. At the start of every task, read `README.md` and `PLAN.md` along with this file.

## Docs & workflow

Each fact lives in one file — never duplicate across files, and do not wrap Markdown lines at a fixed width:

- **`README.md`:** project overview, structure, host setup, and usage. Update it in the same change whenever any of these change.
- **`AGENTS.md`:** shared working rules across repos (only rules that cannot be inferred from code). Human-owned: never edit it during a task; propose changes to the human instead.
- **`PLAN.md`:** repo-specific state and plans (keep under 100 lines). Top-level bullets hold lasting repo-wide decisions and constraints that don't belong in `README.md` or `AGENTS.md` (one short paragraph per topic; no folder maps or command lists). Below them, each open unit of work is a flat `## <commit title>` section containing its related state, gotchas, and `- [ ]` items. When a section finishes, use its heading as the suggested commit title, fold any new lasting decision into the top bullets, and delete the section (git history records completed work).

Before reporting a task done: run the container checks below if code changed, update `README.md` and `PLAN.md` so no doc is stale, and suggest a commit title. Leave `git add`, `git commit`, and `git push` to the human, and never discard uncommitted work or untracked files without asking.

## Privacy & safety

Assume the model provider logs and trains on chat history — every file read, search match, command output, prompt, and reply is sent to external servers:

- **Check names before reading content:** stay within this repo unless the task requires otherwise. Scan folder and file names first; if a path suggests personal records (finance, health, family, notes), user data (`data/`), or proprietary strategies/algorithms not directly requested by the user, do not blindly read or search its contents — ask the human first.
- **Inspect structure, not raw data:** `data/` is gitignored and sensitive; never commit it or copy it into tracked files. Do not read full data files into chat (`read`, `cat`, PDFs, images), print raw records in logs/tracebacks, or pass rows into prompts or subagents. When you need to understand a dataset, run a small local command or script to check only column names, headers, row counts, schemas, or summary statistics (hiding small counts and unique values), and reproduce bugs with synthetic rows.
- **Never send private data externally:** do not send raw `data/` contents or identifying values to external model APIs (including LLM calls from scripts), web searches, `curl`, or third-party services. If sharing a small redacted excerpt is ever unavoidable, explain why and get human permission first.
- **Never read or print secrets:** API keys live in `~/data/_keys/` on the host or a gitignored `.env`. Never open, `cat`, or `grep` `.env` or key files (and don't ask to), never create `~/data/_keys/` from the container, and never print or log key values. Read keys lazily through environment variables or app settings (no import-time file reads).

## Environment

The agent runs in the Linux container (`.venv-container/`); the human works on the host (`.venv/`).

- **Container commands:** prefix every container `uv` command (`sync`, `run`, `add`, `remove`) with `UV_PROJECT_ENVIRONMENT=.venv-container` (unprefixed commands overwrite the host `.venv/` with Linux binaries; if that happens, ask the human to run `uv sync` on the host), and pass `--no-sync` to `uv run` so it doesn't re-sync on every call (`UV_PROJECT_ENVIRONMENT=.venv-container uv run --no-sync <cmd>` for `pytest`, `ruff check .`, `ruff format .`, and `ty check`). When dependencies or packaging change, run `UV_PROJECT_ENVIRONMENT=.venv-container uv sync` and ask the human to run `uv sync` on the host.
- **Host-only git hooks (`prek.toml`):** `.git/hooks/{pre-commit,pre-push,commit-msg}` — `ruff` on commit, `ty` + `pytest` on push, Conventional Commits validation on the message. Never install or modify git hooks from the container; run the container checks manually instead, and ask the human to run `uv run prek install` (one command installs all three) on the host if hooks are missing.

## Code & naming

- **Paths & imports:** define shared path constants (`PROJECT_ROOT`, `DATA_DIR`, `KEYS_DIR`) in `src/<package>/__init__.py` instead of relying on `cwd` (standalone PEP 723 scripts may use `cwd` or CLI args). Import from the installed package, never via `sys.path` hacks.
- **Clean code & self-contained tests:** delete dead code outright rather than commenting it out or keeping unused shims. Keep `pytest` fast and offline: mock network/API calls and use temporary paths (`tmp_path`) so tests never touch live APIs, `.env`, or `data/`.
- **File naming (`snake_case`):** use lowercase ASCII letters, digits, `_`, `-`, `.` only (hyphens only in ISO dates `YYYY-MM-DD`: leading when the date identifies the item like `2025-05-06_contract.pdf`, trailing for snapshots like `backup_2026-10-12.db`, `_YYYY-MM-DD_HH-MM-SS` for captures; no `final`, `new`, `old`, `copy`). Standard tool files (`pyproject.toml`, `README.md`, `AGENTS.md`) and external files keep their required names, while Python modules stay `snake_case`.
- **Descriptive stems:** name the subject, not the mechanism or frequency (`order_parser.py`, not `parser.py`; `monthly_report.py`, not `monthly.py`; no generic `utils` or `temp`). Don't repeat the folder name (`api/client.py`, not `api/api_client.py`), except for a topic's main data file (`inventory/inventory.csv`) or sibling prefixes (`db_import.py`, `db_export.py`). Share one stem across a script, its docs, and its data.
- **Never rename existing files on your own:** apply naming rules to new files, but only propose renaming an existing file when you are already editing it — state the old name, new name, reason, and affected files, and wait for approval (unless you created it in the current session and nothing references it).

## Long-running tasks

Long benchmark or model runs can hang or fail silently — never run them as a single unbounded blocking command:

- **Define stop conditions upfront:** set clear failure thresholds (e.g. repeated failures, no progress for N minutes, stalled downloads) and kill the job as soon as one triggers.
- **Run in the background with a log:** launch with `cmd > log 2>&1 &`, poll progress every 1–2 minutes with bounded timeouts, and check output files afterward (an all-failed or tiny output file means the run failed even if the exit code was 0).
- **Diagnose and resume on failure:** check the log for the *first* error, reproduce and fix it on a minimal case, then resume only the remaining work rather than restarting from scratch.
