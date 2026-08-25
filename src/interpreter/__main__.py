"""Command-line interface: `uv run python -m interpreter <listen|dictate|benchmark>`.

Modes (feature scope, PLAN.md 2026-08-24 — models are internal picks, no
model names to configure):

- `listen`   — transcribe English, translate en->cn. Translation ON by default
               (`--no-translate` to turn off). Auto speaker ID ON: each segment
               is auto-assigned to a speaker (`self`, `other`, ...) as voices
               appear.
- `dictate`  — dictation in en/cn/mixed, no translation. `--en` restricts to
               English only (default: mixed zh/en).
- `benchmark`— the model benchmark harness (interpreter.benchmark).

Both live modes record the session to a FLAC in data/listen/ (listen) or
data/dictate/ (dictate), write the transcript as a sibling .txt, and evaluate
WER/CER afterwards (offline re-transcribe of the saved file).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Sequence

from interpreter import DATA_DIR
from interpreter.transcribe import (
    STT_MODEL_EN_ONLY,
    STT_MODEL_MIXED,
    RealTimeTranscribe,
)
from interpreter.translate import TRANSLATE_MODEL


def _run_live(
    *,
    stt_model: str,
    translate: bool,
    mode: str,
    clean: bool = False,
    speaker_id: bool = False,
) -> None:
    mode_dir = DATA_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    audio_file = mode_dir / f"{mode}_{timestamp}.flac"
    print(
        "Loading models (first run downloads weights from Hugging Face)...",
        flush=True,
    )
    rtt = RealTimeTranscribe(
        audio_file_path=audio_file,
        stt_model=stt_model,
        translate_model=TRANSLATE_MODEL,
        translate_to="Chinese" if translate else None,
        clean=clean,
        speaker_id=speaker_id,
    )
    rtt.run()
    rtt.evaluate()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interpreter",
        description="Real-time, fully local transcription + translation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_listen = sub.add_parser(
        "listen", help="transcribe English + translate to Chinese (translation on)"
    )
    p_listen.add_argument(
        "--no-translate", action="store_true", help="disable translation"
    )

    p_dictate = sub.add_parser(
        "dictate", help="dictate in en/cn/mixed (no translation)"
    )
    p_dictate.add_argument(
        "--en", action="store_true", help="English only (default: mixed zh/en)"
    )

    sub.add_parser(
        "benchmark", help="model benchmark harness (see --help after benchmark)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        sys.argv = ["interpreter benchmark", *argv[1:]]
        from interpreter.benchmark import main as benchmark_main

        benchmark_main()
        return

    args = _build_parser().parse_args(argv)

    if args.command == "listen":
        _run_live(
            stt_model=STT_MODEL_EN_ONLY,
            translate=not args.no_translate,
            mode="listen",
            speaker_id=True,
        )
    elif args.command == "dictate":
        stt_model = STT_MODEL_EN_ONLY if args.en else STT_MODEL_MIXED
        _run_live(
            stt_model=stt_model,
            translate=False,
            mode="dictate",
            clean=True,
        )


if __name__ == "__main__":
    main()
