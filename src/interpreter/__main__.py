"""Command-line interface: `uv run python -m interpreter <listen|dictate|benchmark>`.

Modes (feature scope, PLAN.md 2026-08-24 — models are internal picks, no
model names to configure):

- `listen`   — transcribe English, translate en->cn. Translation ON by default
               (`--no-translate` to turn off).
- `dictate`  — dictation in en/cn/mixed. `--language en-only|mixed` (default
               mixed), translation OFF by default (`--translate` to turn on).
- `benchmark`— the model benchmark harness (interpreter.benchmark).

Both live modes record the session to a WAV in data/ and evaluate WER/CER
afterwards (offline re-transcribe of the saved file).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Sequence
from pathlib import Path

from interpreter import DATA_DIR
from interpreter.transcribe import (
    STT_MODEL_EN_ONLY,
    STT_MODEL_MIXED,
    RealTimeTranscribe,
)
from interpreter.translate import TRANSLATE_MODEL

MAX_SEGMENT_DURATION = 10.0

def _run_live(
    *,
    stt_model: str,
    translate: bool,
    audio_file: Path | None,
    mode: str,
    plain_output: bool = False,
) -> None:
    if audio_file is None:
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        audio_file = DATA_DIR / f"{mode}_{timestamp}.wav"
    rtt = RealTimeTranscribe(
        audio_file_path=audio_file,
        stt_model=stt_model,
        translate_model=TRANSLATE_MODEL,
        translate_to="Chinese" if translate else None,
        max_segment_duration=MAX_SEGMENT_DURATION,
        plain_output=plain_output,
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
    p_listen.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="session WAV path (default: data/)",
    )

    p_dictate = sub.add_parser(
        "dictate", help="dictate in en/cn/mixed (translation off by default)"
    )
    p_dictate.add_argument(
        "--translate", action="store_true", help="also translate en->cn"
    )
    p_dictate.add_argument(
        "--language",
        choices=("en-only", "mixed"),
        default="mixed",
        help="language scope (default: mixed)",
    )
    p_dictate.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="session WAV path (default: data/)",
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
            audio_file=args.audio_file,
            mode="listen",
        )
    elif args.command == "dictate":
        stt_model = STT_MODEL_EN_ONLY if args.language == "en-only" else STT_MODEL_MIXED
        _run_live(
            stt_model=stt_model,
            translate=args.translate,
            audio_file=args.audio_file,
            mode="dictate",
            plain_output=True,
        )


if __name__ == "__main__":
    main()
