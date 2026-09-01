"""Shared utilities used by multiple interpreter modules.

This is the single place for small helpers that would otherwise be duplicated
across ``transcribe``, ``benchmark`` and ``speaker``. Keep it dependency-light
(no heavy imports) so importing it never pulls in sherpa/transformers.

Currently:

- sherpa-onnx / onnxruntime shim (macOS dylib copy + Linux ``libonnxruntime.so``
  symlink + ``LD_LIBRARY_PATH``) — see :func:`ensure_onnxruntime`
- CJK helpers (``_contains_cjk`` / ``_is_cjk_char``) — used by the live engine
  and the benchmark's per-block scoring

Heavy model registries (``MODEL_SPECS``) stay in ``transcribe.py`` (product
source of truth); the benchmark imports from there.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# onnxruntime shim for sherpa-onnx
# ---------------------------------------------------------------------------


def onnxruntime_capi_dir() -> Path | None:
    """Path to the onnxruntime capi dir (Linux), or None if unavailable."""
    if sys.platform != "linux":
        return None
    try:
        import onnxruntime
    except Exception:  # noqa: BLE001 - missing dep
        return None
    capi = Path(onnxruntime.__file__).parent / "capi"
    return capi if capi.exists() else None


def ensure_onnxruntime() -> None:
    """Self-heal sherpa-onnx's onnxruntime dlopen on both platforms.

    Best-effort: never raises, lets the real import error surface later.
    """
    if sys.platform == "darwin":
        try:
            import onnxruntime

            spec = importlib.util.find_spec("sherpa_onnx")
            if spec is None or not spec.submodule_search_locations:
                return
            sherpa_lib = Path(spec.submodule_search_locations[0]) / "lib"
            sherpa_lib.mkdir(parents=True, exist_ok=True)
            capi = Path(onnxruntime.__file__).parent / "capi"
            for src in capi.glob("libonnxruntime*.dylib"):
                dest = sherpa_lib / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
        except Exception:  # noqa: S110, BLE001 - best-effort fix
            pass
        return
    try:
        import onnxruntime

        capi = onnxruntime_capi_dir()
        if capi is None:
            return
        plain = capi / "libonnxruntime.so"
        if not plain.exists():
            libs = sorted(capi.glob("libonnxruntime.so.*"))
            if libs:
                plain.symlink_to(libs[-1].name)
        path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{capi}{os.pathsep}{path}"
    except Exception:  # noqa: S110, BLE001 - best-effort env fix
        pass


# ---------------------------------------------------------------------------
# CJK helpers — shared by engine and benchmark
# ---------------------------------------------------------------------------


def _contains_cjk(text: str) -> bool:
    """True if *text* contains any CJK Unified Ideograph (``\\u4e00``–``\\u9fff``)."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _is_cjk_char(ch: str) -> bool:
    """True if *ch* is a single CJK Unified Ideograph."""
    return "\u4e00" <= ch <= "\u9fff"


def _has_cjk(text: str) -> bool:
    """Alias for :func:`_contains_cjk` (benchmark legacy name)."""
    return _contains_cjk(text)
