"""Progress and cancellation helpers for model downloads.

The GUI passes a :data:`interpreter.common.ProgressCallback` and a
``threading.Event`` into the loaders. Hugging Face downloads use a custom
``tqdm_class`` so byte progress is reported and a cancel raises
:class:`interpreter.common.OperationCancelled`; plain URL downloads use
:func:`download_url` for the same behavior.
"""

from __future__ import annotations

import threading
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub.utils import tqdm as hf_tqdm

from interpreter.common import ProgressCallback, check_cancel

# Only one model download runs at a time. Four loaders start concurrently, but
# interleaved byte progress made the GUI bar jump between models (e.g. 0% STT
# and 85% translation); serializing downloads gives one clean, monotonic bar
# while model loading still runs concurrently after each download releases it.
_download_lock = threading.Lock()


@contextmanager
def download_guard():
    """Serialize downloads so progress is not interleaved across models."""
    with _download_lock:
        yield


def hf_tqdm_class(
    label: str,
    progress: ProgressCallback | None,
    cancel: threading.Event | None,
) -> type[hf_tqdm]:
    """Build a tqdm subclass for ``hf_hub_download(..., tqdm_class=...)``."""

    class _ProgressTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            # The GUI owns the visible progress bar; keep the terminal clean.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> None:
            super().update(n)
            if progress is not None:
                fraction = self.n / self.total if self.total else None
                progress(label, fraction)
            check_cancel(cancel)

    return _ProgressTqdm


def download_url(
    url: str,
    dest: Path,
    *,
    label: str,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    timeout: int = 120,
) -> Path:
    """Download ``url`` to ``dest`` with progress and cancellation.

    Writes to ``dest.part`` and renames only after a complete download, so a
    cancelled/failed download never looks like a usable model file.
    """
    check_cancel(cancel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=timeout) as src,
            part.open("wb") as out,
        ):
            raw_total = src.headers.get("Content-Length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else None
            done = 0
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(label, done / total if total else None)
                check_cancel(cancel)
        check_cancel(cancel)
        part.replace(dest)
        return dest
    except BaseException:
        part.unlink(missing_ok=True)
        raise
