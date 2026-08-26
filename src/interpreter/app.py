"""Local desktop GUI (PySide6): native Qt window over the same engine the CLI
uses (`python -m interpreter app`).

History: the first GUI used pywebview (native window + HTML/JS). Its WKWebView
compositing tripped the host's external display link — the Dell flickered when
the window was moved onto it even with no session running. That is a webview
compositing artifact, not engine behavior, so the stack was switched to Qt
(PySide6): plain AppKit windows, no webview anywhere in the chain. The
engine, controller and session handling are unchanged.

Architecture: the engine runs in a background thread and pushes immutable
snapshots into a `SnapshotStore`; a QTimer on the main thread polls the store
and renders into widgets — the same decoupled design as the pywebview UI.
"""

from __future__ import annotations

import datetime
import difflib
import html
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC
from html import unescape
from pathlib import Path
from urllib.parse import quote

import sounddevice as sd
from PySide6.QtCore import QMimeData, QRectF, QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QColor,
    QFontDatabase,
    QIcon,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from interpreter import DATA_DIR, PROJECT_ROOT
from interpreter.speaker import EN_MODEL
from interpreter.transcribe import (
    STT_MODEL_EN_ONLY,
    STT_MODEL_MIXED,
    BaseRenderer,
    RealTimeTranscribe,
    TranscriptSnapshot,
    _flowing_styled_parts,
    _flowing_text,
    _has_confidence_colors,
    _strip_timing,
)
from interpreter.translate import TRANSLATE_MODEL

__all__ = ["MainWindow", "SessionController", "SnapshotStore", "main"]

_MODE_DIRS = ("listen", "dictate")

_BASE_FONT_PT = 14
_TRANSCRIPT_FONT_PT = 15
_FONT_RANGE = (10, 22)


def _saved_font_size() -> int:
    """Persisted app font size (QSettings stores it as string or int)."""
    try:
        return int(str(QSettings().value("fontSize", _BASE_FONT_PT)))
    except TypeError, ValueError:
        return _BASE_FONT_PT


@contextmanager
def _quiet_portaudio():
    """PortAudio prints PaMacCore errors straight to the stderr FD (C-level
    fprintf, not Python), so a Python-level redirect cannot catch them — dup
    the fd to /dev/null for the duration."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(null)
        os.close(saved)


# Rounded action buttons (Start/Stop/Reveal/Delete) — QSS replaces the native
# macOS look, so the full style must be spelled out. Baby-blue palette
# matching the app icon (#4f8cff family): Start is the filled primary action;
# the rest are neutral with blue tints.
_ACTION_BUTTON_QSS = """
QPushButton {
    background: #ffffff;
    border: 1px solid #b9d4f7;
    border-radius: 8px;
    padding: 5px 14px;
    color: #334155;
}
QPushButton:hover:!disabled { background: #e8f1ff; }
QPushButton:pressed { background: #d4e5ff; }
QPushButton:disabled {
    color: #94a3b8;
    border-color: #e2e8f0;
    background: #f8fafc;
}
"""

_PRIMARY_BUTTON_QSS = """
QPushButton {
    background: #4f8cff;
    border: 1px solid #4f8cff;
    border-radius: 8px;
    padding: 5px 14px;
    color: #ffffff;
    font-weight: 600;
}
QPushButton:hover:!disabled { background: #3a7df2; border-color: #3a7df2; }
QPushButton:pressed { background: #2f5fd0; }
QPushButton:disabled {
    background: #bcd6f9;
    border-color: #bcd6f9;
    color: #e8f1ff;
}
"""

# Restore signs: borderless, background matches the window background, and
# the chevron is drawn in the palette's window-text color (white in dark
# mode, black in light) — a QPushButton stylesheet would NOT apply anyway
# (QSS selectors match their own class only), so this targets QToolButton.
_TOOL_BUTTON_QSS = """
QToolButton {
    background: palette(window);
    border: none;
    border-radius: 8px;
    padding: 5px 10px;
}
QToolButton:hover:!disabled { background: palette(alternate-base); }
QToolButton:pressed { background: palette(mid); }
"""

_SPEAKER_COLORS = {"self": "#4f8cff", "other": "#35c48a"}
_UNCERTAIN_COLOR = "#8b91a0"
_TRANSLATION_COLOR = "#b8c4d9"
_WINDOW_COLOR = "#ffd479"

# Finder-style rounded panes: the transcript and detail panes are theme-
# following cards (palette colors — black in dark mode, white in light) with
# a soft border; the session list matches with rounded selection rows (flush
# rows — spacing 0 — but the highlight itself rounds to the pane corners,
# since Qt list items otherwise paint square highlights over them).
# Splitter handles match the window margin color (not the style's default
# separator color), so the lines between panes blend with the margin.
_SPLITTER_QSS = "QSplitter::handle { background: palette(window); }"

_PANEL_QSS = """
QTextEdit {
    background: palette(base);
    border: 1px solid palette(midlight);
    border-radius: 8px;
}
"""

_SESSION_LIST_QSS = """
QListWidget {
    background: palette(base);
    border: 1px solid palette(midlight);
    border-radius: 8px;
}
QListWidget::item {
    border-radius: 8px;
    padding: 2px 6px;
}
QListWidget::item:hover:!selected {
    background: palette(alternate-base);
}
QListWidget::item:selected {
    background: #4f8cff;
    color: #ffffff;
}
QListWidget::item:selected:!active {
    background: #cfe0ff;
    color: #334155;
}
"""

_STYLED_RE = re.compile(r"\x1b\[38;2;(\d+);(\d+);(\d+)m([^\x1b]*)\x1b\[0m")

# --- pane rich text → engine ANSI (edit round-trip) -------------------------
# The pane renders a saved session's `.styled` twin (ANSI confidence colors)
# as Qt rich text. Qt preserves per-word character formatting as the user
# types, so converting `toHtml()` back to the engine's ANSI format keeps the
# model's color coding through hand edits. UI meta colors (timestamps,
# speaker tags, translations, the window line) drop to plain — the styled
# twin only carries confidence colors.

_HTML_SPAN_RE = re.compile(r"<span([^>]*)>(.*?)</span>", flags=re.DOTALL)
_HTML_COLOR_RE = re.compile(
    r"color:\s*(?:rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)|#([0-9a-fA-F]{6}))"
)


def _edited_word_indices(original: str, current: str) -> set[int]:
    """Word indices (into `current`, whitespace-split) that were changed or
    added relative to `original`. Used to strip the model's confidence colors
    from hand-edited words — they render plain (white) instead of inheriting
    the neighboring word's color."""
    a = original.split()
    b = current.split()
    edited: set[int] = set()
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
        a=a, b=b, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            edited.update(range(j1, j2))
    return edited


def _html_to_styled(html: str, mode: str, original: str | None) -> str:
    """Qt `toHtml()` → the engine's ANSI styled format. Listen: one line per
    block, translation lines re-indented to the engine's `    → ` convention
    (the pane renders them with a margin instead of spaces). When `original`
    (the text as loaded) is given, words that were edited or added lose their
    color — they render white instead of inheriting the neighbor's."""
    meta_colors = {
        _UNCERTAIN_COLOR.lower(),
        _TRANSLATION_COLOR.lower(),
        _WINDOW_COLOR.lower(),
        *(c.lower() for c in _SPEAKER_COLORS.values()),
    }
    parsed: list[list[tuple[str, str | None]]] = []
    for block in re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.DOTALL):
        parts: list[tuple[str, str | None]] = []
        last = 0
        for m in _HTML_SPAN_RE.finditer(block):
            if m.start() > last:
                parts.append((unescape(block[last : m.start()]), None))
            cm = _HTML_COLOR_RE.search(m.group(1))
            text = unescape(re.sub(r"<[^>]+>", "", m.group(2)))
            color = None
            if cm is not None:
                if cm.group(4) is not None:
                    color = "#" + cm.group(4).lower()
                else:
                    color = (
                        f"#{int(cm.group(1)):02x}{int(cm.group(2)):02x}"
                        f"{int(cm.group(3)):02x}"
                    )
                if color in meta_colors:
                    color = None
            parts.append((text, color))
            last = m.end()
        if last < len(block):
            parts.append((unescape(block[last:]), None))
        parsed.append(parts)
    current_text = "\n".join("".join(t for t, _ in p) for p in parsed)
    edited = (
        _edited_word_indices(original, current_text) if original is not None else set()
    )
    lines: list[str] = []
    word_index = 0
    for parts in parsed:
        text = "".join(t for t, _ in parts)
        # Word-start offsets within this block; `word_index` counts them
        # globally (matching `current_text.split()`), so the diff's indices
        # map onto runs across blocks.
        starts: list[int] = []
        in_word = False
        for i, ch in enumerate(text):
            if ch.isspace():
                in_word = False
            elif not in_word:
                in_word = True
                starts.append(i)
        out: list[str] = []
        offset = 0
        for t, color in parts:
            run_start, run_end = offset, offset + len(t)
            offset = run_end
            count = sum(1 for p in starts if run_start <= p < run_end)
            if color is not None and not any(
                word_index + k in edited for k in range(count)
            ):
                r, g, b = (
                    int(color[1:3], 16),
                    int(color[3:5], 16),
                    int(color[5:7], 16),
                )
                out.append(f"\x1b[38;2;{r};{g};{b}m{t}\x1b[0m")
            else:
                out.append(t)
            word_index += count
        line = "".join(out)
        if mode != "dictate" and line.startswith("→ "):
            line = "    " + line
        lines.append(line)
    return "\n".join(lines)


def _indent_translation_lines(text: str) -> str:
    """The pane renders translations with a margin (no leading spaces); the
    engine's `.txt` convention indents them four spaces."""
    return "\n".join(
        ("    " + line) if line.startswith("→ ") else line for line in text.splitlines()
    )


# --- trash (recoverable deletion) ----------------------------------------
# Move files to the operating system's trash. macOS: AppleScript `Finder`
# delete via `osascript` (paths passed as argv, so quoting is safe). If that
# fails — e.g. macOS automation (TCC) permission for osascript→Finder was
# denied, or Finder is slow to respond — fall back to a plain move into
# `~/.Trash` (Finder still shows the files in the Trash; the "Put Back" menu
# item just isn't available for them). Linux/other: the freedesktop.org trash
# spec (`$XDG_DATA_HOME/Trash/files` + `.trashinfo` metadata; a cross-device
# move falls back to copy+delete via shutil). There is no stdlib trash API
# and this deliberately avoids a new dependency (send2trash would be the
# packaged option). Failure is reported, never silent: the caller decides
# what to do (the GUI surfaces "delete failed").

_OSASCRIPT = """
on run argv
    set filesToDelete to {}
    repeat with p in argv
        set end of filesToDelete to (POSIX file p as alias)
    end repeat
    tell application "Finder"
        delete filesToDelete
    end tell
end run
"""


def _trash_macos(paths: list[Path]) -> bool:
    try:
        result = subprocess.run(
            ["osascript", "-e", _OSASCRIPT, "--", *(str(p) for p in paths)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError, subprocess.TimeoutExpired:
        result = None
    if result is not None and result.returncode == 0:
        return True
    return _trash_dottrash(paths)


def _trash_dottrash(paths: list[Path]) -> bool:
    """Fallback: move into the user's ~/.Trash folder directly (no Finder /
    automation permission needed)."""
    trash_dir = Path.home() / ".Trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    moved = False
    for path in paths:
        if not path.exists():
            continue
        target = trash_dir / path.name
        n = 1
        while target.exists():
            target = trash_dir / f"{path.name}.{n}"
            n += 1
        try:
            shutil.move(str(path), str(target))
            moved = True
        except OSError:
            continue
    return moved


def _trash_freedesktop(paths: list[Path]) -> bool:
    trash = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        / "Trash"
    )
    files_dir = trash / "files"
    info_dir = trash / "info"
    moved = False
    for path in paths:
        if not path.exists():
            continue
        target = files_dir / path.name
        info = info_dir / f"{path.name}.trashinfo"
        n = 1
        while target.exists() or info.exists():
            target = files_dir / f"{path.name}.{n}"
            info = info_dir / f"{path.name}.{n}.trashinfo"
            n += 1
        files_dir.mkdir(parents=True, exist_ok=True)
        info_dir.mkdir(parents=True, exist_ok=True)
        info.write_text(
            f"[Trash Info]\nPath={quote(str(path.absolute()), safe='')}\n"
            f"DeletionDate={datetime.datetime.now(tz=UTC).isoformat()}\n",
            encoding="utf-8",
        )
        shutil.move(str(path), str(target))
        moved = True
    return moved


def move_to_trash(paths: list[Path]) -> bool:
    """Move every existing path in `paths` to the system trash. Returns True
    if at least one file was trashed."""
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return False
    if sys.platform == "darwin":
        return _trash_macos(existing)
    return _trash_freedesktop(existing)


def _styled_to_html(styled: str) -> str:
    """Convert the engine's per-word ANSI confidence colors into Qt-rich-text
    spans (the `styled` twin of each chunk)."""
    out = ""
    last = 0
    for m in _STYLED_RE.finditer(styled):
        out += html.escape(styled[last : m.start()].replace("\x1b[0m", ""))
        r, g, b = (min(255, int(m.group(i))) for i in (1, 2, 3))
        out += (
            f'<span style="color: rgb({r},{g},{b});">{html.escape(m.group(4))}</span>'
        )
        last = m.end()
    out += html.escape(styled[last:].replace("\x1b[0m", ""))
    return out


def _meta_html(ts: str | None, speaker: str | None) -> str:
    out = f'<span style="color: {_UNCERTAIN_COLOR};">[{html.escape(ts or "")}]</span>'
    if speaker:
        color = _SPEAKER_COLORS.get(speaker, _UNCERTAIN_COLOR)
        out += (
            f' <span style="color: {color}; font-weight: 600;">'
            f"[{html.escape(speaker)}]</span>"
        )
    return out


def _join_prefix_text(prefix: str, text: str) -> str:
    """Join a resumed session's old transcript (plain) with the new flowing
    text, keeping them as one natural line."""
    if prefix and text:
        return f"{prefix} {text}"
    return prefix or text


def _chevron_icon(direction: str, color: str) -> QIcon:
    """Paint a chevron arrowhead (two strokes) pointing in the given
    direction. Used by the sidebar restore sign (left) and the transcript
    pane's restore sign (up). The color follows the theme's window-text
    color."""
    pm = QPixmap(16, 16)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(2.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    if direction == "left":
        painter.drawLine(4, 8, 12, 3)
        painter.drawLine(4, 8, 12, 13)
    elif direction == "up":
        painter.drawLine(8, 3, 3, 11)
        painter.drawLine(8, 3, 13, 11)
    painter.end()
    return QIcon(pm)


def _copy_icon(color: str) -> QIcon:
    """Paint a copy glyph — two rounded-rect sheets, back sheet offset
    up-left — in the same drawn style as the chevron restore signs."""
    pm = QPixmap(16, 16)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.4)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(4.5, 2.5, 7, 9.5), 1.6, 1.6)
    painter.drawRoundedRect(QRectF(6.5, 4.5, 7, 9.5), 1.6, 1.6)
    painter.end()
    return QIcon(pm)


def _snapshot_html(snap: TranscriptSnapshot) -> str:
    """Listen rendering: committed chunks with per-word confidence colors,
    speaker tags and translations, then the live window partial at the bottom
    (newest last, like the CLI). Every line is its own block with a fixed
    4 px gap before chunks and a fixed 30 px translation indent — uniform
    spacing whether or not a chunk has a translation (the old `<br>`-joined
    mix of runs and divs spaced inconsistently)."""
    parts: list[str] = []
    for c in snap.chunks:
        text = _styled_to_html(c.styled) or html.escape(c.plain)
        parts.append(
            f'<div style="margin-top: 4px;">{_meta_html(c.ts, c.speaker)} {text}</div>'
        )
        if c.translation:
            parts.append(
                f'<div style="margin-left: 30px; color: {_TRANSLATION_COLOR};">'
                f"→ {html.escape(_strip_timing(c.translation))}</div>"
            )
    if snap.window is not None:
        text = _styled_to_html(snap.window.styled) or html.escape(snap.window.plain)
        parts.append(
            f'<div style="margin-top: 4px; color: {_WINDOW_COLOR};">'
            f"{_meta_html(snap.window.ts, snap.window.speaker)} {text}</div>"
        )
        if snap.window.translation:
            parts.append(
                f'<div style="margin-left: 30px; color: {_TRANSLATION_COLOR};">'
                f"→ {html.escape(_strip_timing(snap.window.translation))}</div>"
            )
    return "".join(parts)


# One line of the saved listen transcript: `[ts] [speaker] styled-text`.
_STYLED_LINE_RE = re.compile(r"^\[([^\]]+)\](?: \[([^\]]+)\])? (.*)$")


def _session_styled_html(mode: str, styled: str) -> str:
    """Re-render a saved session's `.styled` twin (ANSI confidence colors
    preserved) the same way the live renderer draws it: dictation renders as
    one flowing colored block; listen keeps the per-line timestamp/speaker
    meta and translations."""
    if mode == "dictate":
        return _styled_to_html(styled)
    parts: list[str] = []
    for line in styled.splitlines():
        if line.startswith("    → "):
            parts.append(
                f'<div style="margin-left: 30px; color: {_TRANSLATION_COLOR};">'
                f"→ {html.escape(line[6:])}</div>"
            )
            continue
        m = _STYLED_LINE_RE.match(line)
        if m is None:
            parts.append(f"<div>{html.escape(line)}</div>")
            continue
        ts, speaker, rest = m.groups()
        parts.append(
            f'<div style="margin-top: 4px;">{_meta_html(ts, speaker)} '
            f"{_styled_to_html(rest)}</div>"
        )
    return "".join(parts)


class SnapshotStore(BaseRenderer):
    """Renderer that keeps the most recent snapshot (+ version counter) for
    the UI poll timer. Called from engine worker threads; `pull()` runs on the
    Qt main thread — both sides take the lock."""

    def __init__(self) -> None:
        self._latest: TranscriptSnapshot | None = None
        self._version = 0
        self._lock = threading.Lock()

    def render(self, snapshot: TranscriptSnapshot) -> None:
        with self._lock:
            self._latest = snapshot
            self._version += 1

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._version = 0

    def pull(self) -> tuple[TranscriptSnapshot | None, int]:
        with self._lock:
            return self._latest, self._version


class SessionController:
    """Owns engine threads, snapshots and session history. All heavy work
    (model loading, the engine) runs in background threads so the UI never
    blocks."""

    def __init__(self) -> None:
        self._store = SnapshotStore()
        self._engine: RealTimeTranscribe | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._status = "idle"
        self._status_lock = threading.Lock()
        self._session: tuple[str, str] | None = None
        self._models: tuple[str, ...] = ()

    def _set_status(self, status: str) -> None:
        with self._status_lock:
            self._status = status

    def _status_value(self) -> str:
        with self._status_lock:
            return self._status

    # --- live sessions ---------------------------------------------------

    def start_listen(self, translate: bool, resume_from=None) -> None:
        self._launch(
            stt=STT_MODEL_EN_ONLY,
            translate=bool(translate),
            mode="listen",
            speaker_id=True,
            clean=False,
            resume_from=resume_from,
        )

    def start_dictate(self, en: bool, resume_from=None) -> None:
        self._launch(
            stt=STT_MODEL_EN_ONLY if en else STT_MODEL_MIXED,
            translate=False,
            mode="dictate",
            speaker_id=False,
            clean=True,
            resume_from=resume_from,
        )

    def _launch(self, **kwargs) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                # A session is still winding down (the old join here — up to
                # 5 s on the GUI thread — contributed to the Stop hang; the
                # poll only re-enables Start once the thread's own status
                # reads "stopped", so this refusal is rarely reached).
                return
            self._thread = None
        self._stop_requested = False
        self._session = None
        self._models = ()
        self._store.clear()
        # Flip the status synchronously: the new engine thread only reaches
        # "loading" a few microseconds after start(), so without this a waiter
        # polling right after launch reads the PREVIOUS session's stale
        # "stopped" and thinks the new session already ended.
        self._set_status("starting")
        self._thread = threading.Thread(
            target=self._run_engine, kwargs=kwargs, daemon=True
        )
        self._thread.start()

    def _run_engine(
        self,
        *,
        stt,
        translate,
        mode,
        speaker_id,
        clean,
        input_path=None,
        resume_from=None,
    ) -> None:
        mode_dir = DATA_DIR / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        # Resume: keep the selected session's files; a stale/mismatched path
        # silently falls back to a fresh session (the GUI validates first).
        resume = None
        if resume_from is not None:
            path = Path(resume_from)
            if path.parent == mode_dir and path.is_file():
                resume = path
        if resume is not None:
            audio_file = resume
        else:
            timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            audio_file = mode_dir / f"{mode}_{timestamp}.flac"
        self._session = (mode, audio_file.stem)
        self._set_status("loading")
        models = [f"stt: {stt}"]
        if translate:
            models.append(f"translate: {TRANSLATE_MODEL}")
        if speaker_id:
            models.append(f"speaker id: {EN_MODEL.stem}")
        self._models = tuple(models)
        try:
            rtt = RealTimeTranscribe(
                audio_file_path=audio_file,
                stt_model=stt,
                translate_model=TRANSLATE_MODEL,
                translate_to="Chinese" if translate else None,
                clean=clean,
                speaker_id=speaker_id,
                input_path=input_path,
                resume_from=resume,
                renderer=self._store,
                quiet=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self._set_status(f"error: {exc}")
            return
        if self._stop_requested:
            self._stop_requested = False
            self._set_status("stopped")
            return
        self._engine = rtt
        self._set_status("listening")
        try:
            rtt.run()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            if isinstance(exc, sd.PortAudioError):
                self._set_status(
                    "error: mic unavailable — check microphone permission "
                    "(System Settings → Privacy & Security) and that no other "
                    "app is using it"
                )
            else:
                self._set_status(f"error: {exc}")
        else:
            self._set_status("stopped")

    def stop(self) -> None:
        """Signal the running engine to stop; run() finalizes and writes the
        session files on its own (no callback can race the file write). NON-
        BLOCKING: joining the engine thread here hung the UI on Stop while
        models were still loading (the join waited out the whole load, and
        `_run_engine` only checks `_stop_requested` after construction). The
        poll picks up the thread's own `stopped` status and refreshes."""
        if self._engine is not None:
            self._engine.stop()
        elif self._thread is not None and self._thread.is_alive():
            self._stop_requested = True

    def shutdown(self) -> None:
        """Window-close hook: stop the engine BEFORE the window tears down,
        so portaudio releases the mic cleanly (a stream killed mid-teardown
        by process exit spams PaMacCore err=-50 to the terminal and can leave
        the mic busy for the next launch)."""
        if self._engine is not None:
            self._engine.stop()
        elif self._thread is not None and self._thread.is_alive():
            self._stop_requested = True

    def pull(self) -> dict:
        snapshot, version = self._store.pull()
        return {
            "status": self._status_value(),
            "snapshot": (snapshot, version) if snapshot is not None else None,
            "session": (
                {"mode": self._session[0], "name": self._session[1]}
                if self._session is not None
                else None
            ),
            "models": self._models,
        }

    # --- session history ---------------------------------------------------

    def list_sessions(self) -> list[dict]:
        items = []
        for mode in _MODE_DIRS:
            mode_dir = DATA_DIR / mode
            if not mode_dir.is_dir():
                continue
            for audio in sorted(mode_dir.glob("*.flac"), reverse=True):
                items.append(
                    {
                        "mode": mode,
                        "name": audio.stem,
                        "has_text": audio.with_suffix(".txt").exists(),
                    }
                )
        return items

    def session_text(self, name: str, mode: str) -> str:
        txt = DATA_DIR / mode / f"{name}.txt"
        if not txt.is_file():
            return ""
        return txt.read_text(encoding="utf-8")

    def session_styled(self, name: str, mode: str) -> str:
        """The session's `.styled` twin: the transcript with its per-word ANSI
        confidence colors preserved, so the GUI can re-render it with the
        original color coding. Empty when the session predates the twin."""
        styled = DATA_DIR / mode / f"{name}.styled"
        if not styled.is_file():
            return ""
        return styled.read_text(encoding="utf-8")

    def rename_session(self, mode: str, old: str, new: str) -> bool:
        """Rename a session's FLAC/.txt/.styled files (the sidebar entry
        renames with them). Guards like `delete_sessions`; refuses when the
        target name already exists (collision) or no files are present."""
        if mode not in _MODE_DIRS:
            return False
        for name in (old, new):
            if Path(name).name != name:
                return False
        if old == new:
            return False
        srcs = [DATA_DIR / mode / f"{old}{s}" for s in (".flac", ".txt", ".styled")]
        existing = [p for p in srcs if p.is_file()]
        if not existing:
            return False
        for src in srcs:
            dst = DATA_DIR / mode / f"{new}{src.suffix}"
            if dst.is_file():
                return False
        for src in existing:
            src.rename(DATA_DIR / mode / f"{new}{src.suffix}")
        return True

    def save_transcript(
        self, mode: str, name: str, text: str, styled: str | None = None
    ) -> bool:
        """Write hand-edited transcript text back to a session's `.txt` (guarded
        against bad modes/paths like `delete_sessions`). When `styled` is given
        (the pane's rich text converted back to the engine's ANSI format), the
        `.styled` twin is rewritten with it — per-word confidence colors survive
        edits; otherwise the twin is left untouched, never clobbered."""
        if mode not in _MODE_DIRS or Path(name).name != name:
            return False
        txt = DATA_DIR / mode / f"{name}.txt"
        if not txt.is_file():
            return False
        txt.write_text(text, encoding="utf-8")
        if styled is not None:
            styled_path = DATA_DIR / mode / f"{name}.styled"
            styled_path.write_text(styled, encoding="utf-8")
        return True

    def delete_sessions(self, items: list[tuple[str, str]]) -> int:
        """Move several (mode, name) sessions to the system trash in ONE
        batch: each `osascript`→Finder round-trip costs ~1 s on macOS, so
        per-session calls made multi-deletes crawl. Returns how many sessions
        are actually gone (files no longer present)."""
        paths: list[Path] = []
        valid: list[tuple[str, str]] = []
        for mode, name in items:
            if mode not in _MODE_DIRS or Path(name).name != name:
                continue
            valid.append((mode, name))
            paths.extend(
                DATA_DIR / mode / f"{name}{suffix}"
                for suffix in (".flac", ".txt", ".styled")
            )
        if not valid:
            return 0
        if not move_to_trash(paths):
            return 0
        deleted = 0
        for mode, name in valid:
            remaining = [
                DATA_DIR / mode / f"{name}{suffix}"
                for suffix in (".flac", ".txt", ".styled")
            ]
            if not any(p.exists() for p in remaining):
                deleted += 1
        return deleted


class TranscriptPane(QTextEdit):
    """QTextEdit with two floating sign buttons pinned to its top-right
    corner: the copy button (always visible) and the sidebar-restore button
    (only visible while the sessions sidebar is collapsed, so the pane's
    right edge is the window's right edge then — dragging the splitter handle
    back is possible but undiscoverable, so the button is the visible "sign"
    that the sidebar can be brought back)."""

    def __init__(self) -> None:
        super().__init__()
        self._expand_btn: QToolButton | None = None
        self._copy_btn: QToolButton | None = None

    def set_expand_button(self, btn: QToolButton) -> None:
        self._expand_btn = btn
        btn.setParent(self)
        self._place_button()

    def set_copy_button(self, btn: QToolButton) -> None:
        self._copy_btn = btn
        btn.setParent(self)
        self._place_button()

    def _place_button(self) -> None:
        if self._copy_btn is not None:
            self._copy_btn.move(self.width() - self._copy_btn.width() - 8, 8)
        if self._expand_btn is not None:
            # The restore sign hangs at the middle of the right edge (it only
            # shows while the sidebar is collapsed) — away from the copy
            # button pinned to the corner.
            self._expand_btn.move(
                self.width() - self._expand_btn.width() - 8,
                (self.height() - self._expand_btn.height()) // 2,
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_button()


class SessionList(QListWidget):
    """Session list that exposes sessions as file URLs in its drag mime data,
    so dragging an item (or several) to Finder copies the FLAC + transcript."""

    def mimeData(self, items: Sequence[QListWidgetItem]) -> QMimeData:
        mime = super().mimeData(items)
        urls: list[QUrl] = []
        for item in items:
            mode, name = item.data(Qt.ItemDataRole.UserRole)
            for suffix in (".flac", ".txt"):
                f = DATA_DIR / mode / f"{name}{suffix}"
                if f.is_file():
                    urls.append(QUrl.fromLocalFile(str(f)))
        if urls:
            mime.setUrls(urls)
        return mime


class MainWindow(QMainWindow):
    def __init__(self, controller: SessionController) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Interpreter")
        self.resize(1000, 700)
        self.setMinimumSize(640, 480)

        self.last_version = -1
        self.running = False
        self.dictating = False
        self._deleted: int | None = None
        # Resumed session: the previous transcript rendered as a read-only
        # prefix above the live content (prepended on every re-render, which
        # replaces the whole document).
        self._resume_prefix = ""
        self._resume_prefix_plain = ""
        # The session whose transcript the pane currently shows (None while a
        # live session owns the pane) — autosave targets this. `_active_loaded`
        # is the plain text as loaded from disk: a flush only writes when the
        # pane text differs (a no-edit flush would rewrite the file in the
        # pane's normalized form and strip the `.styled` twin's colors).
        self._active_session: tuple[str, str] | None = None
        self._active_loaded: str | None = None
        self._msg: str | None = None
        self._msg_ticks = 0
        self._error_rendered = False

        # --- header: mode + options + actions --------------------------------
        self.mode_listen = QPushButton("Listen")
        self.mode_listen.setCheckable(True)
        self.mode_listen.setChecked(True)
        self.mode_listen.setObjectName("mode_listen")
        self.mode_dictate = QPushButton("Dictate")
        self.mode_dictate.setCheckable(True)
        self.mode_dictate.setObjectName("mode_dictate")
        # Pin the pill buttons to their sizeHint: QPushButton's default
        # horizontal policy is Minimum (growable), so on some styles the
        # pill stretches when the window is resized. Fixed keeps the pill
        # constant while the font size is unchanged.
        for b in (self.mode_listen, self.mode_dictate):
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.mode_listen)
        mode_group.addButton(self.mode_dictate)
        self.mode_switch = QWidget()
        self.mode_switch.setObjectName("mode_switch")
        mode_layout = QHBoxLayout(self.mode_switch)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        mode_layout.addWidget(self.mode_listen)
        mode_layout.addWidget(self.mode_dictate)
        self.mode_switch.setStyleSheet(
            """
            QWidget#mode_switch {
                background: #eef2f7;
                border-radius: 9px;
            }
            QPushButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 9px;
                color: #64748b;
                padding: 5px 18px;
            }
            QPushButton:hover:!checked {
                background: rgba(79, 140, 255, 0.10);
            }
            QPushButton:checked {
                background: #4f8cff;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:disabled {
                color: #94a3b8;
            }
            """
        )
        self.translate_cb = QCheckBox("Translate (en→zh)")
        self.translate_cb.setChecked(True)
        self.en_only_cb = QCheckBox("English only")
        # Fixed-width container: only one checkbox is visible at a time, but
        # the container always occupies the same header slot, so switching
        # modes never reflows the header (rapid toggling used to repaint the
        # whole row, which made the font controls flicker).
        self.option_box = QWidget()
        option_layout = QHBoxLayout(self.option_box)
        option_layout.setContentsMargins(0, 0, 0, 0)
        option_layout.setSpacing(0)
        option_layout.addWidget(self.translate_cb)
        option_layout.addWidget(self.en_only_cb)
        self._match_option_widths()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.status_lbl = QLabel("idle")
        self.status_lbl.setStyleSheet("color: #b8c4d9;")
        self.status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.font_lbl = QLabel("A")
        self.font_lbl.setStyleSheet("color: #64748b;")
        self.font_spin = QSpinBox()
        self.font_spin.setRange(*_FONT_RANGE)
        self.font_spin.setValue(_saved_font_size())
        self.font_spin.setSuffix(" pt")
        self.font_spin.setToolTip("App font size (persisted)")

        header = QHBoxLayout()
        # AlignVCenter: without it the header stretches the switch to the row
        # height while the buttons keep their own height — the blue pill
        # stopped short of the track.
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(self.mode_switch, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addSpacing(4)
        header.addWidget(self.option_box)
        header.addSpacing(4)
        header.addWidget(self.start_btn)
        header.addWidget(self.stop_btn)
        header.addSpacing(4)
        header.addWidget(self.font_lbl)
        header.addWidget(self.font_spin)
        # Stretch 1: the status label absorbs the header's slack and draws its
        # text right-aligned, elided to its width — a constant right-hand
        # slot that never reflows the header (see _poll).
        header.addWidget(self.status_lbl, 1, Qt.AlignmentFlag.AlignVCenter)

        # --- transcript ------------------------------------------------------
        self.transcript = TranscriptPane()
        self.transcript.setReadOnly(True)
        self.transcript.setStyleSheet(_PANEL_QSS)
        transcript_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        transcript_font.setPointSize(15)
        self.transcript.setFont(transcript_font)
        self.transcript.document().setDocumentMargin(4)
        self.transcript.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.transcript.setPlaceholderText("Press Start, then begin speaking…")

        # --- sessions sidebar -------------------------------------------------
        self.session_list = SessionList()
        self.session_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.session_list.setDragEnabled(True)
        self.session_list.setSpacing(1)
        self.session_list.setStyleSheet(_SESSION_LIST_QSS)
        self.session_list.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.session_list.setToolTip(
            "Double-click a session name to rename it (its files are renamed)"
        )
        self.reveal_btn = QPushButton("Reveal")
        self.reveal_btn.setEnabled(False)
        self.reveal_btn.setToolTip("Reveal in Finder (open the folder with the files)")
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setEnabled(False)
        self.resume_btn.setToolTip(
            "Record into the selected session — new audio and transcript "
            "lines append to its files"
        )
        for b in (
            self.stop_btn,
            self.reveal_btn,
            self.delete_btn,
            self.resume_btn,
        ):
            b.setStyleSheet(_ACTION_BUTTON_QSS)
        self.start_btn.setStyleSheet(_PRIMARY_BUTTON_QSS)

        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 0, 0)
        sidebar.setSpacing(4)
        sidebar.addWidget(QLabel("Sessions"))
        sidebar.addWidget(self.session_list, 1)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(4)
        buttons.addWidget(self.resume_btn)
        buttons.addWidget(self.reveal_btn)
        buttons.addWidget(self.delete_btn)
        sidebar.addLayout(buttons)
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMinimumWidth(240)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.transcript)
        splitter.addWidget(sidebar_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(_SPLITTER_QSS)
        self.splitter = splitter
        # Sidebar-restore affordance: the sidebar can be dragged shut, but a
        # 1 px handle at the window's right edge is undiscoverable — show a
        # left-pointing chevron button on the transcript's corner while it is
        # collapsed (same drawn chevron as the pane button, rotated).
        self._sidebar_width: int | None = None
        arrow_color = self.palette().color(QPalette.ColorRole.WindowText).name()
        self.expand_btn = QToolButton()
        self.expand_btn.setIcon(_chevron_icon("left", arrow_color))
        self.expand_btn.setIconSize(QSize(16, 16))
        self.expand_btn.setToolTip("Show sessions")
        self.expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_btn.setStyleSheet(_TOOL_BUTTON_QSS)
        self.expand_btn.adjustSize()
        self.transcript.set_expand_button(self.expand_btn)
        self.expand_btn.hide()
        self.copy_btn = QToolButton()
        self.copy_btn.setIcon(_copy_icon(arrow_color))
        self.copy_btn.setIconSize(QSize(16, 16))
        self.copy_btn.setToolTip("Copy transcript to clipboard")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.setStyleSheet(_TOOL_BUTTON_QSS)
        self.copy_btn.adjustSize()
        self.transcript.set_copy_button(self.copy_btn)
        self.expand_btn.clicked.connect(self._expand_sidebar)
        splitter.splitterMoved.connect(self._splitter_moved)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        main_layout = self.layout()
        if main_layout is not None:
            main_layout.setContentsMargins(0, 0, 0, 0)

        # --- wiring -----------------------------------------------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(150)

        self.mode_listen.toggled.connect(self._mode_changed)
        self.start_btn.clicked.connect(self._start)
        self.resume_btn.clicked.connect(lambda: self._start(resume=True))
        self.stop_btn.clicked.connect(self._stop)
        self.session_list.itemChanged.connect(self._session_renamed)
        # Debounced autosave: edits to a saved session's transcript write back
        # to its .txt ~0.8 s after the last keystroke (no Save button).
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self._flush_autosave)
        self.transcript.textChanged.connect(self._schedule_autosave)
        self.font_spin.valueChanged.connect(self._font_size_changed)
        self.reveal_btn.clicked.connect(self._reveal)
        self.delete_btn.clicked.connect(self._delete)
        self.session_list.currentItemChanged.connect(self._session_selected)
        self.session_list.itemSelectionChanged.connect(self._selection_changed)
        self.copy_btn.clicked.connect(self._copy_transcript)

        self._mode_changed()
        self._refresh_sessions()

    # --- actions ---------------------------------------------------------

    def _mode_changed(self) -> None:
        self._flush_autosave()
        listen = self.mode_listen.isChecked()
        self.translate_cb.setVisible(listen)
        self.en_only_cb.setVisible(not listen)
        if not self.running:
            # The window binds to the selected mode once no session is
            # active: reset the transcript to the mode's idle state and bind
            # the session list to the mode's recordings (each mode runs its
            # own models, so listen/dictate sessions are separate worlds).
            # Pin last_version to the store's CURRENT version — resetting it
            # to -1 made the next poll re-render the previous session's
            # snapshot and flick the transcript back to the old mode's
            # content. A new session resets it in _start.
            snap = self.controller.pull()["snapshot"]
            self.last_version = snap[1] if snap is not None else -1
            self._error_rendered = False
            self._resume_prefix = ""
            self._resume_prefix_plain = ""
            self._refresh_sessions()
            # Bind the left transcript window to the mode: select the newest
            # recording of the active mode — the selection handler loads it
            # into the transcript — or show the mode's idle hint when the
            # mode has no recordings yet.
            mode = "dictate" if self.mode_dictate.isChecked() else "listen"
            newest = next(
                (s for s in self.controller.list_sessions() if s["mode"] == mode),
                None,
            )
            if newest is None:
                self.transcript.clear()
                self.transcript.setPlaceholderText(self._idle_hint())
                return
            for i in range(self.session_list.count()):
                item = self.session_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == (mode, newest["name"]):
                    self._select_only(item)
                    break

    def _idle_hint(self) -> str:
        return (
            "Press Start, then begin dictating…"
            if self.mode_dictate.isChecked()
            else "Press Start, then begin speaking…"
        )

    def _splitter_moved(self, pos: int, _index: int) -> None:
        """Track the sidebar width; when it is dragged shut, surface the
        restore button (a "sign" that the sessions pane can come back)."""
        width = self.splitter.width() - pos - self.splitter.handleWidth()
        if width >= 40:
            self._sidebar_width = width
            self.expand_btn.hide()
        else:
            self.expand_btn.show()
            self.transcript._place_button()

    def _expand_sidebar(self) -> None:
        width = self._sidebar_width or 240
        width = min(width, self.splitter.width() - 80)
        if width < 40:
            width = 240
        total = sum(self.splitter.sizes())
        self.splitter.setSizes([max(0, total - width), width])
        self.expand_btn.hide()

    def _set_running(self, running: bool) -> None:
        if self.running == running:
            return
        self.running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.mode_switch, self.translate_cb, self.en_only_cb):
            w.setEnabled(not running)
        # Editing is only ever for SAVED sessions — the live transcript is a
        # read-only view (typing into it would fight the snapshot re-render).
        self.transcript.setReadOnly(running)
        # Renaming a session mid-recording would move the engine's files out
        # from under it — names are locked while a session runs.
        self.session_list.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
            if running
            else QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._selection_changed()

    def _font_size_changed(self, size: int) -> None:
        QSettings().setValue("fontSize", size)
        delta = size - _BASE_FONT_PT
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return
        font = app.font()
        font.setPointSize(size)
        app.setFont(font)
        # Any widget with a stylesheet (buttons, status label) does NOT follow
        # QApplication.setFont in Qt 6 — their fonts must be set explicitly.
        for widget in (
            self.mode_listen,
            self.mode_dictate,
            self.start_btn,
            self.stop_btn,
            self.reveal_btn,
            self.delete_btn,
            self.resume_btn,
            self.status_lbl,
            self.session_list,
        ):
            widget.setFont(font)
        for widget, base in ((self.transcript, _TRANSCRIPT_FONT_PT),):
            f = widget.font()
            f.setPointSize(base + delta)
            widget.setFont(f)
        self._match_option_widths()

    def _match_option_widths(self) -> None:
        """The mode option label keeps the width of the wider one, so the
        options container holds a constant header slot — switching modes
        never shifts the Start/Stop buttons or reflows the header."""
        width = max(
            self.translate_cb.sizeHint().width(),
            self.en_only_cb.sizeHint().width(),
        )
        for cb in (self.translate_cb, self.en_only_cb):
            cb.setMinimumWidth(width)
        self.option_box.setFixedWidth(width)

    def _start(self, resume: bool = False) -> None:
        self._flush_autosave()
        self._active_session = None
        self._active_loaded = None
        self._resume_prefix = ""
        self._resume_prefix_plain = ""
        resume_flac: Path | None = None
        if resume:
            item = self.session_list.currentItem()
            if item is not None:
                mode, name = item.data(Qt.ItemDataRole.UserRole)
                flac = DATA_DIR / mode / f"{name}.flac"
                if flac.is_file():
                    resume_flac = flac
                    # The previous transcript stays visible above the live
                    # content while recording (it is the conversation so far).
                    styled = self.controller.session_styled(name, mode)
                    if styled:
                        self._resume_prefix = _session_styled_html(mode, styled)
                        self._resume_prefix_plain = self.controller.session_text(
                            name, mode
                        )
                    else:
                        text = self.controller.session_text(name, mode)
                        self._resume_prefix_plain = text
                        self._resume_prefix = (
                            f"<div>{html.escape(text)}</div>" if text else ""
                        )
        self._set_running(True)
        self.status_lbl.setText("resuming…" if resume else "starting…")
        self.last_version = -1
        self._error_rendered = False
        self.transcript.clear()
        if self._resume_prefix:
            self.transcript.setHtml(self._resume_prefix)
        self.transcript.setPlaceholderText(
            "Loading models… (first run downloads weights). "
        )
        self.dictating = self.mode_dictate.isChecked()
        if self.dictating:
            self.controller.start_dictate(
                self.en_only_cb.isChecked(), resume_from=resume_flac
            )
        else:
            self.controller.start_listen(
                self.translate_cb.isChecked(), resume_from=resume_flac
            )

    def _stop(self) -> None:
        self.status_lbl.setText("stopping…")
        # No session-list refresh here: the engine's files don't exist until
        # finalize, so a rebuild right now would drop the live entry (the
        # sidebar follows it via `_track_live_session`) and the poll's stop
        # refresh rebuilds once more — a visible double flicker. The poll
        # handles the single final refresh when "stopped" lands.
        self.controller.stop()

    def _reveal(self) -> None:
        """Open the session's folder in Finder with the file selected
        (`open -R` — no automation/TCC permission needed, unlike AppleScript
        Finder scripting)."""
        item = self.session_list.currentItem()
        if item is None:
            return
        mode, name = item.data(Qt.ItemDataRole.UserRole)
        target: Path | None = None
        for suffix in (".flac", ".txt"):
            f = DATA_DIR / mode / f"{name}{suffix}"
            if f.is_file():
                target = f
                break
        if target is None:
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent)])

    def _schedule_autosave(self) -> None:
        """Restart the debounce timer on every text change. Only saved
        sessions autosave: while a session runs the pane is read-only (its
        re-renders are programmatic), and typing into the idle hint pane has
        no session to write to."""
        if self.running or self._active_session is None:
            return
        self.autosave_timer.start(800)

    def _flush_autosave(self) -> None:
        """Write the pane's current text back to the active session's `.txt`
        (silent — no Save button). Called by the debounce timer and at every
        point where the pane's content is about to be replaced. Only writes
        when the pane text actually CHANGED since it was loaded."""
        self.autosave_timer.stop()
        if self.running or self._active_session is None:
            return
        text = self.transcript.toPlainText()
        if self._active_loaded is not None and text == self._active_loaded:
            return
        mode, name = self._active_session
        styled = _html_to_styled(self.transcript.toHtml(), mode, self._active_loaded)
        plain = _indent_translation_lines(text) if mode != "dictate" else text
        if self.controller.save_transcript(mode, name, plain, styled=styled):
            self._active_loaded = text

    def _session_renamed(self, item: QListWidgetItem) -> None:
        """In-place rename of a session list item: the underlying FLAC/.txt/
        .styled files are renamed too, so the list entry keeps matching disk.
        Invalid names (empty, path separators, collisions) revert silently."""
        mode, name = item.data(Qt.ItemDataRole.UserRole)
        new = item.text().strip()
        if new == name or not new:
            return
        if "/" in new or "\\" in new or new != Path(new).name:
            self._set_item_text_blocked(item, name)
            return
        if self.running:
            self._set_item_text_blocked(item, name)
            return
        if self.controller.rename_session(mode, name, new):
            item.setData(Qt.ItemDataRole.UserRole, (mode, new))
            if self._active_session == (mode, name):
                self._active_session = (mode, new)
            self._set_message("renamed ✓")
        else:
            self._set_item_text_blocked(item, name)

    @staticmethod
    def _set_item_text_blocked(item: QListWidgetItem, text: str) -> None:
        """Revert an item's text without re-triggering itemChanged."""
        view = item.listWidget()
        if view is not None:
            view.blockSignals(True)
        item.setText(text)
        if view is not None:
            view.blockSignals(False)

    def _copy_transcript(self) -> None:
        """Copy the transcript's current text to the clipboard (quick share
        of a live or saved session)."""
        QApplication.clipboard().setText(self.transcript.toPlainText())
        self._set_message("copied ✓")

    def _set_message(self, text: str) -> None:
        """Transient status label message (shown ~3 s, then the engine status
        takes over again — a plain setText was invisible because the 150 ms
        poll overwrote it with `idle` within one tick)."""
        self._msg = text
        self._msg_ticks = 20

    def _delete(self) -> None:
        items = [
            it.data(Qt.ItemDataRole.UserRole)
            for it in self.session_list.selectedItems()
        ]
        if not items:
            return
        self.delete_btn.setEnabled(False)
        self._set_message(f"deleting {len(items)}…")
        threading.Thread(target=self._delete_worker, args=(items,), daemon=True).start()

    def _delete_worker(self, items: list[tuple[str, str]]) -> None:
        self._deleted = self.controller.delete_sessions(items)

    def _session_selected(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        mode, name = current.data(Qt.ItemDataRole.UserRole)
        if not self.running:
            # Flush pending edits of the session the pane currently shows
            # BEFORE its content is replaced.
            self._flush_autosave()
            self._resume_prefix = ""
            self._resume_prefix_plain = ""
            # The load below must not schedule an autosave of the loaded
            # content (textChanged would fire and re-write the same text).
            self.transcript.blockSignals(True)
            try:
                # Bind the left transcript window to the mode's session too:
                # it shows the selected recording instead of stale content (a
                # live session keeps its own view). The `.styled` twin
                # restores the original confidence color coding; plain text
                # otherwise.
                styled = self.controller.session_styled(name, mode)
                if styled:
                    self.transcript.setHtml(_session_styled_html(mode, styled))
                elif text := self.controller.session_text(name, mode):
                    self.transcript.setPlainText(text)
                else:
                    self.transcript.clear()
                    self.transcript.setPlaceholderText(self._idle_hint())
            finally:
                self.transcript.blockSignals(False)
            self._active_session = (mode, name)
            self._active_loaded = self.transcript.toPlainText()

    def _selection_changed(self) -> None:
        selected = self.session_list.selectedItems()
        self.reveal_btn.setEnabled(bool(selected))
        self.delete_btn.setEnabled(bool(selected))
        # Resume acts on exactly one finished session.
        self.resume_btn.setEnabled(len(selected) == 1 and not self.running)

    def _track_live_session(self, mode: str, name: str) -> None:
        """Add the in-progress session to the sidebar (and select it) as soon
        as the engine names it. The files don't exist until stop, so the item
        is purely presentational while recording; every operation on it
        (reveal/delete/drag-out/resume) already re-checks the files."""
        for i in range(self.session_list.count()):
            it = self.session_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == (mode, name):
                self._select_only(it)
                return
        it = QListWidgetItem(name)
        it.setData(Qt.ItemDataRole.UserRole, (mode, name))
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
        # Newest first, like the disk-backed rebuild.
        self.session_list.insertItem(0, it)
        self._select_only(it)

    def _select_only(self, item: QListWidgetItem) -> None:
        """Make `item` the sole selection. `setCurrentItem` alone only ADDS
        to the selection in ExtendedSelection mode (the multi-select needed
        for batch delete), so a previously-selected session would stay
        highlighted next to the active recording."""
        self.session_list.clearSelection()
        self.session_list.setCurrentItem(item)

    def _refresh_sessions(self) -> None:
        current = None
        item = self.session_list.currentItem()
        if item is not None:
            current = item.data(Qt.ItemDataRole.UserRole)
        # Suppress repaints across the clear+repopulate so the rebuild paints
        # once instead of flickering (clear() alone triggers a full repaint).
        self.session_list.setUpdatesEnabled(False)
        self.session_list.blockSignals(True)
        self.session_list.clear()
        mode = "dictate" if self.mode_dictate.isChecked() else "listen"
        for s in self.controller.list_sessions():
            if s["mode"] != mode:
                continue
            it = QListWidgetItem(s["name"])
            it.setData(Qt.ItemDataRole.UserRole, (s["mode"], s["name"]))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
            self.session_list.addItem(it)
            if current == (s["mode"], s["name"]):
                self.session_list.setCurrentItem(it)
        self.session_list.blockSignals(False)
        self.session_list.setUpdatesEnabled(True)
        self._selection_changed()
        # The rebuild ran with signals blocked, so no selection handler fired
        # — when idle, load the (re-)selected session's saved transcript into
        # the pane. Without this the pane keeps the LAST LIVE render after a
        # stop, which can miss the final committed window line (and, for a
        # resumed session, would never show the merged old+new content).
        if not self.running:
            self._session_selected(self.session_list.currentItem(), None)

    # --- polling -----------------------------------------------------------

    def _poll(self) -> None:
        st = self.controller.pull()
        status = st["status"]
        if self._deleted is not None:
            n = self._deleted
            self._deleted = None
            self._set_message(f"deleted {n} ✓" if n else "delete failed")
            self._refresh_sessions()
        models = st["models"]
        label = status
        if self._msg_ticks > 0:
            self._msg_ticks -= 1
            label = self._msg or label
        # Single-line status in the header's right slot: elided to the
        # label's width (the full text lives in the tooltip), so any status —
        # long error messages included — renders at a fixed size and the
        # header never reflows. The tooltip is only set when the text changes:
        # resetting it every tick would keep restarting the hover timer.
        self.status_lbl.setText(
            self.status_lbl.fontMetrics().elidedText(
                label, Qt.TextElideMode.ElideRight, self.status_lbl.width() - 4
            )
        )
        if label != self.status_lbl.toolTip():
            self.status_lbl.setToolTip(label)

        if status.startswith("error"):
            # Full error goes into the transcript (the status line above is
            # elided): one red banner appended after the content, once.
            self.transcript.setPlaceholderText("")
            if not self._error_rendered:
                self._error_rendered = True
                self.transcript.append(
                    f'<div style="color: #dc2626; margin-top: 12px;">'
                    f"{html.escape(status)}</div>"
                )
        else:
            # Placeholder for the truly empty document only (idle / the tick
            # before models are set). Once models are loaded the hint + model
            # list render as document content — QTextEdit draws its placeholder
            # as a SINGLE line, embedded newlines are dropped.
            if status == "listening":
                hint = "Listening — start speaking…"
            elif status == "stopped":
                hint = self._idle_hint()
            elif status == "starting" or status == "loading":
                hint = "Loading models… (first run downloads weights). "
            else:
                hint = self._idle_hint()
            self.transcript.setPlaceholderText(hint)

        running_now = status == "listening"
        if running_now:
            self._set_running(True)
        elif (status == "stopped" or status.startswith("error")) and self.running:
            self._set_running(False)
            self._refresh_sessions()

        # The engine names the live session as soon as its thread starts, but
        # its files only appear at stop — move the sidebar selection to it
        # immediately so the list follows the active recording and the stop
        # refresh keeps the NEW session selected (and shown) instead of the
        # previously-selected one.
        if self.running and st["session"] is not None:
            self._track_live_session(st["session"]["mode"], st["session"]["name"])

        if st["snapshot"] is not None:
            snapshot, version = st["snapshot"]
            # Only live sessions render snapshots: after "stopped"/"error" the
            # pane already holds the saved transcript (loaded by the stop
            # refresh), and a final late snapshot must not clobber it.
            if version != self.last_version and status == "listening":
                self.last_version = version
                self._render(snapshot, status, models)
        elif models and self.transcript.document().isEmpty():
            # Models loaded but no snapshot rendered yet (the engine only
            # pushes one on run()): show the empty-state hint + model list now.
            self._render_empty_state(status, models)

    def _render(
        self, snap: TranscriptSnapshot, status: str, models: Sequence[str]
    ) -> None:
        if not snap.chunks and (snap.window is None or not snap.window.plain):
            # No speech yet — keep the hint + model list visible instead of
            # wiping it with an empty snapshot.
            self._render_empty_state(status, models)
            return
        sb = self.transcript.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40
        if self.dictating:
            if _has_confidence_colors(snap):
                parts = _flowing_styled_parts(
                    [c.plain for c in snap.chunks],
                    [c.styled for c in snap.chunks],
                    snap.window.plain if snap.window is not None else None,
                    snap.window.styled if snap.window is not None else None,
                )
                self.transcript.setHtml(
                    self._resume_prefix
                    + (" " if self._resume_prefix else "")
                    + " ".join(_styled_to_html(p) for p in parts)
                )
            else:
                self.transcript.setPlainText(
                    _join_prefix_text(
                        self._resume_prefix_plain,
                        _flowing_text(
                            [c.plain for c in snap.chunks if c.plain],
                            snap.window.plain if snap.window is not None else None,
                        ),
                    )
                )
        else:
            self.transcript.setHtml(self._resume_prefix + _snapshot_html(snap))
        if at_bottom:
            sb.setValue(sb.maximum())

    def _render_empty_state(self, status: str, models: Sequence[str]) -> None:
        """Hint + model list as document content while the transcript has no
        speech yet. QTextEdit renders its placeholder as ONE line (embedded
        newlines are dropped), so the model list must be real document
        content to show. Only rendered once the models are loaded and the
        session is listening — the list belongs under "Listening — start
        speaking…"; during loading the placeholder hint is enough. Models
        render one font size smaller, in the same subtle placeholder color.
        Disappears once speech renders."""
        if status != "listening" or not models:
            return
        hint = (
            "Listening — start dictating…"
            if self.dictating
            else "Listening — start speaking…"
        )
        smaller = max(9, self.transcript.font().pointSize() - 1)
        # Same subtle look as the placeholder ("Loading models…" before the
        # models are set). The palette's PlaceholderText color is typically
        # semi-transparent — QTextEdit alpha-blends it over the base when it
        # draws the placeholder, while rich text would paint it opaque. Blend
        # it against the pane background ourselves to reproduce the exact
        # placeholder appearance.
        pal = self.transcript.palette()
        ph = pal.color(QPalette.ColorRole.PlaceholderText)
        base = pal.color(QPalette.ColorRole.Base)
        a = ph.alpha() / 255
        color = QColor(
            round(ph.red() * a + base.red() * (1 - a)),
            round(ph.green() * a + base.green() * (1 - a)),
            round(ph.blue() * a + base.blue() * (1 - a)),
        ).name()
        self.transcript.setHtml(
            self._resume_prefix
            + f'<div style="color: {color};">'
            + html.escape(hint)
            + "<br>"
            + f'<div style="font-size: {smaller}pt;">'
            + "<br>".join(html.escape(m) for m in models)
            + "</div>"
            + "</div>"
        )

    def closeEvent(self, event) -> None:
        self._flush_autosave()
        self.controller.shutdown()
        thread = self.controller._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        super().closeEvent(event)


def _app_icon() -> QIcon | None:
    """Best-effort window icon: the 256 px PNG `scripts/make_icon.py` drops in
    build/ (macOS dock/title-bar icons come from the .app bundle instead)."""
    png = PROJECT_ROOT / "build" / "icon.png"
    if png.is_file():
        return QIcon(str(png))
    return None


def main() -> None:
    controller = SessionController()
    # Warm up PortAudio's CoreAudio HAL on the MAIN thread. Opening a stream
    # from a worker thread when the HAL was never initialized on the main
    # thread fails with PaMacCore err=-50 on macOS (the CLI — main thread —
    # worked while the GUI's engine thread hit it on Start). The enumeration
    # itself prints err=-50 for broken output endpoints, so it runs with
    # stderr suppressed.
    with _quiet_portaudio():
        try:
            sd.query_devices()
        except Exception:  # noqa: S110, BLE001 - best-effort; audio still handled at Start
            pass

    app = QApplication(sys.argv)
    base_font = app.font()
    base_font.setPointSize(_saved_font_size())
    app.setFont(base_font)
    icon = _app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    window = MainWindow(controller)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
