"""Generate the Interpreter app icon into build/Interpreter.iconset/ plus a
single build/icon.png (256 px, the Qt window icon outside a bundle).

The iconset feeds `iconutil` (macOS) in scripts/make_app.sh to produce
Interpreter.icns for the .app bundle. The icon is drawn with QPainter (no
external art assets): a blue rounded square, a white speech bubble and a
mini audio waveform.

Run: <venv>/bin/python scripts/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
)

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
ICONSET = BUILD / "Interpreter.iconset"

BG_TOP = QColor("#4f8cff")
BG_BOTTOM = QColor("#2f5fd0")

# (px, iconset filename) — the standard mac iconset layout.
SIZES = [
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png"),
]

# Artwork margin (fraction of the canvas): the old full-bleed design looked
# larger than neighboring dock icons.
_MARGIN = 0.08


def _draw_icon(painter: QPainter, size: int) -> None:
    """Draw the icon at the given size: a baby-blue rounded square with a
    white chat bubble (no waveform bars). Geometry relative to `size`."""
    s = float(size)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.save()
    painter.translate(s * _MARGIN, s * _MARGIN)
    painter.scale(1 - 2 * _MARGIN, 1 - 2 * _MARGIN)

    bg = QPainterPath()
    bg.addRoundedRect(QRectF(0, 0, s, s), s * 0.18, s * 0.18)
    grad = QLinearGradient(0, 0, 0, s)
    grad.setColorAt(0.0, BG_TOP)
    grad.setColorAt(1.0, BG_BOTTOM)
    painter.fillPath(bg, grad)

    bubble = QPainterPath()
    bubble.addRoundedRect(
        QRectF(0.24 * s, 0.24 * s, 0.52 * s, 0.42 * s), 0.12 * s, 0.12 * s
    )
    tail = QPainterPath()
    tail.moveTo(0.30 * s, 0.66 * s)
    tail.lineTo(0.50 * s, 0.66 * s)
    tail.lineTo(0.37 * s, 0.84 * s)
    tail.closeSubpath()
    bubble = bubble.united(tail)
    painter.fillPath(bubble, Qt.GlobalColor.white)

    painter.restore()


def main() -> int:
    ICONSET.mkdir(parents=True, exist_ok=True)
    for size, name in SIZES:
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        painter = QPainter(img)
        _draw_icon(painter, size)
        painter.end()
        if not img.save(str(ICONSET / name)):
            print(f"failed to write {name}")
            return 1

    icon_png = BUILD / "icon.png"
    img = QImage(256, 256, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    _draw_icon(painter, 256)
    painter.end()
    if not img.save(str(icon_png)):
        print(f"failed to write {icon_png}")
        return 1

    print(f"icon assets written to {ICONSET} and {icon_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
