"""Audio I/O via ffmpeg (handles FLAC/MP3/etc. without extra Python codecs)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".aif", ".aiff", ".ogg", ".opus"}


def load(path: str | Path, sr: int, mono: bool = True) -> np.ndarray:
    """Decode a file to float32. Returns (n,) if mono else (n, 2)."""
    ch = 1 if mono else 2
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32).copy()
    return y if mono else y.reshape(-1, 2)


def find_audio(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
