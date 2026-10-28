"""Small helpers: natural sort, ogg duration parsing, ffmpeg discovery, console."""
from __future__ import annotations

import os
import re
import struct
import shutil
import subprocess
import sys
from pathlib import Path

_FFMPEG_CANDIDATES = ("ffmpeg", "ffmpeg.exe")


# --------------------------------------------------------------------------- #
# console
# --------------------------------------------------------------------------- #
def setup_console() -> None:
    """Make stdout/stderr UTF-8 on Windows so Chinese/Japanese never crashes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
        except Exception:
            pass


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# sorting
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"(\d+)")


def natural_key(text: str):
    """Sort keys that order 'noz2' before 'noz10'."""
    parts = _NUM_RE.split(text.lower())
    return [int(p) if p.isdigit() else p for p in parts]


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for cand in _FFMPEG_CANDIDATES:
        if Path(cand).exists():
            return str(Path(cand).resolve())
    raise RuntimeError(
        "找不到 ffmpeg。请先安装 FFmpeg（winget install Gyan.FFmpeg）并确保它在 PATH 中。"
    )


def find_ffprobe() -> str:
    """ffprobe next to ffmpeg (never string-replace: the install dir may contain 'ffmpeg')."""
    ext = ".exe" if os.name == "nt" else ""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    try:
        cand = Path(find_ffmpeg()).with_name("ffprobe" + ext)
        if cand.is_file():
            return str(cand)
    except Exception:
        pass
    raise RuntimeError("找不到 ffprobe，请确认 FFmpeg 完整安装并在 PATH 中。")


def has_filter(ffmpeg: str, name: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        ).stdout
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines())


def run(cmd: list[str], quiet: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **({"stdout": subprocess.DEVNULL} if quiet else {}),
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            "命令执行失败:\n  " + " ".join(cmd) + "\n" + (proc.stderr or "")[-2000:]
        )
    return proc


# --------------------------------------------------------------------------- #
# ogg duration (pure python, no ffprobe subprocess per file)
# --------------------------------------------------------------------------- #
def ogg_duration(path: Path) -> float:
    """Duration of an Ogg Vorbis/Opus file from granule positions (fast)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            # sample rate from the Vorbis identification header
            head = fh.read(min(size, 8192))
            rate = 0
            i = head.find(b"\x01vorbis")
            if i >= 0 and len(head) >= i + 16:
                rate = struct.unpack("<I", head[i + 12 : i + 16])[0]
            elif head[:4] == b"OggS":
                # Opus: rate lives in the OpusHead header (input sample rate)
                j = head.find(b"OpusHead")
                if j >= 0 and len(head) >= j + 16:
                    rate = struct.unpack("<I", head[j + 12 : j + 16])[0]
            if not rate:
                return 0.0
            fh.seek(max(0, size - 65536))
            tail = fh.read()
        granule = 0
        idx = tail.rfind(b"OggS")
        while idx >= 0:
            if idx + 14 <= len(tail):
                g = struct.unpack("<q", tail[idx + 6 : idx + 14])[0]
                if g > 0:
                    granule = g
                    break
            idx = tail.rfind(b"OggS", 0, idx)
        return granule / float(rate) if granule else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m:02d}分{s:02d}秒"
    if m:
        return f"{m}分{s:02d}秒"
    return f"{s}秒"


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_INVALID_FN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_filename(name: str, fallback: str = "voicepack") -> str:
    """Windows-safe file base name."""
    cleaned = _INVALID_FN.sub("_", str(name)).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:120] or fallback
