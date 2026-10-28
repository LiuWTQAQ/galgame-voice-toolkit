"""Shared data types for the voice-pack pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Clip:
    """One voice file (one line of dialogue)."""

    path: Path
    name: str  # "noz001_045.ogg"
    stem: str  # "noz001_045"
    prefix: str  # "noz"  (character / group prefix)
    scene: int  # 1     (scene number from the file name)
    idx: int  # 45    (line number inside the scene)
    dur: float  # seconds

    # filled in by the ASR stage
    text: str = ""
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    asr_done: bool = False

    @property
    def scene_key(self) -> str:
        return f"{self.prefix}{self.scene:03d}"

    def __str__(self) -> str:
        return self.name


@dataclass
class Decision:
    """Filter verdict for one clip."""

    clip: Clip
    action: str  # "keep" | "drop" | "review"
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.clip.stem
