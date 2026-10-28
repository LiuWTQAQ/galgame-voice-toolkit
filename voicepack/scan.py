"""Scan a voice folder: find clips, exclude *_ref.ogg, parse names, measure duration."""
from __future__ import annotations

import re
from pathlib import Path

from .types import Clip
from .util import human_time, natural_key, ogg_duration

_NAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z_]+?)(?P<scene>\d+)_(?P<idx>\d+)(?P<tail>.*)$",
    re.IGNORECASE,
)
_REF_RE = re.compile(r"_ref\d*$", re.IGNORECASE)


def is_ref(stem: str) -> bool:
    """True for 'xxx_ref', 'xxx_ref2' ... - the alternate takes the user wants dropped."""
    return bool(_REF_RE.search(stem))


def parse_name(name: str) -> tuple[str, int, int, int] | None:
    """'noz001_045.ogg' -> ('noz', 1, 45, 0). Returns None for unparsable names."""
    p = Path(name)
    stem = p.stem
    if is_ref(stem):
        return None
    m = _NAME_RE.match(stem)
    if not m:
        return None
    prefix = m.group("prefix")
    scene = int(m.group("scene"))
    idx = int(m.group("idx"))
    order = 0
    return prefix, scene, idx, order


def scan_folder(folder: str | Path, keep_ref: bool = False) -> list[Clip]:
    """List the clips of one folder, naturally sorted, refs excluded."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    entries = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".ogg"]
    clips: list[Clip] = []
    skipped_ref = 0
    skipped_odd = 0
    for path in entries:
        parsed = parse_name(path.name)
        if parsed is None:
            if is_ref(path.stem):
                skipped_ref += 1
            else:
                skipped_odd += 1
            continue
        prefix, scene, idx, _ = parsed
        clips.append(
            Clip(
                path=path,
                name=path.name,
                stem=path.stem,
                prefix=prefix,
                scene=scene,
                idx=idx,
                dur=ogg_duration(path),
            )
        )
    clips.sort(key=lambda c: (c.prefix.lower(), c.scene, c.idx, natural_key(c.name)))
    scan_folder.last_stats = {  # type: ignore[attr-defined]
        "ref_skipped": skipped_ref,
        "odd_skipped": skipped_odd,
        "total_files": len(entries),
    }
    return clips


def summarize(clips: list[Clip], gap: float = 0.5, kept: int | None = None) -> str:
    total = sum(c.dur for c in clips)
    n = len(clips)
    out = []
    scenes = sorted({c.scene_key for c in clips})
    out.append(f"  句数      : {n}  (共 {len(scenes)} 个场景编号)")
    out.append(f"  纯语音时长: {human_time(total)}")
    if kept is not None:
        out.append(f"  保留句数  : {kept}")
    out.append(f"  加 {gap}s 停顿后总长: {human_time(total + max(0, n - 1) * gap)}")
    return "\n".join(out)
