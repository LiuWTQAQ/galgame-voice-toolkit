"""Audio/video assembly: PCM stream -> FLAC master -> MP3 -> MP4 (still CG image)."""
from __future__ import annotations

import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .types import Clip
from .util import find_ffmpeg, find_ffprobe, human_size, human_time, log

SR = 48000
CH = 2
BYTES_PER_SEC = SR * CH * 2

# 疑似清单 CSV 的固定列（顺序重要：用户会拿它对时间轴听）
SUSPECT_CSV_FIELDS = ["start", "end", "name", "action", "score", "votes", "text", "reasons"]


def _decode_pcm(path: Path, ffmpeg: str, sr: int = SR, ch: int = CH) -> bytes:
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ar",
            str(sr),
            "-ac",
            str(ch),
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"解码失败: {path}\n{proc.stderr.decode('utf-8', 'replace')[-800:]}")
    return proc.stdout


def build_master(
    clips: list[Clip],
    out_flac: str | Path,
    gap: float = 0.5,
    ffmpeg: str | None = None,
    workers: int = 8,
    progress=None,
) -> Path:
    """Concatenate clips with `gap` seconds of digital silence in between.

    The PCM is streamed into a FLAC encoder, so 5+ hours of audio never sits in RAM
    and never needs a raw WAV on disk.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    out_flac = Path(out_flac)
    out_flac.parent.mkdir(parents=True, exist_ok=True)
    silence = b"\x00" * int(SR * CH * 2 * gap)

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "s16le",
        "-ar",
        str(SR),
        "-ac",
        str(CH),
        "-i",
        "-",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        str(out_flac),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    total = len(clips)
    try:
        chunk = max(workers * 4, 16)
        done = 0
        for start in range(0, total, chunk):
            batch = clips[start : start + chunk]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pcm_batch = list(pool.map(lambda c: _decode_pcm(c.path, ffmpeg), batch))
            for i, pcm in enumerate(pcm_batch):
                proc.stdin.write(pcm)
                if start + i < total - 1:
                    proc.stdin.write(silence)
            done += len(batch)
            if progress:
                progress(done, total)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"拼接编码失败 (ffmpeg exit {code})\n{stderr[-2000:]}")
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
    return out_flac


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 执行失败:\n  " + " ".join(cmd) + "\n" + proc.stderr.decode("utf-8", "replace")[-2000:]
        )


def to_mp3(master: str | Path, out_mp3: str | Path, bitrate: str = "192k", ffmpeg: str | None = None) -> Path:
    ffmpeg = ffmpeg or find_ffmpeg()
    out_mp3 = Path(out_mp3)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(master),
            "-map",
            "0:a",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-id3v2_version",
            "3",
            str(out_mp3),
        ]
    )
    return out_mp3


def to_mp4(
    cover_png: str | Path,
    master: str | Path,
    out_mp4: str | Path,
    fps: int = 5,
    crf: int = 20,
    preset: str = "veryfast",
    ffmpeg: str | None = None,
    audio_bitrate: str = "192k",
) -> Path:
    """Mux a still cover image with the master audio into an upload-friendly MP4."""
    ffmpeg = ffmpeg or find_ffmpeg()
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # "-shortest" alone is unreliable with a looped still image, so pin the exact
    # audio length: otherwise the video track can end up several seconds longer.
    dur = probe_duration(master)
    trim = ["-t", f"{dur:.3f}"] if dur > 0 else []
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(cover_png),
            "-i",
            str(master),
            "-map",
            "1:a",
            "-map",
            "0:v",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-tune",
            "stillimage",
            "-r",
            str(fps),
            "-shortest",
            *trim,
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
    )
    return out_mp4


def split_mp3(
    master: str | Path,
    out_dir: str | Path,
    stem: str,
    minutes: float,
    bitrate: str = "192k",
    ffmpeg: str | None = None,
) -> list[Path]:
    """Split the master into <stem>_part_01.mp3, part_02... every `minutes` minutes."""
    ffmpeg = ffmpeg or find_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{stem}_part_%02d.mp3"
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(master),
            "-map",
            "0:a",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-f",
            "segment",
            "-segment_time",
            str(int(minutes * 60)),
            str(pattern),
        ]
    )
    # an exact-multiple total length makes ffmpeg emit a trailing empty segment
    parts = []
    for p in sorted(out_dir.glob(f"{stem}_part_*.mp3")):
        if p.stat().st_size > 4096:
            parts.append(p)
        else:
            p.unlink(missing_ok=True)
    return parts


def split_mp4(
    cover_png: str | Path,
    master: str | Path,
    out_dir: str | Path,
    stem: str,
    minutes: float,
    fps: int = 5,
    crf: int = 20,
    preset: str = "veryfast",
    audio_bitrate: str = "192k",
    ffmpeg: str | None = None,
) -> list[Path]:
    """Split the still-image video into parts (for sites that cap single-video length)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(master)
    pattern = out_dir / f"{stem}_part_%02d.mp4"
    args = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(cover_png),
        "-i",
        str(master),
        "-map",
        "1:a",
        "-map",
        "0:v",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-tune",
        "stillimage",
        "-r",
        str(fps),
    ]
    if dur > 0:
        args += ["-t", f"{dur:.3f}"]
    args += [
        "-force_key_frames",
        f"expr:gte(t,n_forced*{int(minutes * 60)})",
        "-f",
        "segment",
        "-segment_time",
        str(int(minutes * 60)),
        "-segment_format",
        "mp4",
        # NB: plain "-movflags +faststart" is ignored by the segment muxer;
        # the per-segment format option is what actually moves moov to the front.
        "-segment_format_options",
        "movflags=+faststart",
        "-reset_timestamps",
        "1",
        str(pattern),
    ]
    _run(args)
    parts = []
    for p in sorted(out_dir.glob(f"{stem}_part_*.mp4")):
        if probe_duration(p) >= 0.3:
            parts.append(p)
        else:
            p.unlink(missing_ok=True)
    return parts


def build_suspect_mp3(
    clips: list[Clip],
    out_mp3: str | Path,
    csv_path: str | Path,
    meta: list[dict] | None = None,
    gap: float = 0.5,
    bitrate: str = "192k",
    ffmpeg: str | None = None,
) -> Path:
    """把「疑似句」（drop + review）按原顺序拼成一个 mp3，并写一份时间轴 CSV。

    CSV 列固定为 ``start,end,name,action,score,votes,text,reasons``：

    * ``start`` / ``end`` 是这句在该 mp3 里的起止秒数（保留 3 位小数），
      时间轴 = 各句真实时长 + 句间 ``gap`` 静音，永远和 mp3 对得上；
    * ``meta`` 与 ``clips`` 逐项对齐（可短可缺），用来自带 action/score/votes/text/reasons；
      缺的字段回退到 ``clip`` 本身（text）或空字符串。

    PCM 是边解码边喂给 mp3 编码器的，几百句也不会把整段音频塞进内存。
    """
    if not clips:
        raise ValueError("没有可疑句，无法生成疑似清单 mp3")
    ffmpeg = ffmpeg or find_ffmpeg()
    out_mp3 = Path(out_mp3)
    csv_path = Path(csv_path)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    gap = max(0.0, float(gap))
    silence = b"\x00" * int(SR * CH * 2 * gap)
    gap_dur = len(silence) / BYTES_PER_SEC  # 用实际写入的静音长度算时间轴，不会有取整误差

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "s16le",
        "-ar",
        str(SR),
        "-ac",
        str(CH),
        "-i",
        "-",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-id3v2_version",
        "3",
        str(out_mp3),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    metas = list(meta or [])
    rows: list[dict] = []
    cursor = 0.0
    total = len(clips)
    workers = min(8, max(2, (total + 15) // 16)) if total > 1 else 1
    try:
        chunk = max(workers * 4, 8)
        for start in range(0, total, chunk):
            batch = clips[start : start + chunk]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pcm_batch = list(pool.map(lambda c: _decode_pcm(c.path, ffmpeg), batch))
            for i, pcm in enumerate(pcm_batch):
                index = start + i
                clip = batch[i]
                dur = len(pcm) / BYTES_PER_SEC
                row_start = cursor
                row_end = row_start + dur
                proc.stdin.write(pcm)
                if index < total - 1:
                    proc.stdin.write(silence)
                cursor = row_end + (gap_dur if index < total - 1 else 0.0)

                m = metas[index] if index < len(metas) else {}
                score = m.get("score", "")
                if isinstance(score, (int, float)):
                    score = f"{float(score):.2f}"
                votes = m.get("votes", "")
                if isinstance(votes, (list, tuple, set)):
                    votes = "/".join(str(v) for v in votes)
                reasons = m.get("reasons", "")
                if isinstance(reasons, (list, tuple, set)):
                    reasons = "; ".join(str(r) for r in reasons)
                rows.append(
                    {
                        "start": f"{row_start:.3f}",
                        "end": f"{row_end:.3f}",
                        "name": clip.name,
                        "action": str(m.get("action", "") or ""),
                        "score": str(score or ""),
                        "votes": str(votes or ""),
                        "text": str(m.get("text") or clip.text or "").replace("\r", " ").replace("\n", " "),
                        "reasons": str(reasons or ""),
                    }
                )
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"疑似清单 mp3 编码失败 (ffmpeg exit {code})\n{stderr[-2000:]}")
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUSPECT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_mp3


def probe_duration(path: str | Path, ffprobe: str | None = None) -> float:
    exe = ffprobe or find_ffprobe()
    proc = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def report_progress(done: int, total: int) -> None:
    log(f"    拼接进度: {done}/{total} ({done * 100 // max(total, 1)}%)")
