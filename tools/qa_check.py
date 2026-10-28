"""产物质检：用 ffprobe / ffmpeg 检查生成的 MP3、MP4、封面 PNG 是否符合上传要求。

用法（在项目根目录）：
    python tools/qa_check.py  E:\\mrxyy\\voicepack_out
    python tools/qa_check.py  E:\\mrxyy\\voicepack_out\\noz.mp4 noz.mp3 cover.png
    python tools/qa_check.py  产物目录 --expected-minutes 270
    python tools/qa_check.py  产物目录 --json        # 机器可读

检查项
------
MP4：容器 mp4、视频 h264 / yuv420p、分辨率（默认 1920x1080）、帧率合理、
     moov 在 mdat 之前（faststart，网页“边下边播”必需）、音频 aac 48kHz 立体声、
     时长与同名的 MP3 一致、首帧与开头 5 秒音频可以正常解码。
MP3：格式 mp3、48kHz 立体声、码率接近 192k、时长、开头 5 秒可以正常解码。
PNG：1920x1080、RGB、没有透明通道。
疑似清单：`<名字>_suspect.mp3` 与 `<名字>_suspect.csv` 是否成对、CSV 列是否齐全、
     时间轴（start/end）是否单调递增、CSV 末句 end 与 mp3 实际时长是否吻合
     （对不上说明句间静音或解码时长算错了，按 CSV 听会错位）。

只出 MP3、没有 MP4 的目录也支持（没提供 CG 图时就是这样），不会误报失败。

退出码：0 = 全部通过（可以有警告）；1 = 有失败项；2 = 没有找到可检查的文件。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image  # 只用于封面 PNG
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def _force_utf8_console() -> None:
    """GBK 控制台下输出中文/日文会 UnicodeEncodeError，统一改成 UTF-8（失败则替换不可编码字符）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - 非 TTY / 被重定向时不支持 reconfigure
            pass


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

MP4_VIDEO_CODEC = "h264"
MP4_PIX_FMT = "yuv420p"
MP4_AUDIO_CODEC = "aac"
MP3_CODEC = "mp3"
DEFAULT_SIZE = (1920, 1080)
SUSPECT_COLS = ["start", "end", "name", "action", "score", "votes", "text", "reasons"]


def _tool(name: str) -> str:
    env = os.environ.get(name.upper())
    if env and Path(env).is_file():
        return env
    exe = shutil.which(name) or shutil.which(name + ".exe")
    if exe:
        return exe
    raise RuntimeError(f"找不到 {name}，请先安装 FFmpeg 并加入 PATH")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _probe(path: Path) -> dict | None:
    """ffprobe -show_format -show_streams -> dict；失败返回 None。"""
    cmd = [
        _tool("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except Exception:
        return None


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds or 0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def _stream(probe: dict, kind: str) -> dict | None:
    for st in probe.get("streams", []):
        if st.get("codec_type") == kind:
            return st
    return None


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key) or ""
        if "/" in val:
            num, _, den = val.partition("/")
            try:
                n, d = float(num), float(den)
                if d:
                    return n / d
            except Exception:
                pass
        else:
            try:
                return float(val)
            except Exception:
                pass
    return 0.0


def _mp4_box_order(path: Path, limit: int = 64) -> list[str]:
    """按顺序返回 MP4 顶层 box 名字（moov 是否在 mdat 之前 = faststart）。"""
    order: list[str] = []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size_total = fh.tell()
            pos = 0
            fh.seek(0)
            for _ in range(limit):
                if pos + 8 > size_total:
                    break
                fh.seek(pos)
                head = fh.read(16)
                if len(head) < 8:
                    break
                box_size = struct.unpack(">I", head[0:4])[0]
                box_type = head[4:8].decode("latin-1", "replace")
                header = 8
                if box_size == 1:  # 64 位大小
                    if len(head) < 16:
                        break
                    box_size = struct.unpack(">Q", head[8:16])[0]
                    header = 16
                elif box_size == 0:  # 一直到文件末尾
                    box_size = size_total - pos
                order.append(box_type)
                if box_size < header:
                    break
                pos += box_size
    except Exception:
        pass
    return order


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, level: str, target: str, message: str) -> None:
        self.items.append({"level": level, "target": target, "message": message})

    def count(self, level: str) -> int:
        return sum(1 for it in self.items if it["level"] == level)

    def failed(self) -> bool:
        return self.count(FAIL) > 0


# --------------------------------------------------------------------------- #
# 单项检查
# --------------------------------------------------------------------------- #
def check_mp4(path: Path, rep: Report, expected_seconds: float | None, tolerance: float,
              size: tuple[int, int], decode: bool) -> dict:
    name = path.name
    probe = _probe(path)
    if probe is None:
        rep.add(FAIL, name, "ffprobe 读不出来：文件损坏或不是视频")
        return {}

    fmt = probe.get("format", {})
    dur = float(fmt.get("duration") or 0.0)
    video = _stream(probe, "video")
    audio = _stream(probe, "audio")

    problems: list[str] = []
    warns: list[str] = []

    if video is None:
        problems.append("没有视频流")
    else:
        vcodec = video.get("codec_name", "?")
        pix = video.get("pix_fmt", "?")
        w, h = int(video.get("width") or 0), int(video.get("height") or 0)
        fps = _fps(video)
        if vcodec != MP4_VIDEO_CODEC:
            problems.append(f"视频编码 {vcodec}（应为 {MP4_VIDEO_CODEC}）")
        if pix != MP4_PIX_FMT:
            problems.append(f"像素格式 {pix}（应为 {MP4_PIX_FMT}，否则部分平台/浏览器不识别）")
        if (w, h) != size:
            warns.append(f"分辨率 {w}x{h}（期望 {size[0]}x{size[1]}）")
        if not (1.0 <= fps <= 30.0):
            warns.append(f"帧率 {fps:.2f} 不太正常（静图一般 1~10）")

    if audio is None:
        problems.append("没有音频流")
    else:
        acodec = audio.get("codec_name", "?")
        sr = int(audio.get("sample_rate") or 0)
        ch = int(audio.get("channels") or 0)
        if acodec != MP4_AUDIO_CODEC:
            problems.append(f"音频编码 {acodec}（应为 {MP4_AUDIO_CODEC}）")
        if sr != 48000:
            warns.append(f"采样率 {sr}Hz（B 站/AAC 建议 48000）")
        if ch not in (1, 2):
            warns.append(f"声道数 {ch} 异常")

    order = _mp4_box_order(path)
    faststart = True
    if "moov" in order and "mdat" in order:
        faststart = order.index("moov") < order.index("mdat")
    if not faststart:
        problems.append("moov 在 mdat 之后：没有开启 faststart，网页播放要等整个文件下载完")

    if dur <= 0:
        problems.append("时长读取为 0")
    if expected_seconds and dur > 0:
        diff = abs(dur - expected_seconds)
        if diff > tolerance:
            problems.append(
                f"时长 {_fmt_time(dur)} 与预期 {_fmt_time(expected_seconds)} 相差 {diff:.1f}s（容差 {tolerance:g}s）"
            )

    if decode and not problems:
        proc = _run(
            [_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
             "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
        )
        if proc.returncode != 0 or proc.stderr.strip():
            problems.append(f"视频解码报错：{proc.stderr.strip()[:300]}")
        proc = _run(
            [_tool("ffmpeg"), "-v", "error", "-nostdin", "-t", "5", "-i", str(path),
             "-map", "0:a:0", "-f", "null", "-"]
        )
        if proc.returncode != 0 or proc.stderr.strip():
            problems.append(f"音频解码报错：{proc.stderr.strip()[:300]}")

    summary = (
        f"{video.get('codec_name', '?') if video else '?'} "
        f"{(video.get('width') if video else '?')}x{(video.get('height') if video else '?')} "
        f"{video.get('pix_fmt', '?') if video else '?'} | "
        f"{audio.get('codec_name', '?') if audio else '?'} "
        f"{audio.get('sample_rate', '?') if audio else '?'}Hz | "
        f"时长 {_fmt_time(dur)} | faststart {'✓' if faststart else '✗'} | "
        f"大小 {int(fmt.get('size') or path.stat().st_size) / 1024 / 1024:.1f}MB"
    )

    if problems:
        rep.add(FAIL, name, "; ".join(problems) + " | " + summary)
    elif warns:
        rep.add(WARN, name, "; ".join(warns) + " | " + summary)
    else:
        rep.add(PASS, name, summary)

    return {"file": name, "duration": dur, "video": video, "audio": audio, "faststart": faststart}


def check_mp3(path: Path, rep: Report, expected_seconds: float | None, tolerance: float,
              decode: bool) -> dict:
    name = path.name
    probe = _probe(path)
    if probe is None:
        rep.add(FAIL, name, "ffprobe 读不出来：文件损坏或不是音频")
        return {}
    fmt = probe.get("format", {})
    audio = _stream(probe, "audio")
    dur = float(fmt.get("duration") or 0.0)
    problems: list[str] = []
    warns: list[str] = []

    if audio is None:
        problems.append("没有音频流")
    else:
        acodec = audio.get("codec_name", "?")
        sr = int(audio.get("sample_rate") or 0)
        ch = int(audio.get("channels") or 0)
        br = int(audio.get("bit_rate") or fmt.get("bit_rate") or 0)
        if acodec != MP3_CODEC:
            problems.append(f"编码 {acodec}（应为 mp3）")
        if sr != 48000:
            warns.append(f"采样率 {sr}Hz（建议 48000）")
        if ch not in (1, 2):
            warns.append(f"声道数 {ch} 异常")
        if br and not (128_000 <= br <= 320_000):
            warns.append(f"码率 {br // 1000}k 偏低（建议 192k 以上）")

    if dur <= 0:
        problems.append("时长读取为 0")
    if expected_seconds and dur > 0:
        diff = abs(dur - expected_seconds)
        if diff > tolerance:
            problems.append(f"时长 {_fmt_time(dur)} 与预期 {_fmt_time(expected_seconds)} 相差 {diff:.1f}s")

    if decode and not problems:
        proc = _run(
            [_tool("ffmpeg"), "-v", "error", "-nostdin", "-t", "5", "-i", str(path),
             "-map", "0:a:0", "-f", "null", "-"]
        )
        if proc.returncode != 0 or proc.stderr.strip():
            problems.append(f"音频解码报错：{proc.stderr.strip()[:300]}")

    summary = (
        f"{audio.get('codec_name', '?') if audio else '?'} "
        f"{audio.get('sample_rate', '?') if audio else '?'}Hz "
        f"{int(audio.get('channels') or 0) if audio else '?'}ch "
        f"{int((audio.get('bit_rate') or 0)) // 1000 if audio else '?'}k | "
        f"时长 {_fmt_time(dur)} | 大小 {int(fmt.get('size') or path.stat().st_size) / 1024 / 1024:.1f}MB"
    )
    if problems:
        rep.add(FAIL, name, "; ".join(problems) + " | " + summary)
    elif warns:
        rep.add(WARN, name, "; ".join(warns) + " | " + summary)
    else:
        rep.add(PASS, name, summary)
    return {"file": name, "duration": dur, "audio": audio}


def check_cover(path: Path, rep: Report, size: tuple[int, int]) -> dict:
    name = path.name
    info: dict = {"file": name}
    problems: list[str] = []
    if Image is not None:
        try:
            with Image.open(path) as im:
                w, h = im.size
                mode = im.mode
            info.update({"width": w, "height": h, "mode": mode})
            if (w, h) != size:
                problems.append(f"尺寸 {w}x{h}（应为 {size[0]}x{size[1]}）")
            if mode != "RGB":
                problems.append(f"模式 {mode}（应为 RGB，否则部分编码器/平台会出问题）")
            summary = f"{w}x{h} {mode} | 大小 {path.stat().st_size / 1024:.0f}KB"
        except Exception as exc:  # noqa: BLE001
            rep.add(FAIL, name, f"打不开：{exc}")
            return info
    else:
        probe = _probe(path)
        if probe is None:
            rep.add(FAIL, name, "读不出来（且没有 Pillow）")
            return info
        st = _stream(probe, "video") or {}
        w, h = int(st.get("width") or 0), int(st.get("height") or 0)
        info.update({"width": w, "height": h, "mode": st.get("pix_fmt", "?")})
        if (w, h) != size:
            problems.append(f"尺寸 {w}x{h}（应为 {size[0]}x{size[1]}）")
        summary = f"{w}x{h} {st.get('pix_fmt', '?')}"

    if problems:
        rep.add(FAIL, name, "; ".join(problems) + " | " + summary)
    else:
        rep.add(PASS, name, summary)
    return info


# --------------------------------------------------------------------------- #
# 疑似清单 mp3 + CSV 对齐检查
# --------------------------------------------------------------------------- #
def check_suspect_pair(mp3: Path, csv_path: Path, rep: Report, tolerance: float = 1.0) -> dict:
    """核对 `<名字>_suspect.mp3` 与 `<名字>_suspect.csv`：列、行数、时间轴是否自洽。"""
    name = mp3.name
    info: dict = {"file": name}
    if not csv_path.is_file():
        rep.add(WARN, name, f"缺少对照 CSV（{csv_path.name}），无法核对时间轴；重新跑 build 会重新生成")
        return info
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        rep.add(FAIL, name, f"读不了 {csv_path.name}: {exc}")
        return info

    probe = _probe(mp3)
    if probe is None:
        rep.add(FAIL, name, "ffprobe 读不出来：文件损坏或不是音频")
        return info
    dur = float(probe.get("format", {}).get("duration") or 0.0)
    info.update({"duration": dur, "rows": len(rows)})

    missing = [c for c in SUSPECT_COLS if c not in columns]
    problems: list[str] = []
    if missing:
        problems.append("CSV 缺列: " + ",".join(missing))
    if not rows:
        problems.append("CSV 没有数据行")

    last_end = 0.0
    if not missing and rows:
        prev_end = 0.0
        for i, row in enumerate(rows):
            try:
                start = float(row.get("start") or 0.0)
                end = float(row.get("end") or 0.0)
            except (TypeError, ValueError):
                problems.append(f"第 {i + 2} 行 start/end 不是数字（按 CSV 听会错位）")
                break
            if end <= start:
                problems.append(f"第 {i + 2} 行 end <= start")
                break
            if i == 0 and abs(start) > 0.05:
                problems.append(f"首句 start 应为 0，实际 {start:.3f}s")
            if i > 0 and start < prev_end - 0.001:
                problems.append(f"第 {i + 2} 行与上一行时间重叠/倒序")
                break
            prev_end = end
        last_end = prev_end
        if dur > 0:
            diff = abs(dur - last_end)
            tol = max(float(tolerance), 1.0)
            if diff > tol:
                problems.append(
                    f"CSV 末句 end {last_end:.3f}s 与 mp3 实际时长 {dur:.3f}s 差 {diff:.2f}s（容差 {tol:g}s）"
                )

    summary = f"CSV {len(rows)} 行 | mp3 时长 {_fmt_time(dur)} | 末句 end {last_end:.2f}s"
    if problems:
        rep.add(FAIL, name, "; ".join(problems) + " | " + summary)
    else:
        rep.add(PASS, name, "时间轴自洽 | " + summary)
    return info


def _suspect_pairs(files: list[Path], dirs: list[Path]) -> list[tuple[Path, Path]]:
    """找出所有 `*_suspect.mp3` 及其同目录同名 CSV。"""
    pairs: list[tuple[Path, Path]] = []
    for f in files:
        if f.suffix.lower() == ".mp3" and f.stem.endswith("_suspect"):
            pairs.append((f, f.with_suffix(".csv")))
    known = {str(mp3.resolve()).lower() for mp3, _ in pairs}
    for d in dirs:  # 目录里只有 CSV、mp3 被删了的情况也提示一下
        for csv_path in sorted(d.glob("*_suspect.csv")):
            mp3 = csv_path.with_suffix(".mp3")
            if mp3.is_file() and str(mp3.resolve()).lower() not in known:
                pairs.append((mp3, csv_path))
                known.add(str(mp3.resolve()).lower())
    return pairs


# --------------------------------------------------------------------------- #
# 目录批量检查
# --------------------------------------------------------------------------- #
def check_target(path: Path, rep: Report, expected_seconds: float | None, tolerance: float,
                 size: tuple[int, int], decode: bool) -> None:
    suffix = path.suffix.lower()
    try:
        empty = path.stat().st_size == 0
    except OSError:
        empty = True
    if empty:
        rep.add(FAIL, path.name, "空文件（0 字节），生成过程可能被中断了")
        return
    if suffix == ".mp4":
        check_mp4(path, rep, expected_seconds, tolerance, size, decode)
    elif suffix == ".mp3":
        check_mp3(path, rep, expected_seconds, tolerance, decode)
    elif suffix in (".png", ".jpg", ".jpeg"):
        check_cover(path, rep, size)
    else:
        rep.add(WARN, path.name, "不认识的文件类型，已跳过")


def _pair_durations(paths: list[Path], rep: Report) -> None:
    """同名 MP3 / MP4 时长应一致（说明音轨没有在封装时被截断）。"""
    durations: dict[str, dict[str, float]] = {}
    for p in paths:
        probe = _probe(p) if p.suffix.lower() in (".mp3", ".mp4") else None
        if not probe:
            continue
        dur = float(probe.get("format", {}).get("duration") or 0.0)
        durations.setdefault(p.stem, {})[p.suffix.lower()] = dur
    for stem, durs in sorted(durations.items()):
        if ".mp3" in durs and ".mp4" in durs and durs[".mp3"] > 0 and durs[".mp4"] > 0:
            diff = abs(durs[".mp3"] - durs[".mp4"])
            if diff > 0.75:
                rep.add(
                    FAIL,
                    stem,
                    f"MP3 时长 {_fmt_time(durs['.mp3'])} 与 MP4 {_fmt_time(durs['.mp4'])} 不一致（差 {diff:.1f}s）",
                )
            else:
                rep.add(PASS, stem, f"MP3/MP4 时长一致（差 {diff:.2f}s）")


def collect(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    dirs: list[Path] = []
    for p in paths:
        if p.is_dir():
            dirs.append(p)
            for ext in ("*.mp4", "*.mp3", "*.png"):
                files.extend(sorted(p.glob(ext)))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[警告] 路径不存在: {p}", file=sys.stderr)
    # 去重并保持顺序
    seen = set()
    uniq = []
    for f in files:
        key = str(f.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq, dirs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="检查 voicepack 产物：MP4 编码/封装/时长、MP3 质量、封面 PNG 尺寸"
    )
    ap.add_argument("paths", nargs="+", help="文件或目录（目录会检查其中的 mp4/mp3/png）")
    ap.add_argument(
        "--expected-minutes",
        type=float,
        default=None,
        help="预期总时长（分钟），用于核对 MP3/MP4 是否被截断，例如 270",
    )
    ap.add_argument("--expected-seconds", type=float, default=None, help="预期总时长（秒）")
    ap.add_argument("--tolerance", type=float, default=2.0, help="时长允许误差（秒，默认 2）")
    ap.add_argument("--size", default="1920x1080", help="期望分辨率（默认 1920x1080）")
    ap.add_argument("--no-decode", action="store_true", help="跳过实际解码，只做 ffprobe 检查（更快）")
    ap.add_argument("--json", action="store_true", help="输出 JSON，便于脚本读取")
    args = ap.parse_args(argv)
    _force_utf8_console()
    try:
        _tool("ffprobe")
        _tool("ffmpeg")
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        return 2

    try:
        size = tuple(int(x) for x in args.size.lower().split("x"))  # type: ignore[assignment]
        if len(size) != 2:
            raise ValueError
    except Exception:
        print(f"[错误] --size 格式应为 1920x1080，收到 {args.size}")
        return 2

    expected = args.expected_seconds
    if expected is None and args.expected_minutes is not None:
        expected = args.expected_minutes * 60.0

    targets = [Path(p) for p in args.paths]
    files, dirs = collect(targets)
    if not files:
        print("[错误] 没有找到可检查的 mp4/mp3/png 文件。")
        return 2

    rep = Report()
    mp4s = [f for f in files if f.suffix.lower() == ".mp4"]
    # 多个分段文件时，总时长预期只对单文件有意义
    per_file_expected = expected if len(mp4s) <= 1 else None

    # 所有检查只跑一遍（文本 / JSON 两种输出共用）
    for f in files:
        # --expected-minutes 只核对主产物：疑似清单 mp3 天然比正片短，
        # 分段产物（_part_）也不是整段时长，都不该套总时长预期
        exp = per_file_expected
        if exp is not None:
            low = f.name.lower()
            if "_suspect" in low or "_part_" in low:
                exp = None
        check_target(f, rep, exp, args.tolerance, size, not args.no_decode)
    _pair_durations(files, rep)

    # 疑似清单：mp3 时间轴与 CSV 是否自洽
    suspect_pairs = _suspect_pairs(files, dirs)
    for mp3, csv_path in suspect_pairs:
        check_suspect_pair(mp3, csv_path, rep, tolerance=max(args.tolerance, 1.0))

    if not args.json:
        for d in dirs:
            print(f"== 检查目录: {d}")
        if not mp4s:
            print("[提示] 没有找到 MP4：只出 MP3 也算正常（没提供 CG 图时就是这样）。")
        if any(f.stem.endswith("_suspect") for f in files):
            print("[提示] 发现疑似清单 mp3：CSV 的 start/end 就是该 mp3 里的秒数，可以直接拖着进度条对听。")
        for it in rep.items:
            print(f"[{it['level']}] {it['target']}: {it['message']}")
        print()
        print(
            f"汇总: {rep.count(PASS)} 项通过, {rep.count(WARN)} 项警告, "
            f"{rep.count(FAIL)} 项失败"
        )
        if rep.failed():
            print("结论: 有失败项，请按上面的提示修复后再上传。")
        elif rep.count(WARN):
            print("结论: 基本可用，警告项建议看一下（一般是分辨率/码率/采样率）。")
        else:
            print("结论: 全部通过，可以上传。")
    else:
        print(
            json.dumps(
                {
                    "ok": not rep.failed(),
                    "pass": rep.count(PASS),
                    "warn": rep.count(WARN),
                    "fail": rep.count(FAIL),
                    "has_mp4": bool(mp4s),
                    "suspect_pairs": len(suspect_pairs),
                    "items": rep.items,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 1 if rep.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
