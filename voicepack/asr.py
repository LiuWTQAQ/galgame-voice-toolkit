"""ASR 转写：ffmpeg 解码 + faster-whisper 转写。

为什么不用 faster-whisper 自带的 PyAV 解码
-----------------------------------------
本机 av 19.0.2 的 API 与 faster-whisper 1.2.1 不兼容（``stream.resample`` 直接
AttributeError / metadata_errors），因此统一用 ffmpeg 子进程解码成 16kHz 单声道
float32，再用 ``np.frombuffer`` 还原成 numpy 数组喂给 ``model.transcribe``。

三条转写路径
------------
1. **分块批量（默认，GPU 首选）** ``_transcribe_chunked``：
   按顺序把句子拼成约 240s 的块（句间插 0.3s 静音，块内最多 32 句），每块只调用一次
   ``transcribe``，用 segment 的绝对时间戳按「段中点落在哪句」归属回每一句。
   这样把逐句调用的固定开销摊掉：块小到只有 3~4 句时固定开销会和推理时间同量级，
   全量 16.7k 句会白白多花几十分钟；240s/块 + 解码多线程预取后 GPU fp16 实测 RTF ≈ 0.055。
2. **逐句（fallback）** ``_transcribe_per_clip``：
   VAD 打开转写；如果 VAD 把整句吞掉（segs 为空），自动关掉 VAD 重试，
   并把该句标记为 ``clip.vad_empty = True``（娇喘/气声经常被 VAD 吞掉，必须留痕）。
3. **CPU 多进程** ``workers > 1``：每个进程一个模型，按句分发，适合没有 GPU 的机器。

其他约定
--------
- 模型/设备：``device="auto"`` 先试 cuda(float16)，失败自动回落 cpu(int8)；
  会自动把 ``.deps/nvidia/*/bin``（cublas/cudnn DLL）prepend 到 ``PATH``。
- 断点续传：JSONL 缓存，键 = ``name + 文件大小 + mtime + 模型名``；
  命中缓存直接回填 ``clip.text`` 等字段，不重复转写。失败文件不写缓存（下次重试）。
- 失败容错：单句解码/转写异常只记空文本 + ``clip.asr_error``，不中断整批。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .types import Clip
from .util import find_ffmpeg, log, setup_console

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
SR = 16000  # Whisper 固定 16kHz
SILENCE_SEC = 0.3  # 分块时句间插入的静音
DEFAULT_CHUNK_SEC = 240.0  # 每块目标音频长度：块越大固定开销摊得越薄（实测 240s/32 句 18x 实时，40s/14 句只有 8x）；显存紧就调小
DEFAULT_CHUNK_CLIPS = 32  # 每块句子数上限
DEFAULT_DECODE_WORKERS = 8  # 解码线程数（ffmpeg 子进程解码，与 GPU 推理重叠）
CACHE_VERSION = 1

ProgressFn = Callable[[int, int], None] | None


# --------------------------------------------------------------------------- #
# DLL / 设备准备
# --------------------------------------------------------------------------- #
_NVIDIA_READY = False


def _prepare_nvidia_dlls() -> None:
    """把 .deps/nvidia/*/bin 里的 CUDA DLL 目录加到 Windows 搜索路径。

    必须在 ``import faster_whisper`` / 加载模型之前调用，否则 ctranslate2 找不到
    cublas64_12.dll / cudnn64_9.dll，cuda 初始化会失败。
    """
    global _NVIDIA_READY
    if _NVIDIA_READY:
        return
    _NVIDIA_READY = True
    roots: list[Path] = []
    env_root = os.environ.get("VOICEPACK_DEPS")
    if env_root:
        roots.append(Path(env_root))
    # __file__ = <workspace>/voicepack/asr.py → parents[1] = <workspace>
    roots.append(Path(__file__).resolve().parents[1] / ".deps")
    for root in roots:
        nv = root / "nvidia"
        if not nv.is_dir():
            continue
        for sub in sorted(nv.iterdir()):
            bin_dir = sub / "bin"
            if not bin_dir.is_dir():
                continue
            try:
                os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def cuda_available() -> bool:
    """ctranslate2 是否能看见 CUDA 设备（不保证显存够，加载失败仍有回落）。"""
    try:
        _prepare_nvidia_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """把 ``("auto", "auto")`` 解析成具体设备与量化类型。"""
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


# --------------------------------------------------------------------------- #
# 音频解码
# --------------------------------------------------------------------------- #
def decode_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """用 ffmpeg 把任意音频解码成 16kHz 单声道 float32 波形。

    Raises:
        RuntimeError: ffmpeg 解码失败（损坏文件/不支持的编码）。
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")[-500:]
        raise RuntimeError(f"ffmpeg 解码失败: {path}\n{err}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg 解码得到空音频: {path}")
    return audio.copy()


# --------------------------------------------------------------------------- #
# 模型缓存
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[tuple[str, str, str, int], object] = {}


def _load_model(model_name: str, device: str, compute_type: str, cpu_threads: int = 8):
    """按 (模型, 设备, 量化, 线程) 缓存 WhisperModel 实例。"""
    _prepare_nvidia_dlls()
    key = (model_name, device, compute_type, cpu_threads)
    model = _MODEL_CACHE.get(key)
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=1,
        )
        _MODEL_CACHE[key] = model
    return model


# --------------------------------------------------------------------------- #
# JSONL 断点续传缓存
# --------------------------------------------------------------------------- #
@dataclass
class CacheRecord:
    """一条缓存记录（JSONL 一行）。"""

    name: str
    model: str
    size: int
    mtime: float
    text: str = ""
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    vad_empty: bool = False
    dur: float = 0.0

    @classmethod
    def from_dict(cls, obj: dict) -> "CacheRecord":
        return cls(
            name=str(obj.get("name", "")),
            model=str(obj.get("model", "")),
            size=int(obj.get("size", -1)),
            mtime=float(obj.get("mtime", 0.0)),
            text=str(obj.get("text", "")),
            avg_logprob=float(obj.get("avg_logprob", 0.0)),
            no_speech_prob=float(obj.get("no_speech_prob", 0.0)),
            vad_empty=bool(obj.get("vad_empty", False)),
            dur=float(obj.get("dur", 0.0)),
        )

    def as_dict(self) -> dict:
        return {
            "v": CACHE_VERSION,
            "name": self.name,
            "model": self.model,
            "size": self.size,
            "mtime": round(self.mtime, 3),
            "text": self.text,
            "avg_logprob": round(self.avg_logprob, 4),
            "no_speech_prob": round(self.no_speech_prob, 4),
            "vad_empty": self.vad_empty,
            "dur": round(self.dur, 3),
        }


class AsrCache:
    """JSONL 断点续传缓存。后写的同名记录覆盖先写的。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, CacheRecord] = {}
        self._fh = None

    def load(self) -> "AsrCache":
        """读取已有缓存（允许文件不存在）。"""
        if not self.path.is_file():
            return self
        bad = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = CacheRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    bad += 1
                    continue
                if rec.name:
                    self.entries[rec.name] = rec
        if bad:
            log(f"[缓存] 跳过 {bad} 行损坏记录: {self.path}")
        return self

    @staticmethod
    def _stat(path: Path) -> tuple[int, float]:
        try:
            st = path.stat()
            return st.st_size, st.st_mtime
        except OSError:
            return -1, 0.0

    def get_valid(self, clip: Clip, model_name: str) -> CacheRecord | None:
        """命中条件：name 相同 + 大小/mtime 未变 + 模型名相同。"""
        rec = self.entries.get(clip.name)
        if rec is None or rec.model != model_name:
            return None
        size, mtime = self._stat(clip.path)
        if rec.size != size or abs(rec.mtime - mtime) > 1.0:
            return None
        return rec

    def put(self, rec: CacheRecord) -> None:
        """追加一行并立即 flush（进程被杀也最多丢最后一句）。"""
        self.entries[rec.name] = rec
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "AsrCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 结果回填
# --------------------------------------------------------------------------- #
def _apply_result(clip: Clip, res: dict) -> None:
    """把一次转写结果写回 Clip（含动态字段 vad_empty / asr_error）。"""
    clip.text = res.get("text", "") or ""
    clip.avg_logprob = float(res.get("avg_logprob", 0.0))
    clip.no_speech_prob = float(res.get("no_speech_prob", 0.0))
    clip.asr_done = True
    # Clip 是普通 dataclass（没有 __slots__），可以挂额外字段；
    # 下游 filter.py 用 getattr(clip, "vad_empty", False) 读取。
    clip.vad_empty = bool(res.get("vad_empty", False))  # type: ignore[attr-defined]
    clip.asr_error = res.get("error")  # type: ignore[attr-defined]


def _cache_record(clip: Clip, model_name: str) -> CacheRecord:
    size, mtime = AsrCache._stat(clip.path)
    return CacheRecord(
        name=clip.name,
        model=model_name,
        size=size,
        mtime=mtime,
        text=clip.text,
        avg_logprob=clip.avg_logprob,
        no_speech_prob=clip.no_speech_prob,
        vad_empty=bool(getattr(clip, "vad_empty", False)),
        dur=clip.dur,
    )


# --------------------------------------------------------------------------- #
# 分块批量转写
# --------------------------------------------------------------------------- #
def _group_chunks(
    clips: Sequence[Clip], chunk_sec: float = DEFAULT_CHUNK_SEC, max_clips: int = DEFAULT_CHUNK_CLIPS
) -> list[list[Clip]]:
    """按顺序切块：每块音频 ≤ chunk_sec，句子数 ≤ max_clips。"""
    groups: list[list[Clip]] = []
    cur: list[Clip] = []
    cur_sec = 0.0
    for clip in clips:
        dur = max(clip.dur, 0.2)
        need = dur + (SILENCE_SEC if cur else 0.0)
        if cur and (cur_sec + need > chunk_sec or len(cur) >= max_clips):
            groups.append(cur)
            cur = [clip]
            cur_sec = dur
        else:
            cur.append(clip)
            cur_sec += need
    if cur:
        groups.append(cur)
    return groups


def _decode_or_error(path: str | Path) -> tuple[np.ndarray | None, str | None]:
    """解码一句；失败返回 ``(None, 错误信息)``，不打断整块。"""
    try:
        return decode_audio(path), None
    except Exception as exc:  # 损坏文件/不支持编码：只影响这一句
        return None, f"{type(exc).__name__}: {exc}"


def _build_chunk_audio(audios: Sequence[np.ndarray]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """把一块里的句子用 0.3s 静音拼起来，返回 (波形, 每句的 [start,end) 区间)。"""
    silence = np.zeros(int(SILENCE_SEC * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    offset = 0
    for i, audio in enumerate(audios):
        start = offset / SR
        parts.append(audio)
        offset += audio.size
        spans.append((start, offset / SR))
        if i < len(audios) - 1:
            parts.append(silence)
            offset += silence.size
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32), spans


def _locate_clip(spans: Sequence[tuple[float, float]], mid: float) -> int:
    """把段中点映射到句号：落在静音间隙的归前一句（lead 约定）。"""
    for i, (start, end) in enumerate(spans):
        if mid < start:
            return max(0, i - 1) if mid >= 0 else 0
        if start <= mid < end:
            return i
    return len(spans) - 1


def _transcribe_chunked(
    clips: Sequence[Clip],
    model,
    cache: AsrCache | None,
    model_name: str,
    progress: ProgressFn,
    chunk_sec: float,
    max_clips: int,
    decode_workers: int = 8,
) -> int:
    """分块转写主循环，返回成功句数。

    性能要点（全量 16.7k 句的关键）：

    - 块要**大**：默认 240s / 32 句（240s/32 句实测 18x 实时，40s/14 句只有 8x）。
      faster-whisper 对 >30s 输入会自己按 30s 窗口
      解码，但 segment 时间戳仍是相对整块的绝对时间，切句逻辑不受影响。
    - 解码**并行 + 预取**：解码池 8 线程，转写当前块时同时解码下一块，
      ffmpeg 耗时被 GPU 推理完全掩盖（串行解码仅解码就要 20+ 分钟/全量）。
    - 模型只建一次（``transcribe_clips`` 里加载），块间不重建。
    """
    groups = _group_chunks(clips, chunk_sec, max_clips)
    total = len(clips)
    done = 0
    empty = 0
    errors = 0
    t0 = time.time()
    decode_pool = ThreadPoolExecutor(max_workers=max(1, decode_workers))

    def submit_chunk(group: Sequence[Clip]) -> list:
        return [decode_pool.submit(_decode_or_error, c.path) for c in group]

    try:
        if not groups:
            return 0
        pending = submit_chunk(groups[0])
        for gi, group in enumerate(groups):
            # 先拿当前块解码结果，同时把下一块解码提交出去（与 GPU 推理重叠）
            audios: list[np.ndarray] = []
            decoded_clips: list[Clip] = []
            failed: list[tuple[Clip, str]] = []
            for clip, fut in zip(group, pending):
                audio, err = fut.result()
                if audio is None:
                    failed.append((clip, err or "解码失败"))
                else:
                    audios.append(audio)
                    decoded_clips.append(clip)
            pending = submit_chunk(groups[gi + 1]) if gi + 1 < len(groups) else []

            if audios:
                audio, spans = _build_chunk_audio(audios)
                results = _transcribe_one_chunk_audio(audio, spans, model)
                for clip, res in zip(decoded_clips, results):
                    _apply_result(clip, res)
                    if not clip.text:
                        empty += 1
                    if cache is not None:
                        cache.put(_cache_record(clip, model_name))
                    done += 1
            for clip, err in failed:
                _apply_result(clip, {"text": "", "avg_logprob": 0.0, "no_speech_prob": 0.0,
                                     "vad_empty": False, "error": err})
                errors += 1
                done += 1

            if progress is not None:
                progress(done, total)
            elif gi % 25 == 0 or done >= total:
                _log_progress(done, total, t0)
    finally:
        decode_pool.shutdown(wait=False, cancel_futures=True)
    if empty:
        log(f"[ASR] 其中 {empty} 句分块后没有分到任何 segment（文本为空，交给 filter 的启发式判定）")
    if errors:
        log(f"[ASR] {errors} 句解码失败（已记空文本，下次运行会重试）")
    return done


def _transcribe_one_chunk_audio(
    audio: np.ndarray, spans: Sequence[tuple[float, float]], model
) -> list[dict]:
    """对已经拼好的块音频调用一次 transcribe，并按时间戳切回每一句。"""
    segments = list(
        model.transcribe(
            audio,
            language="ja",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=False,
            word_timestamps=False,
            task="transcribe",
        )[0]
    )
    buckets: list[list] = [[] for _ in spans]
    for seg in segments:
        mid = (float(seg.start) + float(seg.end)) / 2.0
        buckets[_locate_clip(spans, mid)].append(seg)

    results: list[dict] = []
    for segs in buckets:
        if not segs:
            results.append({"text": "", "avg_logprob": 0.0, "no_speech_prob": 0.0, "vad_empty": False})
            continue
        text = "".join(str(s.text).strip() for s in segs).strip()
        weight = sum(max(float(s.end) - float(s.start), 1e-3) for s in segs)
        avg_lp = (
            sum(float(s.avg_logprob) * max(float(s.end) - float(s.start), 1e-3) for s in segs) / weight
            if weight > 0
            else float(segs[0].avg_logprob)
        )
        nsp = max(float(s.no_speech_prob) for s in segs)
        results.append({"text": text, "avg_logprob": avg_lp, "no_speech_prob": nsp, "vad_empty": False})
    return results


# --------------------------------------------------------------------------- #
# 逐句转写（fallback / CPU 多进程）
# --------------------------------------------------------------------------- #
def _transcribe_audio_per_clip(audio: np.ndarray, model) -> dict:
    """逐句转写：先开 VAD；VAD 吞掉整句时关 VAD 重试并标记 vad_empty。"""
    def _run(vad: bool) -> list:
        return list(
            model.transcribe(
                audio,
                language="ja",
                beam_size=1,
                vad_filter=vad,
                condition_on_previous_text=False,
                without_timestamps=False,
                word_timestamps=False,
                task="transcribe",
            )[0]
        )

    segs = _run(True)
    vad_empty = False
    if not segs:
        # 典型情况：娇喘/呼吸声被 VAD 判成静音整句丢掉 → 关 VAD 重试拿回文本
        vad_empty = True
        segs = _run(False)
    if not segs:
        return {"text": "", "avg_logprob": 0.0, "no_speech_prob": 1.0, "vad_empty": True}

    weight = sum(max(float(s.end) - float(s.start), 1e-3) for s in segs)
    avg_lp = (
        sum(float(s.avg_logprob) * max(float(s.end) - float(s.start), 1e-3) for s in segs) / weight
        if weight > 0
        else float(segs[0].avg_logprob)
    )
    return {
        "text": "".join(str(s.text).strip() for s in segs).strip(),
        "avg_logprob": avg_lp,
        "no_speech_prob": max(float(s.no_speech_prob) for s in segs),
        "vad_empty": vad_empty,
    }


def _transcribe_per_clip(
    clips: Sequence[Clip],
    model,
    cache: AsrCache | None,
    model_name: str,
    progress: ProgressFn,
) -> int:
    """单进程逐句转写（分块路径失败时的兜底）。"""
    total = len(clips)
    done = 0
    t0 = time.time()
    for clip in clips:
        try:
            audio = decode_audio(clip.path)
            res = _transcribe_audio_per_clip(audio, model)
        except Exception as exc:  # 单句失败不影响整批
            res = {"text": "", "avg_logprob": 0.0, "no_speech_prob": 0.0, "vad_empty": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        _apply_result(clip, res)
        if cache is not None and not res.get("error"):
            cache.put(_cache_record(clip, model_name))
        done += 1
        if progress is not None:
            progress(done, total)
        elif done % 50 == 0 or done == total:
            _log_progress(done, total, t0)
    return done


# --------------------------------------------------------------------------- #
# 多进程 worker（Windows 用 spawn，需要模块级函数）
# --------------------------------------------------------------------------- #
_WORKER_MODEL = None


def _worker_init(model_name: str, device: str, compute_type: str, cpu_threads: int) -> None:
    global _WORKER_MODEL
    _prepare_nvidia_dlls()
    _WORKER_MODEL = _load_model(model_name, device, compute_type, cpu_threads)


def _worker_transcribe(item: tuple[str, str]) -> dict:
    """worker 内转写一句；任何异常都转成空文本结果。"""
    path, name = item
    try:
        audio = decode_audio(path)
        res = _transcribe_audio_per_clip(audio, _WORKER_MODEL)
    except Exception as exc:
        res = {
            "text": "",
            "avg_logprob": 0.0,
            "no_speech_prob": 0.0,
            "vad_empty": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    res["name"] = name
    return res


def _transcribe_multiprocess(
    clips: Sequence[Clip],
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    workers: int,
    cache: AsrCache | None,
    progress: ProgressFn,
) -> int:
    """CPU 多进程逐句转写：每进程一个模型，按句分发。"""
    total = len(clips)
    by_name = {c.name: c for c in clips}
    items = [(str(c.path), c.name) for c in clips]
    done = 0
    empty = 0
    errors = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(model_name, device, compute_type, cpu_threads),
    ) as pool:
        for res in pool.imap_unordered(_worker_transcribe, items, chunksize=4):
            clip = by_name.get(res.get("name", ""))
            if clip is None:
                continue
            _apply_result(clip, res)
            if res.get("error"):
                errors += 1
            elif not clip.text:
                empty += 1
            if cache is not None and not res.get("error"):
                cache.put(_cache_record(clip, model_name))
            done += 1
            if progress is not None:
                progress(done, total)
            elif done % 50 == 0 or done == total:
                _log_progress(done, total, t0)
    if errors:
        log(f"[ASR] {errors} 句转写失败（已记空文本，下次运行会重试）")
    if empty:
        log(f"[ASR] {empty} 句转写为空")
    return done


# --------------------------------------------------------------------------- #
# 进度
# --------------------------------------------------------------------------- #
def _log_progress(done: int, total: int, t0: float) -> None:
    """每 N 句打印一次进度 + 剩余时间估算。"""
    if total <= 0:
        return
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / rate if rate > 0 else 0.0
    pct = done * 100.0 / total
    log(
        f"[ASR] {done}/{total} 句 ({pct:5.1f}%)  "
        f"{rate:5.2f} 句/秒  剩余约 {left / 60:4.1f} 分钟"
    )


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def transcribe_clips(
    clips: Sequence[Clip],
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    workers: int = 1,
    cache_path: str | Path | None = None,
    progress: ProgressFn = None,
    chunked: bool = True,
    chunk_sec: float = DEFAULT_CHUNK_SEC,
    max_clips_per_chunk: int = DEFAULT_CHUNK_CLIPS,
    decode_workers: int = DEFAULT_DECODE_WORKERS,
    cpu_threads: int = 8,
) -> None:
    """原地转写 clips，填充 ``clip.text / avg_logprob / no_speech_prob / asr_done``。

    Args:
        clips: 待转写句子（会被原地修改）。
        model_name: faster-whisper 模型名或本地目录，默认 ``small``。
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``。auto 先试 cuda 失败回落 cpu。
        compute_type: ``"auto"`` → cuda 用 float16、cpu 用 int8。
        workers: >1 时走 CPU 多进程逐句路径（GPU 不支持多进程共享模型，会忽略）。
        cache_path: JSONL 缓存路径，None 表示不缓存。
        progress: 进度回调 ``progress(done, total)``，每个分块/句子调用一次。
        chunked: 默认 True。分块批量转写，速度约为逐句的 3~4 倍；
            分块过程抛异常会自动退回逐句路径。
        chunk_sec: 每块目标音频秒数（显存小就调小；块越大固定开销摊得越薄）。
        max_clips_per_chunk: 每块句子数上限。
        decode_workers: 分块路径的解码线程数（ffmpeg 并行解码 + 预取下一块）。
        cpu_threads: CPU 线程数。

    Notes:
        额外的动态字段：
        - ``clip.vad_empty``：VAD 吞掉了整句后关 VAD 才拿回文本（娇喘常见）。
        - ``clip.asr_error``：转写/解码失败原因（失败句 text 为空，且不写缓存）。
    """
    setup_console()
    if not clips:
        return

    cache: AsrCache | None = None
    if cache_path:
        cache = AsrCache(cache_path).load()
        log(f"[ASR] 缓存: {cache.path}（已有 {len(cache.entries)} 条）")

    # 1) 已完成的直接跳过（asr_done 由调用方设置或命中缓存）
    pending: list[Clip] = []
    hits = 0
    for clip in clips:
        if clip.asr_done:
            hits += 1
            continue
        if cache is not None:
            rec = cache.get_valid(clip, model_name)
            if rec is not None:
                _apply_result(
                    clip,
                    {
                        "text": rec.text,
                        "avg_logprob": rec.avg_logprob,
                        "no_speech_prob": rec.no_speech_prob,
                        "vad_empty": rec.vad_empty,
                    },
                )
                hits += 1
                continue
        pending.append(clip)

    total = len(clips)
    if hits:
        log(f"[ASR] 命中缓存/已完成 {hits}/{total} 句，剩余 {len(pending)} 句待转写")
    if not pending:
        if progress is not None:
            progress(total, total)
        if cache is not None:
            cache.close()
        return

    device, compute_type = _resolve_device(device, compute_type)
    log(
        f"[ASR] 模型={model_name} 设备={device} 量化={compute_type} "
        f"workers={workers if device != 'cuda' else 1} chunked={chunked}"
    )

    try:
        if device == "cuda":
            try:
                model = _load_model(model_name, device, compute_type, cpu_threads)
            except Exception as exc:
                log(f"[ASR] CUDA 模型加载失败，回落 CPU int8：{type(exc).__name__}: {exc}")
                device, compute_type = "cpu", "int8"
                model = _load_model(model_name, device, compute_type, cpu_threads)
        else:
            model = _load_model(model_name, device, compute_type, cpu_threads)

        use_workers = workers > 1 and device != "cuda"
        if chunked and not use_workers:
            try:
                _transcribe_chunked(
                    pending,
                    model,
                    cache,
                    model_name,
                    progress,
                    chunk_sec,
                    max_clips_per_chunk,
                    decode_workers,
                )
            except Exception as exc:
                log(
                    f"[ASR] 分块转写失败（{type(exc).__name__}: {exc}），"
                    f"退回逐句路径重试剩余句子"
                )
                remaining = [c for c in pending if not c.asr_done]
                if remaining:
                    if device == "cuda":
                        # 显存/内核问题：换 CPU 再兜一层
                        try:
                            model = _load_model(model_name, "cpu", "int8", cpu_threads)
                        except Exception:
                            pass
                    _transcribe_per_clip(remaining, model, cache, model_name, progress)
        elif use_workers:
            _transcribe_multiprocess(
                pending, model_name, device, compute_type, cpu_threads, workers, cache, progress
            )
        else:
            _transcribe_per_clip(pending, model, cache, model_name, progress)

        if progress is not None:
            progress(total, total)
    finally:
        if cache is not None:
            cache.close()

    done = sum(1 for c in clips if c.asr_done)
    empty = sum(1 for c in clips if c.asr_done and not c.text)
    log(f"[ASR] 完成 {done}/{total} 句，其中空文本 {empty} 句")


# --------------------------------------------------------------------------- #
# compact 统计（calibrate/CLI 用）
# --------------------------------------------------------------------------- #
def asr_stats(clips: Sequence[Clip]) -> dict:
    """汇总 ASR 结果：句数/总时长/空文本/平均 logprob/VAD 吞句数。"""
    done = [c for c in clips if c.asr_done]
    total_dur = sum(c.dur for c in clips)
    empty = [c for c in done if not c.text]
    vad_empty = [c for c in done if getattr(c, "vad_empty", False)]
    errors = [c for c in done if getattr(c, "asr_error", None)]
    lps = [c.avg_logprob for c in done if c.text]
    return {
        "clips": len(clips),
        "done": len(done),
        "duration": total_dur,
        "empty": len(empty),
        "vad_empty": len(vad_empty),
        "errors": len(errors),
        "avg_logprob": (sum(lps) / len(lps)) if lps else 0.0,
    }
