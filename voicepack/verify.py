"""多轮独立判别 + 多数表决：对第一轮可疑句（drop / review / 空文本）再做 N 轮 ASR。

设计目标（Lead 的第二轮需求）：

* **只复核可疑句**：第一轮 ``keep`` 的句子不重复花钱；
* **每轮配置不同**：beam 大小、VAD 开关、no_speech/log_prob 阈值各不同，
  避免同一解码路径重复犯同一个错；
* **必须走分块批量**：复用 ``asr._group_chunks`` / 并行解码 / 块内一次 transcribe +
  segment 中点归属，1200 句 × 3 轮在 RTX 4060 fp16 上应是分钟级而不是小时级；
* **多数表决**：3 轮里 ≥2 轮判敏感就剔除，反之保留；平票按剔除（过审优先）；
* **空文本可救回**：第一轮被 VAD 吞掉/解码不出文本的句子，换配置后拿到文本且不敏感
  就重新进入 ``keep``，``reason`` 里标注「转写恢复: …」。

本模块不改 ``asr.py`` / ``cli.py``：只 import 其已公开（含下划线）的框架函数。

缓存独立于 ASR 缓存：默认写 ``<folder>/voicepack_out/verify_<model>.jsonl``
（cli 会显式传 ``out_dir/verify_<model>.jsonl``），键 = 文件名 + 模型 + 轮次配置签名。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from . import asr, lexicon
from .filter import DROP_THRESHOLD, REVIEW_THRESHOLD, score_clip
from .types import Clip, Decision
from .util import human_time, log

ProgressFn = Callable[[int, int], None] | None

#: 纯喘息字符集：整句只由这些字符组成且长度够长时，无论分数高低都算「判敏感」。
#: 目的：补上 lexicon 对「ああああ…（20 个あ）」「うぅぅ…」这类长串单音喘息打分偏低的缺口
#: （长串归一化后已经不像短句，`moan_features` 只给 1.0 分，落在 review 线以下）。
#: 不含 は/ふ/え/へ 等会出现在笑声（あははは）里的假名，避免把笑声判成喘息。
_MOANISH_CHARS = frozenset("あいうえおんっー゛ぁぃぅぇぉゃゅょゎ")
#: 纯喘息文本最短长度：太短的（「うん」「あっ」）交给分数判断，不在这里一刀切。
_MOANISH_MIN_LEN = 6

CACHE_VERSION = 1
#: 默认每轮的最少轮数（``passes`` 小于它时按实际轮数跑）。
DEFAULT_PASSES = 3

# --------------------------------------------------------------------------- #
# 每轮解码配置
# --------------------------------------------------------------------------- #
#: 三轮固定配置：同一句用三种不同意的方式各听一遍。
#: 1. beam_size=5：整体最优路径，最接近「认真听」；
#: 2. vad_filter=True + 放宽 no_speech/log_prob：先静音切分再放宽门限，
#:    专门救「有声但第一轮没出文本」的喊叫/喘息段；
#: 3. beam_size=1 贪心 + condition_on_previous_text=False：快速路径，解码轨迹与 beam5 不同。
#:    注意：不要在这里开 temperature 回退列表——长娇喘句压缩比异常会连续触发回退，
#:    CPU 上单句成本能翻 3~5 倍（实测 40 句长句从 98s 涨到 6 分钟以上），得不偿失。
PASS_PRESETS: tuple[dict, ...] = (
    {
        "name": "beam5",
        "opts": {
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "word_timestamps": False,
        },
    },
    {
        "name": "vad-relaxed",
        "opts": {
            "beam_size": 1,
            "temperature": 0.0,
            "vad_filter": True,
            # 放宽：no_speech 很高时也照解，log_prob 门限也放松，尽量把文本捞出来
            "no_speech_threshold": 0.95,
            "log_prob_threshold": -2.0,
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "word_timestamps": False,
        },
    },
    {
        "name": "beam1-nocontext",
        "opts": {
            "beam_size": 1,
            "temperature": 0.0,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "word_timestamps": False,
        },
    },
)


def _opts_key(opts: Mapping[str, object]) -> str:
    """把一轮解码配置压成 8 位签名，进缓存键（配置改了旧缓存自然失效）。"""
    raw = json.dumps(dict(opts), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class VerifyAttempt:
    """一轮独立识别的结果。"""

    transcript: str  # 本轮转写文本（可能为空）
    score: float  # 用 filter.score_clip 打的敏感分
    asr_ok: bool  # 是否拿到了非空文本（解码失败/空文本 = False）
    sensitive: bool  # score >= drop_threshold
    detail: dict = field(default_factory=dict)  # 本轮配置与置信度，调试/CSV 用


@dataclass
class VerifyResult:
    """一句可疑句的多轮表决结果。"""

    stem: str
    attempts: list[VerifyAttempt] = field(default_factory=list)
    votes_bad: int = 0
    votes_good: int = 0
    action: str = "keep"  # 表决后的动作：keep | drop（多轮后不再有 review）
    best_text: str = ""  # 各轮里最可信的文本（空文本救回时很有用）
    reason: str = ""  # 一句话解释：「多轮判别 2/3 判敏感」「转写恢复: …」

    @property
    def total(self) -> int:
        """有效轮数（不含加载失败被跳过的轮）。"""
        return len(self.attempts)

    def as_dict(self) -> dict:
        """JSON 友好形式（调试/落盘用）。"""
        return {
            "stem": self.stem,
            "votes_bad": self.votes_bad,
            "votes_good": self.votes_good,
            "action": self.action,
            "best_text": self.best_text,
            "reason": self.reason,
            "attempts": [
                {
                    "transcript": a.transcript,
                    "score": a.score,
                    "asr_ok": a.asr_ok,
                    "sensitive": a.sensitive,
                    "detail": a.detail,
                }
                for a in self.attempts
            ],
        }


# --------------------------------------------------------------------------- #
# 选句
# --------------------------------------------------------------------------- #
def plan_suspects(
    decisions: Sequence[Decision],
    include_review: bool = True,
    include_drop: bool = True,
    min_score: float = REVIEW_THRESHOLD,
) -> list[Clip]:
    """挑出需要复核的句子。

    默认 = 第一轮判 ``drop`` 或 ``review`` 且分数 ≥ review 线的句子。
    空文本（≥1s）第一轮分数 1.6，通常落在 review 桶，也会被选中；
    ``min_score`` 只是兜底过滤，不需要时可传 0。

    Args:
        decisions: 第一轮过滤结果。
        include_review: 是否复核 review 桶。
        include_drop: 是否复核 drop 桶（drop 也复核一遍能减少误杀）。
        min_score: 只要分数达到该值的句子。

    Returns:
        去重后的 Clip 列表，保持传入顺序。
    """
    out: list[Clip] = []
    seen: set[str] = set()
    for d in decisions:
        if d.action == "review" and not include_review:
            continue
        if d.action == "drop" and not include_drop:
            continue
        if d.action not in ("review", "drop"):
            continue
        if d.score < min_score:
            continue
        if d.stem in seen:
            continue
        seen.add(d.stem)
        d.clip.__dict__["suspect_score"] = float(d.score)  # 表决时要知道「第一轮已经觉得可疑」
        out.append(d.clip)
    return out


# --------------------------------------------------------------------------- #
# 独立缓存（不污染 asr_*.jsonl）
# --------------------------------------------------------------------------- #
class VerifyCache:
    """JSONL 断点续传缓存：键 = (文件名, 模型, 轮次配置签名)。

    只缓存「这一轮跑出了什么」，表决结果每次重算（表决议价低且规则可能调整）。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[tuple[str, str, str], dict] = {}
        self._fh = None

    def load(self) -> "VerifyCache":
        """读取已有缓存（文件不存在时直接返回）。"""
        if not self.path.is_file():
            return self
        bad = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                key = (str(rec.get("name", "")), str(rec.get("model", "")), str(rec.get("pass", "")))
                if all(key):
                    self.entries[key] = rec
        if bad:
            log(f"[复核缓存] 跳过 {bad} 行损坏记录: {self.path}")
        return self

    @staticmethod
    def _stat(path: Path) -> tuple[int, float]:
        try:
            st = path.stat()
            return st.st_size, st.st_mtime
        except OSError:
            return -1, 0.0

    def get(self, clip: Clip, model_name: str, pass_key: str) -> dict | None:
        """命中条件：文件名 + 模型 + 配置签名 + 文件大小/mtime 都对得上。"""
        rec = self.entries.get((clip.name, model_name, pass_key))
        if rec is None:
            return None
        size, mtime = self._stat(clip.path)
        if rec.get("size") != size or abs(float(rec.get("mtime", 0.0)) - mtime) > 1.0:
            return None
        return rec

    def put(self, clip: Clip, model_name: str, pass_key: str, res: dict) -> None:
        """追加一行并立即 flush。"""
        size, mtime = self._stat(clip.path)
        rec = {
            "v": CACHE_VERSION,
            "name": clip.name,
            "model": model_name,
            "pass": pass_key,
            "size": size,
            "mtime": round(mtime, 3),
            "text": res.get("text", ""),
            "avg_logprob": round(float(res.get("avg_logprob", 0.0)), 4),
            "no_speech_prob": round(float(res.get("no_speech_prob", 0.0)), 4),
            "asr_ok": bool(res.get("asr_ok", bool(str(res.get("text", "")).strip()))),
        }
        self.entries[(clip.name, model_name, pass_key)] = rec
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "VerifyCache":
        return self.load()

    def __exit__(self, *exc) -> None:
        self.close()


def _default_cache_path(clips: Sequence[Clip], model_name: str) -> Path:
    """默认缓存位置：优先 ``<folder>/voicepack_out/verify_<model>.jsonl``。"""
    if not clips:
        raise ValueError("clips 为空，无法推导默认缓存路径")
    base = Path(clips[0].path).resolve().parent
    out_dir = base / "voicepack_out"
    if not out_dir.is_dir():
        out_dir = base
    return out_dir / f"verify_{model_name}.jsonl"


# --------------------------------------------------------------------------- #
# 分块转写（可配置解码参数）
# --------------------------------------------------------------------------- #
def _transcribe_chunk_with_opts(
    audio: np.ndarray, spans: Sequence[tuple[float, float]], model, opts: Mapping[str, object]
) -> list[dict]:
    """对拼好的块跑一次 ``model.transcribe(**opts)``，按段中点切回每句。"""
    segments = list(model.transcribe(audio, language="ja", task="transcribe", **dict(opts))[0])
    buckets: list[list] = [[] for _ in spans]
    for seg in segments:
        mid = (float(seg.start) + float(seg.end)) / 2.0
        buckets[asr._locate_clip(spans, mid)].append(seg)

    results: list[dict] = []
    for segs in buckets:
        if not segs:
            results.append({"text": "", "avg_logprob": 0.0, "no_speech_prob": 0.0, "asr_ok": False})
            continue
        text = "".join(str(s.text).strip() for s in segs).strip()
        weight = sum(max(float(s.end) - float(s.start), 1e-3) for s in segs)
        avg_lp = (
            sum(float(s.avg_logprob) * max(float(s.end) - float(s.start), 1e-3) for s in segs) / weight
            if weight > 0
            else float(segs[0].avg_logprob)
        )
        nsp = max(float(s.no_speech_prob) for s in segs)
        results.append(
            {"text": text, "avg_logprob": avg_lp, "no_speech_prob": nsp, "asr_ok": bool(text)}
        )
    return results


def _transcribe_round(
    todo: Sequence[Clip],
    model,
    opts: Mapping[str, object],
    cache: VerifyCache | None,
    model_name: str,
    pass_key: str,
    on_done: Callable[[int], None],
    decode_workers: int = 8,
) -> dict[str, dict]:
    """对一批句子跑某一轮配置，返回 ``{name: result}``（含缓存命中）。

    结构与 ``asr._transcribe_chunked`` 相同：分块 + 并行解码 + 预取下一块，
    这样每轮只把「真正需要转写的句子」喂给模型，固定开销被摊薄。
    """
    from concurrent.futures import ThreadPoolExecutor

    results: dict[str, dict] = {}
    pending_clips: list[Clip] = []
    for clip in todo:
        rec = cache.get(clip, model_name, pass_key) if cache is not None else None
        if rec is not None:
            results[clip.name] = {
                "text": str(rec.get("text", "")),
                "avg_logprob": float(rec.get("avg_logprob", 0.0)),
                "no_speech_prob": float(rec.get("no_speech_prob", 0.0)),
                "asr_ok": bool(rec.get("asr_ok", bool(str(rec.get("text", "")).strip()))),
            }
            on_done(1)
        else:
            pending_clips.append(clip)

    if not pending_clips:
        return results

    groups = asr._group_chunks(pending_clips)
    pool = ThreadPoolExecutor(max_workers=max(1, decode_workers))

    def submit(group: Sequence[Clip]) -> list:
        return [pool.submit(asr._decode_or_error, c.path) for c in group]

    try:
        pending = submit(groups[0])
        for gi, group in enumerate(groups):
            audios: list[np.ndarray] = []
            ok_clips: list[Clip] = []
            failed: list[tuple[Clip, str]] = []
            for clip, fut in zip(group, pending):
                audio, err = fut.result()
                if audio is None:
                    failed.append((clip, err or "解码失败"))
                else:
                    audios.append(audio)
                    ok_clips.append(clip)
            pending = submit(groups[gi + 1]) if gi + 1 < len(groups) else []

            if audios:
                chunk_audio, spans = asr._build_chunk_audio(audios)
                chunk_results = _transcribe_chunk_with_opts(chunk_audio, spans, model, opts)
                for clip, res in zip(ok_clips, chunk_results):
                    results[clip.name] = res
                    if cache is not None:
                        cache.put(clip, model_name, pass_key, res)
                for _ in ok_clips:
                    on_done(1)
            for clip, err in failed:
                res = {"text": "", "avg_logprob": 0.0, "no_speech_prob": 1.0,
                       "asr_ok": False, "error": err}
                results[clip.name] = res
                if cache is not None:
                    cache.put(clip, model_name, pass_key, res)
                on_done(1)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


# --------------------------------------------------------------------------- #
# 表决辅助
# --------------------------------------------------------------------------- #
def _moanish_only(text: str, min_len: int = _MOANISH_MIN_LEN) -> bool:
    """整句是否只由喘息字符构成（长串单音喘息的补漏规则）。

    ``ああああ…``（20 字）归一化后只是很长的同一个假名，`lexicon.moan_features`
    只按「同字连续」给 1.0 分，落在 review 线（1.5）以下——这类句子必须靠本规则补判。

    Args:
        text: 本轮转写文本。
        min_len: 归一化后的最短长度。

    Returns:
        True 表示「只由喘息/气声字符组成」，应记一票「判敏感」。
    """
    core = lexicon.normalize(text or "")
    if len(core) < min_len:
        return False
    return all(ch in _MOANISH_CHARS for ch in core)


_SUBSTANTIVE_MIN_LEN = 8  # 「实质正常文本」的最小长度（归一化后）
_SUBSTANTIVE_REPEAT_COVER = 0.5  # 重复片段覆盖超过该比例 → 不像正常台词


def _repeat_cover(norm: str, max_unit: int = 6) -> float:
    """最长「同一片段连续重复」覆盖了多少比例（0~1）。

    「行く行く行く」→ 单位 2×3=6/6=1.0；「ちょっと待ってちょっと」→ 低。
    """
    n = len(norm)
    if n < 4:
        return 0.0
    best = 0.0
    for unit in range(1, min(max_unit, n // 2) + 1):
        i = 0
        while i + unit <= n:
            base = norm[i: i + unit]
            run = 1
            j = i + unit
            while j + unit <= n and norm[j: j + unit] == base:
                run += 1
                j += unit
            if run >= 3:
                best = max(best, (run * unit) / n)
            i += 1
    return best


def _is_substantive(text: str) -> bool:
    """文本像一句完整正常台词，而不是「うん」「行く!行く!行く!」这种短词/重复。

    条件：归一化后 ≥8 字、含至少一个汉字、重复片段覆盖 < 60%。
    """
    norm = lexicon.normalize(text)
    if len(norm) < _SUBSTANTIVE_MIN_LEN:
        return False
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return False
    if _repeat_cover(norm) >= _SUBSTANTIVE_REPEAT_COVER:
        return False
    return True


def _attempt_label(
    text: str, asr_ok: bool, score: float, review_threshold: float, strong_good: bool
) -> str:
    """单轮投票标签：``bad`` / ``good`` / ``abstain``。

    规则（保守优先，宁删勿留）：

    1. 空文本/解码失败：默认记 **bad**（用户要求「失败多就踢出」）；但若同一句
       已有另一轮给出「实质正常文本」（见 :func:`_is_substantive`），说明空文本
       只是这一轮 whisper/VAD 没抓到，记 **abstain** 不参与多数；
    2. 分数 ≥ review 线（灰区也算）→ bad；
    3. 拿到文本但整句只有喘息字符（长串单音）→ bad；
    4. 其余（有文本、分数低于 review 线）→ good。
    """
    if not asr_ok or not text:
        return "abstain" if strong_good else "bad"
    if score >= review_threshold:
        return "bad"
    if _moanish_only(text):
        return "bad"
    return "good"


def _tally_votes(
    attempts: Sequence[VerifyAttempt],
    review_threshold: float,
    first_score: float,
    drop_threshold: float = DROP_THRESHOLD,
) -> tuple[int, int, int, str]:
    """多数表决。

    空文本的记法：默认 bad（用户要求「失败多就踢出」）；只有「另一轮拿到实质正常
    台词」时才弃权，且若第一轮已经是强证据（分数 ≥ ``drop_threshold``）则必须至少
    两轮实质正常才允许弃权，避免「一轮幻觉出正常台词 + 两轮失败」放行真喘息。

    Returns:
        ``(votes_bad, votes_good, abstain, override_reason)``；``override_reason``
        非空表示「票数上该留、但证据不足所以仍剔除」，会被拼进理由。
    """
    strong = [
        a for a in attempts
        if a.asr_ok and _is_substantive(a.transcript) and a.score < review_threshold
        and not _moanish_only(a.transcript)
    ]
    strong_good = bool(strong) and (first_score < drop_threshold or len(strong) >= 2)
    votes_bad = votes_good = abstain = 0
    for a in attempts:
        label = _attempt_label(a.transcript, a.asr_ok, a.score, review_threshold, strong_good)
        if label == "bad":
            votes_bad += 1
        elif label == "good":
            votes_good += 1
        else:
            abstain += 1
    override = ""
    if votes_bad < votes_good and not strong_good and first_score >= review_threshold:
        # 第一轮（生产管线）已经判可疑，多轮又没拿到实质正常文本 → 不能凭「うん」放行
        override = "首轮已判可疑且多轮未恢复实质台词，保守剔除"
    return votes_bad, votes_good, abstain, override


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def verify_suspects(
    clips: Sequence[Clip],
    model_name: str = "small",
    second_model: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
    passes: int = DEFAULT_PASSES,
    drop_threshold: float = DROP_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
    cache_path: str | Path | None = None,
    progress: ProgressFn = None,
    decode_workers: int = 8,
    cpu_threads: int = 8,
) -> dict[str, VerifyResult]:
    """对可疑句做多轮独立识别 + 多数表决。

    Args:
        clips: 待复核句子（通常来自 :func:`plan_suspects`）。
        model_name: 主模型（默认 small）。
        second_model: 可选的第二个模型，给定时追加一轮交叉验证（如 medium）。
        device: ``"auto"`` 先试 cuda，失败回落 cpu。
        compute_type: ``"auto"`` 时 cuda→float16、cpu→int8。
        passes: 轮数（0/负数 = 直接返回空 dict，即不复核）。
        drop_threshold: 单轮判敏感线（默认 3.0）。
        review_threshold: 单轮判正常线（默认 1.5）；灰区按敏感票。
        cache_path: 复核缓存路径；None 时写 ``<folder>/voicepack_out/verify_<model>.jsonl``。
        progress: ``progress(done, total)``，done 会跨轮累计（total = 句数 × 轮数）。
        decode_workers: ffmpeg 解码线程数（与 GPU 推理重叠）。
        cpu_threads: CPU 推理线程数。

    Returns:
        ``{stem: VerifyResult}``；第三轮模型加载失败等异常会跳过该轮并写日志。
    """
    if passes <= 0 or not clips:
        return {}
    if drop_threshold < review_threshold:
        raise ValueError(f"drop_threshold({drop_threshold}) 不能小于 review_threshold({review_threshold})")

    resolved_device, resolved_ct = asr._resolve_device(device, compute_type)
    models: dict[str, object] = {}

    def load(model_id: str) -> object | None:
        if model_id in models:
            return models[model_id]
        try:
            m = asr._load_model(model_id, resolved_device, resolved_ct, cpu_threads)
            models[model_id] = m
            return m
        except Exception as exc:  # noqa: BLE001  显存不足/模型缺失：回落 CPU 或跳过该轮
            if resolved_device == "cuda":
                log(f"[复核] 模型 {model_id} 在 cuda 加载失败（{exc}），回落 CPU int8")
                try:
                    m = asr._load_model(model_id, "cpu", "int8", cpu_threads)
                    models[model_id] = m
                    return m
                except Exception as exc2:  # noqa: BLE001
                    log(f"[复核] 模型 {model_id} CPU 也加载失败: {exc2}")
                    return None
            log(f"[复核] 模型 {model_id} 加载失败: {exc}")
            return None

    primary = load(model_name)
    if primary is None:
        raise RuntimeError(f"复核模型 {model_name} 加载失败")
    second = load(second_model) if second_model else None

    rounds: list[tuple[str, str, dict]] = []
    for i in range(passes):
        preset = PASS_PRESETS[i % len(PASS_PRESETS)]
        name = preset["name"] if i < len(PASS_PRESETS) else f"{preset['name']}#{i + 1}"
        rounds.append((name, model_name, dict(preset["opts"])))
    if second and second_model:
        rounds.append((f"cross-{second_model}", second_model, dict(PASS_PRESETS[0]["opts"])))

    total_rounds = len(rounds)
    total = len(clips) * total_rounds
    done = [0]
    t0 = time.time()

    def on_done(n: int) -> None:
        done[0] += n
        if progress is not None:
            progress(done[0], total)

    cache = VerifyCache(cache_path or _default_cache_path(clips, model_name))
    cache.load()
    log(
        f"[复核] {len(clips)} 句 × {total_rounds} 轮，模型={model_name}"
        + (f" + {second_model}" if second_model and second else "")
        + f"，device={resolved_device}/{resolved_ct}，缓存={cache.path.name}"
    )

    # name → 每轮结果列表 [(pass_name, model_name, res)]
    per_clip: dict[str, list[tuple[str, str, dict]]] = {c.name: [] for c in clips}
    try:
        for round_idx, (pass_name, model_id, opts) in enumerate(rounds, start=1):
            model = models.get(model_id)
            if model is None:
                log(f"[复核] 第 {round_idx}/{total_rounds} 轮（{pass_name}）模型不可用，跳过")
                continue
            key = f"{pass_name}@{_opts_key(opts)}"
            log(f"[复核] 第 {round_idx}/{total_rounds} 轮: {pass_name}"
                f"（beam={opts.get('beam_size')} vad={opts.get('vad_filter')}）")
            round_start = done[0]
            got = _transcribe_round(
                clips, model, opts, cache, model_id, key, on_done, decode_workers
            )
            for clip in clips:
                res = got.get(clip.name)
                if res is not None:
                    per_clip[clip.name].append((pass_name, model_id, res))
            log(f"  本轮完成 {done[0] - round_start}/{len(clips)} 句，累计 {done[0]}/{total}"
                f"（{human_time(time.time() - t0)}）")
    finally:
        cache.close()

    # --- 表决 ---
    results: dict[str, VerifyResult] = {}
    for clip in clips:
        attempts: list[VerifyAttempt] = []
        for pass_name, model_id, res in per_clip.get(clip.name, ()):
            text = str(res.get("text", "")).strip()
            asr_ok = bool(res.get("asr_ok", bool(text)))
            probe = replace(clip, text=text, avg_logprob=float(res.get("avg_logprob", 0.0)),
                            no_speech_prob=float(res.get("no_speech_prob", 0.0)),
                            asr_done=True)
            # 空文本也走 score_clip：>=1s 的空文本会被 +1.6 判可疑（这正是需要的保守）
            score, reasons = score_clip(probe)
            sensitive = score >= drop_threshold
            attempts.append(
                VerifyAttempt(
                    transcript=text,
                    score=round(float(score), 2),
                    asr_ok=asr_ok,
                    sensitive=sensitive,
                    detail={
                        "pass": pass_name,
                        "model": model_id,
                        "avg_logprob": round(float(res.get("avg_logprob", 0.0)), 3),
                        "no_speech_prob": round(float(res.get("no_speech_prob", 0.0)), 3),
                        "reasons": reasons,
                        "error": res.get("error"),
                    },
                )
            )

        # 表决：灰区/纯喘息文本/空文本 → bad，只有「拿到文本且分数低于 review 线」才算 good
        first_score = float(clip.__dict__.get("suspect_score", 0.0))
        votes_bad, votes_good, abstain, override = _tally_votes(
            attempts, review_threshold, first_score, drop_threshold
        )
        action = "drop" if votes_bad >= votes_good else "keep"  # 平票剔除（过审优先）
        if override:
            action = "drop"
        best_text = _pick_best_text(attempts)
        reason = _build_reason(
            attempts, votes_bad, votes_good, action, best_text, drop_threshold,
            abstain=abstain, override=override,
        )
        result = VerifyResult(
            stem=clip.stem,
            attempts=attempts,
            votes_bad=votes_bad,
            votes_good=votes_good,
            action=action,
            best_text=best_text,
            reason=reason,
        )
        results[clip.stem] = result

        # 把最可信文本写回 Clip（下游 CSV/疑似清单能看到救回的台词）
        if best_text:
            best_attempt = _pick_best_attempt(attempts)
            if best_attempt is not None:
                clip.text = best_attempt.transcript
                clip.avg_logprob = float(best_attempt.detail.get("avg_logprob", clip.avg_logprob))
                clip.no_speech_prob = float(best_attempt.detail.get("no_speech_prob", clip.no_speech_prob))
                clip.asr_done = True
                clip.__dict__["verify_score"] = best_attempt.score
                clip.__dict__.pop("vad_empty", None)
                clip.__dict__.pop("asr_error", None)

    n_drop = sum(1 for r in results.values() if r.action == "drop")
    log(
        f"[复核] 完成 {len(results)} 句，判删 {n_drop} / 判留 {len(results) - n_drop}，"
        f"用时 {human_time(time.time() - t0)}"
    )
    return results


def _pick_best_attempt(attempts: Sequence[VerifyAttempt]) -> VerifyAttempt | None:
    """挑最可信的一轮：优先有文本的，再按 avg_logprob 高、分数低排序。"""
    ok = [a for a in attempts if a.asr_ok and a.transcript]
    if not ok:
        return None
    ok.sort(key=lambda a: (-float(a.detail.get("avg_logprob", -99.0)), a.score))
    return ok[0]


def _pick_best_text(attempts: Sequence[VerifyAttempt]) -> str:
    """多轮里给出的最可信文本（空文本句救回时用）。"""
    best = _pick_best_attempt(attempts)
    return best.transcript if best else ""


def _truncate(text: str, limit: int = 20) -> str:
    """示例文本截断（报告里不贴长原文）。"""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _build_reason(
    attempts: Sequence[VerifyAttempt],
    votes_bad: int,
    votes_good: int,
    action: str,
    best_text: str,
    drop_threshold: float,
    abstain: int = 0,
    override: str = "",
) -> str:
    """拼一句人看的理由：「多轮判别 2/3 判敏感」「转写恢复: …」。"""
    total = len(attempts)
    if total == 0:
        return "多轮判别未执行(无有效轮次)"
    n_empty = sum(1 for a in attempts if not a.asr_ok)
    parts = [f"多轮判别 {votes_bad}/{total} 判敏感"]
    if votes_bad == votes_good:
        parts.append("平票按剔除处理")
    if abstain:
        parts.append(f"{abstain} 轮空文本弃权(另一轮已恢复实质台词)")
    elif n_empty:
        parts.append(f"空文本{n_empty}次")
    if override:
        parts.append(override)
    # 第一轮是空文本、复核后拿到正常文本 → 明确标「转写恢复」
    first_empty = bool(attempts and not attempts[0].asr_ok)
    if best_text and (first_empty or (action == "keep" and attempts and attempts[0].score >= 1.5)):
        snapshot = _truncate(best_text)
        if snapshot:
            parts.append(f"转写恢复: {snapshot}")
    return "；".join(parts)


# --------------------------------------------------------------------------- #
# 合并回 Decision
# --------------------------------------------------------------------------- #
def apply_votes(
    decisions: Sequence[Decision], results: Mapping[str, VerifyResult]
) -> list[Decision]:
    """把表决结果合并回 `Decision` 列表（返回新列表，不改原对象）。

    * 参与复核的句子：``action`` 用表决结果（keep/drop），``score`` 取最有代表性的一轮
      （判删 → 各轮最高敏感分；判留 → 各轮最低分），``reasons`` 追加多轮理由，
      并挂两个动态属性供 CSV/疑似清单使用：``votes``（如 ``"2/3"``）与
      ``verify``（完整 :class:`VerifyResult`）；
    * 未参与复核的句子：原样复制（浅拷贝 Decision，reasons 列表新建）。
    """
    out: list[Decision] = []
    for d in decisions:
        res = results.get(d.stem)
        if res is None:
            out.append(Decision(clip=d.clip, action=d.action, score=d.score, reasons=list(d.reasons)))
            continue
        if res.attempts:
            scores = [a.score for a in res.attempts]
            score = (max(scores) if res.action == "drop" else min(scores))
        else:
            score = d.score
        merged = Decision(
            clip=d.clip,
            action=res.action,
            score=round(float(score), 2),
            reasons=list(d.reasons) + [res.reason],
        )
        merged.__dict__["votes"] = f"{res.votes_bad}/{res.total}" if res.total else ""
        merged.__dict__["verify"] = res
        out.append(merged)
    return out


def verify_stats(results: Mapping[str, VerifyResult]) -> dict:
    """汇总多轮判别统计（报告/日志用）。

    Returns:
        含句数、判留/判删、空文本救回数、误杀翻转数等字段的字典。
    """
    total = len(results)
    keep = sum(1 for r in results.values() if r.action == "keep")
    drop = total - keep
    rescued = 0  # 第一轮空文本、复核后有文本且判留
    rescued_drop = 0  # 第一轮空文本、复核后有文本但仍判删（说明拿到喘息文本）
    recovered_any = 0
    for r in results.values():
        if not r.attempts:
            continue
        first_empty = not r.attempts[0].asr_ok
        if first_empty and r.best_text:
            recovered_any += 1
            if r.action == "keep":
                rescued += 1
            else:
                rescued_drop += 1
    return {
        "total": total,
        "keep": keep,
        "drop": drop,
        "empty_first": sum(1 for r in results.values() if r.attempts and not r.attempts[0].asr_ok),
        "recovered_text": recovered_any,
        "rescued_keep": rescued,
        "recovered_but_drop": rescued_drop,
        "avg_rounds": round(sum(r.total for r in results.values()) / total, 2) if total else 0.0,
    }


__all__ = [
    "VerifyAttempt",
    "VerifyResult",
    "VerifyCache",
    "PASS_PRESETS",
    "DEFAULT_PASSES",
    "plan_suspects",
    "verify_suspects",
    "apply_votes",
    "verify_stats",
]
