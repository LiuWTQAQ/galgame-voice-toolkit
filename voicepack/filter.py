"""内容审核过滤：打分 + 三桶判定 + CSV 报告 + 场景聚合统计。

设计目标
--------
把 Galgame 角色语音里**成人内容**识别出来剔除，便于合成音频后过平台审核。
判据不是"看到敏感词就删"，而是加权累加 + 组合短语 + 非语言发声启发式：

- 单个暧昧词（気持ちいい / だめ / やめて / はぁ）权重低，单独出现只进 review，
  避免把日常对话误杀。
- 露骨解剖/行为/体液词权重高，单个即可 drop。
- 娇喘/喘息靠"文本形态 + ASR 置信度"联合判断：极短假名串、重复假名、
  VAD 把整句判成静音（``clip.vad_empty``）、低 ``avg_logprob``、高
  ``no_speech_prob`` 都会加分。
- 同一场景内多句命中时给整场景小幅加成（成人场景通常连续多句，日常场景不会）。

三桶
----
``score >= drop_threshold`` → drop；``>= review_threshold`` → review；否则 keep。
review 是留给人耳复核的中间桶，``write_report`` 导出 CSV，用户改完 action 列后
``read_report`` 读回来当 ``overrides`` 覆盖重跑。

优先级（高 → 低）：单句人工覆盖 > 场景强制剔除 > 场景强制保留 > 分数阈值。
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import lexicon
from .types import Clip, Decision
from .util import human_time

# 默认阈值（校准实测后确定，见 docs/filter_notes.md）
# drop=3.0：554 句真实数据上 3.0~4.0 档全部是呻吟/喊叫/成人台词，无正常台词误伤；
#           取 4.0 会漏掉「んっ、あっ」「うっ!うっ!…」这类纯呻吟（它们在 3.0~4.0）。
# review=1.5：低于 1.5 的样本里几乎没有可疑句，再降只会把「ちょっと」这类日常词送人工。
DROP_THRESHOLD = 3.0
REVIEW_THRESHOLD = 1.5

ACTIONS = ("keep", "drop", "review")

# CSV 列（机器可读，固定顺序）
CSV_FIELDS = ["action", "score", "scene", "idx", "name", "dur", "text", "reasons"]

# 用户手改 CSV 时可能写的中文/变体 → 标准 action
_ACTION_ALIASES: dict[str, str] = {
    "keep": "keep",
    "保留": "keep",
    "通过": "keep",
    "留": "keep",
    "drop": "drop",
    "剔除": "drop",
    "删除": "drop",
    "丢弃": "drop",
    "删": "drop",
    "review": "review",
    "待确认": "review",
    "复核": "review",
    "复审": "review",
    "?": "review",
}

# ASR 置信度启发式的阈值
_LOW_LOGPROB = -0.9  # 低于此视为"识别置信度低"
_VERY_LOW_LOGPROB = -1.5
_HIGH_NOSPEECH = 0.6
_VERY_HIGH_NOSPEECH = 0.85


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _scene_key(value: str | int | tuple[str, int]) -> str:
    """把 ``"noz_001"`` / ``"noz001"`` / ``("noz", 1)`` 归一到 ``Clip.scene_key`` 格式。"""
    if isinstance(value, tuple) and len(value) == 2:
        return f"{str(value[0]).lower()}{int(value[1]):03d}"
    if isinstance(value, int):
        return f"{value:03d}"
    raw = str(value).strip().lower()
    m = re.fullmatch(r"([a-z]+)[\s_\-]*0*(\d+)", raw)
    if m:
        return f"{m.group(1)}{int(m.group(2)):03d}"
    return raw


def _norm_action(value: str) -> str:
    """把用户写的 action 变体归一成 keep/drop/review。

    Raises:
        ValueError: 无法识别的 action。
    """
    raw = str(value).strip().lower()
    action = _ACTION_ALIASES.get(raw)
    if action is None:
        raise ValueError(f"无法识别的 action: {value!r}（支持 keep/drop/review/保留/剔除/待确认）")
    return action


def _clip_markers(clip: Clip) -> list[str]:
    """非分数类的标记（便于 CSV/grep 排查）。"""
    markers: list[str] = []
    if not clip.asr_done:
        markers.append("未转写")
    if getattr(clip, "vad_empty", False):
        markers.append("VAD吞句")
    if getattr(clip, "asr_error", None):
        markers.append("转写失败")
    if clip.asr_done and not clip.text.strip():
        markers.append("空文本")
    return markers


# --------------------------------------------------------------------------- #
# 单句打分
# --------------------------------------------------------------------------- #
def score_clip(clip: Clip) -> tuple[float, list[str]]:
    """给一句台词打分，返回 ``(score, reasons)``。

    分数来源（全部累加）：

    ==========================  ==========================================
    文本词表/短语/娇喘模式      ``lexicon.score_text``（主体分值）
    VAD 吞句                    +1.8  （疑似娇喘/气声被 VAD 判为静音）
    空文本 <1s / >=1s           +0.4 / +1.6（短气口放行；有声无文本按可疑送复核）
    识别置信度低                +0.8 ~ +1.2（``avg_logprob < -0.9``）
    高 no_speech_prob           +0.6 ~ +1.0
    低置信 + 短暂 + 高静音      +0.7  （呼吸/喘息典型形态）
    文本极短 + 娇喘特征 + 高静音 +0.5
    未转写                      → 直接 review（读不到文本不能打包票）
    ==========================  ==========================================

    Args:
        clip: 已尽量转写过的句子（未转写也能打分，只是无法看文本）。

    Returns:
        ``(score, reasons)``；score 越高越可能成人内容。
    """
    if not clip.asr_done:
        return REVIEW_THRESHOLD, ["未转写(未跑 ASR)，保守送复核"]

    score = 0.0
    reasons: list[str] = []
    text = (clip.text or "").strip()

    if getattr(clip, "asr_error", None):
        reasons.append(f"转写失败: {getattr(clip, 'asr_error')}")

    if text:
        text_score, text_reasons = lexicon.score_text(text)
        score += text_score
        reasons.extend(text_reasons)
    elif getattr(clip, "vad_empty", False):
        # VAD 把整句判成静音：Galgame 里这几乎一定是纯喘息/气声，最该命中的形态之一
        score += 1.8
        reasons.append("VAD 判定整句静音(疑似娇喘/气声)")
    else:
        # 空文本按长度分档：0.3~1s 的气口/静音很常见，全送复核会淹掉真正可疑的长句
        if clip.dur < 1.0:
            score += 0.4
            reasons.append("极短空文本(<1s，更像气口/静音)")
        else:
            # 有声音但完全转写不出（实测这些片段音量正常，是喊叫/喘息/非语音），
            # 对「过审」场景一律送复核：宁可多看几行 CSV，不可漏删。
            score += 1.6
            reasons.append("空文本(>=1s，有声但转写不出，疑似喘息/喊叫)")

    # --- ASR 置信度 / 非语言声音启发式 -------------------------------------
    if text:
        if clip.avg_logprob < _LOW_LOGPROB:
            score += 1.2 if clip.avg_logprob < _VERY_LOW_LOGPROB else 0.8
            reasons.append(f"识别置信度低(avg_logprob={clip.avg_logprob:.2f})")
        if clip.no_speech_prob >= _HIGH_NOSPEECH:
            score += 1.0 if clip.no_speech_prob >= _VERY_HIGH_NOSPEECH else 0.6
            reasons.append(f"高非语音概率(no_speech_prob={clip.no_speech_prob:.2f})")

        # 呼吸/喘息形态：短、低置信、高非语音
        if (
            clip.dur <= 2.5
            and clip.avg_logprob < _LOW_LOGPROB - 0.1
            and clip.no_speech_prob >= 0.4
        ):
            score += 0.7
            reasons.append("短暂+低置信+高静音概率(疑似喘息)")

        # 极短文本 + 娇喘特征 + 非语音概率高 → 加一点
        if len(lexicon.normalize(text)) <= 6 and clip.no_speech_prob >= 0.4:
            moan_score, moan_reasons = lexicon.moan_features(text)
            if moan_score > 0.0:
                score += 0.5
                reasons.append("极短语音+高静音概率(疑似喘息)")

    return round(score, 2), reasons


# --------------------------------------------------------------------------- #
# 场景加成
# --------------------------------------------------------------------------- #
def _scene_bonus(
    scored: Sequence[tuple[Clip, float]], review_threshold: float
) -> dict[str, float]:
    """同场景多句命中 → 整场景小幅加成。

    Galgame 的成人场景通常连续若干句都是，而日常场景偶尔出现一个「気持ちいい」
    不会成串。命中句数 ≥2 且占比 ≥25% 时，给场景内每句加
    ``min(1.2, 0.5 * (命中句数 - 1))``，让"夹在中间的低分句"也能进 review/drop。

    Returns:
        ``{scene_key: bonus}``（未触发加成的场景不在字典里）。
    """
    by_scene: dict[str, list[float]] = defaultdict(list)
    for clip, score in scored:
        by_scene[clip.scene_key].append(score)

    bonus: dict[str, float] = {}
    for key, scores in by_scene.items():
        high = sum(1 for s in scores if s >= review_threshold)
        if high >= 2 and high / len(scores) >= 0.25:
            bonus[key] = min(1.2, 0.5 * (high - 1))
    return bonus


# --------------------------------------------------------------------------- #
# 批量判定
# --------------------------------------------------------------------------- #
def decide(
    clips: Sequence[Clip],
    drop_threshold: float = DROP_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
    drop_scenes: Iterable[str | int | tuple[str, int]] = (),
    keep_scenes: Iterable[str | int | tuple[str, int]] = (),
    overrides: Mapping[str, str] | None = None,
) -> list[Decision]:
    """对整批句子做三桶判定。

    优先级（高 → 低）：单句人工覆盖 > 场景强制剔除 > 场景强制保留 > 分数阈值。

    Args:
        clips: 已转写的句子。
        drop_threshold: ``score >=`` 判定 drop，默认 3.0。
        review_threshold: ``score >=`` 判定 review，默认 1.5。
        drop_scenes: 整场景强制剔除，元素形如 ``"noz001"`` / ``("noz", 1)``。
        keep_scenes: 整场景强制保留（与 drop_scenes 冲突时以 drop 为准，过审优先）。
        overrides: ``{stem 或 name: action}``，一般来自 ``read_report``；
            单句级、优先级最高。

    Returns:
        与 clips 等长、同序的 Decision 列表。
    """
    overrides = overrides or {}
    drop_set = {_scene_key(s) for s in drop_scenes}
    keep_set = {_scene_key(s) for s in keep_scenes}
    # 冲突时 drop 优先（过审优先），keep 里的重叠场景直接丢掉
    keep_set -= drop_set

    scored = [(clip, *score_clip(clip)) for clip in clips]
    bonuses = _scene_bonus([(c, s) for c, s, _ in scored], review_threshold)

    decisions: list[Decision] = []
    for clip, score, reasons in scored:
        reasons = list(reasons)
        bonus = bonuses.get(clip.scene_key, 0.0)
        if bonus:
            score = round(score + bonus, 2)
            reasons.append(f"同场景多句命中 +{bonus:.1f}")

        action = (
            "drop"
            if score >= drop_threshold
            else "review"
            if score >= review_threshold
            else "keep"
        )

        if clip.scene_key in drop_set:
            action = "drop"
            reasons.append("场景强制剔除")
        elif clip.scene_key in keep_set:
            action = "keep"
            reasons.append("场景强制保留")

        override = overrides.get(clip.stem) or overrides.get(clip.name)
        if override:
            action = _norm_action(override)
            reasons.append(f"人工覆盖={action}")

        markers = _clip_markers(clip)
        if markers:
            reasons.extend(f"[{m}]" for m in markers)

        decisions.append(
            Decision(clip=clip, action=action, score=round(score, 2), reasons=reasons)
        )
    return decisions


# --------------------------------------------------------------------------- #
# CSV 报告
# --------------------------------------------------------------------------- #
def write_report(decisions: Sequence[Decision], csv_path: str | Path) -> Path:
    """导出 CSV 供人工复核。

    列（固定顺序）：``action,score,scene,idx,name,dur,text,reasons``。
    用 utf-8-sig 编码（Excel 直接打开中文不乱码）。用户改 ``action`` 列后，
    用 ``read_report`` 读回并作为 ``decide(..., overrides=...)`` 覆盖重跑。

    Args:
        decisions: ``decide`` 的结果。
        csv_path: 输出路径（父目录自动创建）。

    Returns:
        实际写入的路径。
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for d in decisions:
            clip = d.clip
            writer.writerow(
                {
                    "action": d.action,
                    "score": f"{d.score:.2f}",
                    "scene": clip.scene_key,
                    "idx": clip.idx,
                    "name": clip.name,
                    "dur": f"{clip.dur:.2f}",
                    "text": clip.text or "",
                    "reasons": " | ".join(d.reasons),
                }
            )
    return path


def read_report(csv_path: str | Path) -> dict[str, str]:
    """读取人工复核后的 CSV，返回 ``{stem: action}``。

    支持中文 action（保留/剔除/待确认），空 action 或无法识别时抛 ``ValueError``
    —— 宁可报错也不静默丢用户的修改。

    Args:
        csv_path: ``write_report`` 生成的 CSV（``action`` 列可被手改）。

    Returns:
        ``{stem: "keep"|"drop"|"review"}``，可直接传给 ``decide(overrides=...)``。
    """
    path = Path(csv_path)
    result: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头: {path}")
        if "action" not in reader.fieldnames:
            raise ValueError(f"CSV 缺少 action 列: {path}")
        if "stem" not in reader.fieldnames and "name" not in reader.fieldnames:
            raise ValueError(f"CSV 缺少 name/stem 列: {path}")
        for line_no, row in enumerate(reader, start=2):
            stem = (row.get("stem") or "").strip()
            if not stem:
                stem = Path((row.get("name") or "").strip()).stem
            if not stem:
                continue
            raw = (row.get("action") or "").strip()
            try:
                result[stem] = _norm_action(raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} {exc}") from None
    return result


# --------------------------------------------------------------------------- #
# 场景聚合
# --------------------------------------------------------------------------- #
def scene_table(decisions: Sequence[Decision]) -> list[dict]:
    """按 ``prefix + scene`` 聚合统计，供人工判断"整个场景是否要剔"。

    Returns:
        每个场景一个 dict（按 prefix、scene 编号自然排序），键为：
        ``scene``（scene_key）、``prefix``、``scene_no``、``clips``、``keep``、
        ``drop``、``review``、``avg_score``、``avg_dur``、``total_dur``、
        ``drop_ratio``、``keep_dur``。
    """
    groups: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        groups[d.clip.scene_key].append(d)

    rows: list[dict] = []
    for key, items in groups.items():
        n = len(items)
        clip0 = items[0].clip
        drop = sum(1 for d in items if d.action == "drop")
        review = sum(1 for d in items if d.action == "review")
        keep = n - drop - review
        total_dur = sum(d.clip.dur for d in items)
        rows.append(
            {
                "scene": key,
                "prefix": clip0.prefix,
                "scene_no": clip0.scene,
                "clips": n,
                "keep": keep,
                "drop": drop,
                "review": review,
                "avg_score": round(sum(d.score for d in items) / n, 2),
                "avg_dur": round(total_dur / n, 2),
                "total_dur": round(total_dur, 1),
                "drop_ratio": round(drop / n, 3),
                "keep_dur": round(sum(d.clip.dur for d in items if d.action == "keep"), 1),
            }
        )
    rows.sort(key=lambda r: (r["prefix"], r["scene_no"]))
    return rows


# --------------------------------------------------------------------------- #
# 中文摘要
# --------------------------------------------------------------------------- #
def summarize(decisions: Sequence[Decision]) -> str:
    """生成中文统计摘要（控制台/文档用）。

    包含三桶句数与时长、平均分、显式词命中数、VAD 吞句数、最"危险"的场景 Top5。
    """
    n = len(decisions)
    if n == 0:
        return "【过滤统计】没有句子。"

    buckets: dict[str, list[Decision]] = {a: [] for a in ACTIONS}
    for d in decisions:
        buckets.setdefault(d.action, []).append(d)

    total_dur = sum(d.clip.dur for d in decisions)
    lines: list[str] = []
    lines.append(
        f"【过滤统计】共 {n} 句 / {human_time(total_dur)}，"
        f"平均时长 {total_dur / n:.2f}s"
    )
    for action, label in (("keep", "保留"), ("drop", "剔除"), ("review", "待确认")):
        items = buckets.get(action, [])
        dur = sum(d.clip.dur for d in items)
        pct = len(items) * 100.0 / n
        score = sum(d.score for d in items) / len(items) if items else 0.0
        lines.append(
            f"  {label:3s}({action:6s}) {len(items):5d} 句 ({pct:5.1f}%)  "
            f"{human_time(dur):>10s}  平均分 {score:.2f}"
        )

    kept_dur = sum(d.clip.dur for d in buckets.get("keep", []))
    lines.append(
        f"  剔除后保留音频约 {human_time(kept_dur)}"
        f"（原 {human_time(total_dur)}，减少 {100 * (1 - kept_dur / total_dur) if total_dur else 0:.1f}%）"
    )

    explicit = [d for d in decisions if lexicon.has_explicit(d.clip.text or "")]
    vad_empty = [d for d in decisions if getattr(d.clip, "vad_empty", False)]
    empty = [d for d in decisions if d.clip.asr_done and not (d.clip.text or "").strip()]
    errors = [d for d in decisions if getattr(d.clip, "asr_error", None)]
    lines.append(
        f"  显式词命中 {len(explicit)} 句 | VAD 吞句 {len(vad_empty)} 句 | "
        f"空文本 {len(empty)} 句 | 转写失败 {len(errors)} 句"
    )

    # 最危险的场景 Top5（按 drop 句数、平均分）
    rows = [r for r in scene_table(decisions) if r["drop"] or r["review"]]
    rows.sort(key=lambda r: (r["drop"], r["avg_score"]), reverse=True)
    if rows:
        lines.append("  高危场景 Top5:")
        for r in rows[:5]:
            lines.append(
                f"    {r['scene']}: {r['clips']} 句 剔除{r['drop']} 待确认{r['review']} "
                f"平均分 {r['avg_score']:.2f}"
            )

    overridden = sum(1 for d in decisions if "人工覆盖" in " ".join(d.reasons))
    if overridden:
        lines.append(f"  人工覆盖 {overridden} 句（来自 CSV action 列）")

    review = buckets.get("review", [])
    if review:
        lines.append("  建议：先听 review 桶里的高分句，确认后回写 CSV 再重跑。")
    return "\n".join(lines)
