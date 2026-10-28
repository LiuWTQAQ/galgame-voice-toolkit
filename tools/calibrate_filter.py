#!/usr/bin/env python
"""ASR + 过滤器的校准/验收脚本。

做三件事：
1. 从真实语音包里抽 N 句 → 跑 ASR（GPU 分块）→ 打分 → 三桶判定 → 打印统计。
2. 统计显式词命中率、漏检样例、疑似误杀样例、阈值扫描表，导出 CSV 供人工复核。
3. 在另一个作品（默认 E:\\etyyyyy）上抽 100 句交叉验证，检查分布是否一致。

用法::

    # 全流程（默认 E:\\mrxyy 300 句 + E:\\etyyyyy 100 句交叉验证）
    python tools/calibrate_filter.py

    # 只跑 150 句、不做交叉验证
    python tools/calibrate_filter.py --sample 150 --cross-sample 0

    # 不跑 ASR，只用已有缓存里的文本快速迭代词表
    python tools/calibrate_filter.py --no-asr

    # 换阈值看重分布
    python tools/calibrate_filter.py --drop-threshold 4.5 --review-threshold 2.0

依赖：workspace 下 .deps（faster-whisper/numpy），脚本会自动加 sys.path。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# --- 路径引导：允许直接 `python tools/calibrate_filter.py` 而不设 PYTHONPATH ----
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / ".deps", ROOT):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("VOICEPACK_DEPS", str(ROOT / ".deps"))

from voicepack import filter as vfilter  # noqa: E402
from voicepack import lexicon  # noqa: E402
from voicepack.scan import scan_folder  # noqa: E402
from voicepack.types import Clip  # noqa: E402
from voicepack.util import human_time, log, setup_console  # noqa: E402


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _cut(text: str, limit: int = 20) -> str:
    """示例文本只截断到 20 字以内，避免把露骨内容整句写进报告/日志。"""
    one = re.sub(r"\s+", " ", (text or "").strip())
    return one if len(one) <= limit else one[:limit] + "…"


def _sample(clips: list[Clip], n: int, seed: int) -> list[Clip]:
    """分层抽样：按场景均匀取样，再按原始顺序排序（分块 ASR 需要相邻句）。"""
    if n <= 0 or n >= len(clips):
        return list(clips)
    by_scene: dict[str, list[Clip]] = {}
    for c in clips:
        by_scene.setdefault(c.scene_key, []).append(c)
    keys = sorted(by_scene, key=lambda k: (by_scene[k][0].prefix, by_scene[k][0].scene))
    picked: list[Clip] = []
    per_scene = max(1, n // len(keys))
    for key in keys:
        group = by_scene[key]
        take = min(per_scene, len(group))
        picked.extend(group[:take] if len(group) <= take else random.Random(seed).sample(group, take))
    # 不够就随机补，超了就随机裁
    rng = random.Random(seed)
    if len(picked) < n:
        have = {c.stem for c in picked}
        rest = [c for c in clips if c.stem not in have]
        picked.extend(rng.sample(rest, min(n - len(picked), len(rest))))
    elif len(picked) > n:
        picked = rng.sample(picked, n)
    picked.sort(key=lambda c: (c.prefix, c.scene, c.idx))
    return picked


def _apply_cache(clips: list[Clip], cache_path: Path) -> int:
    """--no-asr 模式：只把已有缓存回填进 Clip，返回命中数。"""
    from voicepack.asr import AsrCache

    cache = AsrCache(cache_path).load()
    hits = 0
    for clip in clips:
        # --no-asr 只关心文本，跳过模型名/文件大小的严格校验
        rec = cache.entries.get(clip.name)
        if rec is None:
            continue
        clip.text = rec.text
        clip.avg_logprob = rec.avg_logprob
        clip.no_speech_prob = rec.no_speech_prob
        clip.asr_done = True
        clip.vad_empty = rec.vad_empty  # type: ignore[attr-defined]
        hits += 1
    return hits


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def _bucket_stats(decisions: list[vfilter.Decision]) -> dict:
    """三桶句数/时长/平均分 + 标记计数。"""
    n = len(decisions)
    total_dur = sum(d.clip.dur for d in decisions)
    out: dict = {"clips": n, "duration": round(total_dur, 1), "buckets": {}}
    for action in ("keep", "drop", "review"):
        items = [d for d in decisions if d.action == action]
        dur = sum(d.clip.dur for d in items)
        out["buckets"][action] = {
            "clips": len(items),
            "pct": round(len(items) * 100.0 / n, 1) if n else 0.0,
            "duration": round(dur, 1),
            "avg_score": round(sum(d.score for d in items) / len(items), 2) if items else 0.0,
        }
    scores = sorted(d.score for d in decisions)
    if scores:
        def q(p: float) -> float:
            return round(scores[min(len(scores) - 1, int(p * len(scores)))], 2)

        out["score_quantiles"] = {"p50": q(0.50), "p75": q(0.75), "p90": q(0.90), "p99": q(0.99),
                                  "max": round(scores[-1], 2)}
    out["markers"] = {
        "empty_text": sum(1 for d in decisions if d.clip.asr_done and not (d.clip.text or "").strip()),
        "vad_empty": sum(1 for d in decisions if getattr(d.clip, "vad_empty", False)),
        "asr_error": sum(1 for d in decisions if getattr(d.clip, "asr_error", None)),
        "not_transcribed": sum(1 for d in decisions if not d.clip.asr_done),
        "explicit_hits": sum(1 for d in decisions if lexicon.has_explicit(d.clip.text or "")),
    }
    return out


def _explicit_stats(decisions: list[vfilter.Decision]) -> dict:
    """显式词（organ/act/fluid）命中落在哪个桶 —— 衡量召回。"""
    hits = [d for d in decisions if lexicon.has_explicit(d.clip.text or "")]
    by_action: Counter[str] = Counter(d.action for d in hits)
    leaks = [d for d in hits if d.action == "keep"]
    return {
        "total": len(hits),
        "by_action": dict(by_action),
        "recall_drop_or_review": round(
            (by_action.get("drop", 0) + by_action.get("review", 0)) / len(hits), 3
        ) if hits else None,
        "leaked": [{"name": d.clip.name, "score": d.score, "text": _cut(d.clip.text)}
                   for d in leaks[:10]],
    }


def _reason_counter(decisions: list[vfilter.Decision], top: int = 15) -> list[tuple[str, int]]:
    """统计命中原因（去掉数字，避免同一原因被拆成很多条）。"""
    counter: Counter[str] = Counter()
    for d in decisions:
        for r in d.reasons:
            counter[re.sub(r"[\d.+\-]+", "#", r)] += 1
    return counter.most_common(top)


def _threshold_sweep(clips: list[Clip], base: dict) -> list[dict]:
    """固定 review 扫 drop、固定 drop 扫 review，看桶大小怎么变。"""
    rows: list[dict] = []
    for drop_t in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0):
        ds = vfilter.decide(clips, drop_threshold=drop_t, review_threshold=base["review"])
        c = Counter(d.action for d in ds)
        rows.append({"drop_threshold": drop_t, "review_threshold": base["review"],
                     "drop": c.get("drop", 0), "review": c.get("review", 0), "keep": c.get("keep", 0)})
    for review_t in (1.0, 1.5, 2.0, 2.5, 3.0):
        ds = vfilter.decide(clips, drop_threshold=base["drop"], review_threshold=review_t)
        c = Counter(d.action for d in ds)
        rows.append({"drop_threshold": base["drop"], "review_threshold": review_t,
                     "drop": c.get("drop", 0), "review": c.get("review", 0), "keep": c.get("keep", 0)})
    return rows


def _suspected_false_positives(decisions: list[vfilter.Decision], limit: int = 12) -> list[dict]:
    """drop 桶里既没有显式词、也没有娇喘/静音标记的句子 —— 可能是误杀，人工听一下。"""
    out: list[dict] = []
    for d in decisions:
        if d.action != "drop":
            continue
        text = d.clip.text or ""
        if lexicon.has_explicit(text) or getattr(d.clip, "vad_empty", False):
            continue
        moan, _ = lexicon.moan_features(text)
        if moan > 0 or not text.strip():
            continue
        out.append({"name": d.clip.name, "score": d.score, "text": _cut(text),
                    "reasons": " | ".join(d.reasons)[:120]})
        if len(out) >= limit:
            break
    return out


def _preview(decisions: list[vfilter.Decision], action: str, limit: int = 12) -> list[dict]:
    """给某个桶打印若干条示例（文本截断，便于人工抽查）。"""
    items = sorted((d for d in decisions if d.action == action), key=lambda d: d.score, reverse=True)
    return [
        {"name": d.clip.name, "score": d.score, "text": _cut(d.clip.text),
         "reasons": " | ".join(d.reasons)[:100]}
        for d in items[:limit]
    ]


# --------------------------------------------------------------------------- #
# 单数据集流程
# --------------------------------------------------------------------------- #
def _run_dataset(
    folder: Path,
    n: int,
    args: argparse.Namespace,
    cache_path: Path,
    out_dir: Path,
    tag: str,
) -> dict | None:
    """对一个语音包抽句、跑 ASR、过滤，返回统计 dict（并写 CSV/预览）。"""
    if not folder.is_dir():
        log(f"[校准] 跳过不存在的目录: {folder}")
        return None
    all_clips = scan_folder(folder)
    if not all_clips:
        log(f"[校准] 目录里没扫到 clip: {folder}")
        return None
    clips = _sample(all_clips, n, args.seed)
    log(f"[校准][{tag}] {folder} 共 {len(all_clips)} 句，抽样 {len(clips)} 句")

    t0 = time.time()
    if args.no_asr:
        hits = _apply_cache(clips, cache_path)
        log(f"[校准][{tag}] --no-asr：从缓存回填 {hits}/{len(clips)} 句")
    else:
        from voicepack.asr import transcribe_clips

        transcribe_clips(
            clips,
            model_name=args.model,
            device=args.device,
            workers=args.workers,
            cache_path=cache_path,
            chunked=not args.per_clip,
        )
        log(f"[校准][{tag}] ASR 用时 {time.time() - t0:.1f}s")

    decisions = vfilter.decide(
        clips,
        drop_threshold=args.drop_threshold,
        review_threshold=args.review_threshold,
    )
    stats = _bucket_stats(decisions)
    stats.update(
        {
            "folder": str(folder),
            "sampled": len(clips),
            "sample_total": len(all_clips),
            "asr": {
                "done": sum(1 for c in clips if c.asr_done),
                "empty": stats["markers"]["empty_text"],
                "seconds": round(time.time() - t0, 1),
            },
            "explicit": _explicit_stats(decisions),
            "top_reasons": _reason_counter(decisions),
            "suspected_false_positives": _suspected_false_positives(decisions),
            "preview_drop": _preview(decisions, "drop"),
            "preview_review": _preview(decisions, "review"),
            "scenes": vfilter.scene_table(decisions)[:12],
            "sweep": _threshold_sweep(clips, {
                "drop": args.drop_threshold, "review": args.review_threshold,
            }),
        }
    )

    csv_path = out_dir / f"report_{tag}.csv"
    vfilter.write_report(decisions, csv_path)
    stats["csv"] = str(csv_path)

    print()
    print(vfilter.summarize(decisions))
    print()
    print(_fmt_stats(stats))
    return stats


def _fmt_stats(s: dict) -> str:
    """把统计 dict 渲染成人类可读文本。"""
    lines: list[str] = []
    b = s["buckets"]
    lines.append(f"  [{s['folder']}] 抽样 {s['sampled']}/{s['sample_total']} 句，"
                 f"ASR 完成 {s['asr']['done']} 句，用时 {s['asr']['seconds']}s")
    lines.append(f"  桶: keep {b['keep']['clips']} | drop {b['drop']['clips']} | review {b['review']['clips']}")
    lines.append(f"  标记: {s['markers']}")
    if "score_quantiles" in s:
        lines.append(f"  分数分位: {s['score_quantiles']}")
    ex = s["explicit"]
    lines.append(f"  显式词命中: {ex['total']} 句，分布 {ex['by_action']}，"
                 f"召回(进 drop/review) {ex['recall_drop_or_review']}")
    for x in ex["leaked"]:
        lines.append(f"    漏检: {x['name']} {x['score']:.2f} 「{x['text']}」")
    lines.append(f"  原因 Top: {s['top_reasons'][:8]}")
    fps = s["suspected_false_positives"]
    lines.append(f"  疑似误杀(无显式词/无娇喘标记的 drop): {len(fps)} 句")
    for x in fps[:6]:
        lines.append(f"    ? {x['name']} {x['score']:.2f} 「{x['text']}」 <- {x['reasons']}")
    lines.append("  阈值扫描（drop 固定/扫 review；review 固定/扫 drop）:")
    for row in s["sweep"]:
        lines.append(f"    drop>={row['drop_threshold']} review>={row['review_threshold']}: "
                     f"drop {row['drop']} / review {row['review']} / keep {row['keep']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    p = argparse.ArgumentParser(description="ASR + 成人内容过滤器校准脚本")
    p.add_argument("--folder", default=r"E:\mrxyy", help="主数据集目录（默认 E:\\mrxyy）")
    p.add_argument("--sample", type=int, default=300, help="主数据集抽样句数（默认 300）")
    p.add_argument("--cross-folder", default=r"E:\etyyyyy", help="交叉验证目录（默认 E:\\etyyyyy）")
    p.add_argument("--cross-sample", type=int, default=100, help="交叉验证句数（默认 100，0=关闭）")
    p.add_argument("--seed", type=int, default=20241028, help="抽样随机种子")
    p.add_argument("--model", default="small", help="faster-whisper 模型名（默认 small）")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu（默认 auto）")
    p.add_argument("--workers", type=int, default=1, help="CPU 多进程数（GPU 下忽略）")
    p.add_argument("--per-clip", action="store_true", help="关闭分块批量，逐句转写")
    p.add_argument("--no-asr", action="store_true", help="不跑 ASR，只用缓存里的文本")
    p.add_argument("--cache", default=str(ROOT / ".calib" / "asr_cache.jsonl"), help="ASR 缓存路径")
    p.add_argument("--out", default=str(ROOT / ".calib"), help="输出目录")
    p.add_argument("--drop-threshold", type=float, default=vfilter.DROP_THRESHOLD)
    p.add_argument("--review-threshold", type=float, default=vfilter.REVIEW_THRESHOLD)
    p.add_argument("--json", action="store_true", help="把统计 JSON 打印到 stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    """入口：跑主数据集 + 可选交叉验证，写 CSV/JSON 报告。"""
    setup_console()
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    log(f"[校准] 模型={args.model} 设备={args.device} 分块={not args.per_clip} "
        f"drop>={args.drop_threshold} review>={args.review_threshold}")
    t0 = time.time()
    main_stats = _run_dataset(Path(args.folder), args.sample, args, cache_path, out_dir, "main")
    cross_stats = None
    if args.cross_sample > 0 and args.cross_folder:
        cross_stats = _run_dataset(
            Path(args.cross_folder), args.cross_sample, args, cache_path, out_dir, "cross"
        )

    payload = {"main": main_stats, "cross": cross_stats,
               "thresholds": {"drop": args.drop_threshold, "review": args.review_threshold},
               "elapsed_s": round(time.time() - t0, 1)}
    json_path = out_dir / "stats.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[校准] 统计写入 {json_path}（耗时 {time.time() - t0:.1f}s）")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
