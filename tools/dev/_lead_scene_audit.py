"""Lead 用：分析 filter_report.csv，看每个场景的可疑程度，辅助调阈值。

用法: PYTHONIOENCODING=utf-8 PYTHONPATH=.deps python tools/_lead_scene_audit.py <report.csv> [top]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    report = Path(sys.argv[1])
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rows = list(csv.DictReader(report.open(encoding="utf-8-sig")))
    print(f"总句数 {len(rows)}")
    print("action 分布:", dict(Counter(r["action"] for r in rows)))
    scores = sorted(float(r["score"]) for r in rows)
    def q(p: float) -> float:
        return scores[min(len(scores) - 1, int(p * len(scores)))]
    print(f"分数 p50={q(.5):.2f} p75={q(.75):.2f} p90={q(.9):.2f} p95={q(.95):.2f} p99={q(.99):.2f} max={scores[-1]:.2f}")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)

    stats = []
    for scene, rs in by_scene.items():
        n = len(rs)
        n_drop = sum(1 for r in rs if r["action"] == "drop")
        hi = sum(1 for r in rs if float(r["score"]) >= 2.5)
        empty = sum(1 for r in rs if not r["text"].strip())
        stats.append((hi / n, n_drop / n, n, scene, n_drop, hi, empty))

    print(f"\n场景数 {len(stats)}；按「高分句占比」Top{top}:")
    print(f"{'scene':10s} {'句数':>4s} {'drop':>4s} {'≥2.5':>5s} {'空文本':>5s}  示例")
    for frac, dfrac, n, scene, n_drop, hi, empty in sorted(stats, reverse=True)[:top]:
        samples = sorted(by_scene[scene], key=lambda r: -float(r["score"]))[:3]
        ex = " | ".join(f"{float(r['score']):.1f}:{(r['text'] or '(空)')[:26]}" for r in samples)
        print(f"{scene:10s} {n:4d} {n_drop:4d} {hi:5d} {empty:5d}  {ex}")

    # 场景连坐 / 自动整场判定的模拟
    for hi_thr, frac_thr, min_n in ((2.5, 0.5, 3), (2.0, 0.6, 3), (3.0, 0.35, 3)):
        auto = {s for frac, dfrac, n, s, n_drop, hi, empty in stats
                if n >= min_n and frac >= frac_thr and n_drop == 0}
        with_drop = {s for frac, dfrac, n, s, n_drop, hi, empty in stats if n_drop > 0}
        flagged = auto | with_drop
        kept = sum(1 for r in rows if r["action"] != "drop" and r["scene"] not in flagged)
        secs = sum(float(r["dur"]) for r in rows if r["action"] != "drop" and r["scene"] not in flagged)
        print(f"\n[规则 hi>={hi_thr} frac>={frac_thr:.0%} n>={min_n}] 自动判定敏感场景 {len(auto)} 个（另有 {len(with_drop)} 个已有显式 drop）"
              f" → 连带剔除后保留 {kept} 句 / {secs/3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
