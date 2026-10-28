"""Command line interface for the Yuzusoft voice-pack pipeline."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import media, scan as scan_mod, util
from .types import Clip, Decision
from .util import human_size, human_time, log, setup_console

VERIFIED_REPORT = "filter_report_verified.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_out(folder: Path, out: str | None) -> Path:
    path = Path(out) if out else folder / "voicepack_out"
    return util.ensure_dir(path)


def _find_cover_image(folder: Path, image: str | None) -> Path | None:
    """封面图：优先 --image，其次文件夹里自带的 cover/cg 图；都没有就返回 None。"""
    if image:
        p = Path(image)
        if not p.is_file():
            raise FileNotFoundError(f"找不到封面图: {p}")
        return p
    for name in ("cover.png", "cover.jpg", "cover.jpeg", "cg.png", "cg.jpg", "CG.png", "CG.jpg"):
        p = folder / name
        if p.is_file():
            return p
    return None


def _title_for(folder: Path, clips: list[Clip], title: str | None) -> str:
    if title:
        return title
    prefixes = sorted({c.prefix for c in clips})
    return f"{folder.name} ({'/'.join(prefixes)})"


def _print_scan(clips: list[Clip], folder: Path, gap: float) -> None:
    stats = getattr(scan_mod.scan_folder, "last_stats", {})
    log(f"[扫描] {folder}")
    log(f"  ogg 文件  : {stats.get('total_files', '?')}")
    log(f"  已排除 ref: {stats.get('ref_skipped', 0)}")
    if stats.get("odd_skipped"):
        log(f"  名称未识别: {stats.get('odd_skipped')} (已跳过)")
    log(scan_mod.summarize(clips, gap=gap))


def _load_cache(clips: list[Clip], cache_path: Path) -> int:
    """Restore ASR results from a JSONL cache. Returns how many clips were loaded."""
    if not cache_path.is_file():
        return 0

    records: dict[str, dict] = {}
    with open(cache_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["name"]] = rec
            except Exception:
                continue
    n = 0
    for clip in clips:
        rec = records.get(clip.name)
        if rec is None:
            continue
        clip.text = rec.get("text", "")
        clip.avg_logprob = float(rec.get("avg_logprob", 0.0))
        clip.no_speech_prob = float(rec.get("no_speech_prob", 0.0))
        clip.asr_done = True
        n += 1
    return n


def _bar(done: int, total: int, started: float, label: str) -> None:
    total = max(total, 1)
    pct = done * 100.0 / total
    el = time.time() - started
    rate = done / el if el > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    log(f"  {label} {done}/{total} ({pct:5.1f}%)  {rate:5.2f} 句/秒  剩余约 {human_time(eta)}")


# --------------------------------------------------------------------------- #
# sub-commands
# --------------------------------------------------------------------------- #
def cmd_scan(args) -> int:
    folder = Path(args.folder)
    clips = scan_mod.scan_folder(folder)
    _print_scan(clips, folder, args.gap)
    return 0


def cmd_asr(args) -> int:
    from . import asr

    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    clips = scan_mod.scan_folder(folder)
    _print_scan(clips, folder, args.gap)

    cache = out_dir / f"asr_{args.model}.jsonl"
    loaded = _load_cache(clips, cache)
    todo = [c for c in clips if not c.asr_done]
    log(f"\n[转写] 模型={args.model} device={args.device} workers={args.workers}")
    log(f"  缓存命中 {loaded} / {len(clips)}，本次需转写 {len(todo)} 句")
    if todo:
        t0 = time.time()
        asr.transcribe_clips(
            todo,
            model_name=args.model,
            device=args.device,
            workers=args.workers,
            cache_path=cache,
        )
        el = time.time() - t0
        log(f"  完成，用时 {human_time(el)}")
    return 0


def _scene_tokens(raw: str, clips: list[Clip]) -> list[str]:
    """'12,15' -> ['noz012','noz015'] (bare numbers are expanded with the folder prefix)."""
    prefixes = sorted({c.prefix.lower() for c in clips}) or [""]
    out: list[str] = []
    for token in (raw or "").replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            out.extend(f"{p}{int(token):03d}" for p in prefixes)
        else:
            out.append(token)
    return out


CASCADE_MODES = ("neighbour", "scene", "off")
SOFT_SCORE = 2.0      # 相邻扩散时「可疑线」
HARD_SCORE = 2.5      # 连续成片时「可疑线」
RUN_MIN = 3           # 连续几句可疑就整段剔除
SCENE_MIN_CLIPS = 3
SCENE_HI_FRAC = 0.5


def _apply_cascade(args, decisions: list[Decision]) -> None:
    """在逐句判定之后做「上下文扩散」，把夹在敏感台词旁边的娇喘也一起剔掉。

    neighbour：以 drop 句为种子向左右扩散（只吞并分数 >= SOFT_SCORE 的邻居），
               另外把连续 RUN_MIN 句分数 >= HARD_SCORE 的段落整段剔除。
    scene：   某个场景只要有一句确认剔除，整场（含高分片段）全部剔除，最保险但可能误删剧情。
    off：     只信逐句判定。
    """
    mode = getattr(args, "cascade", "neighbour")
    if mode == "off":
        return
    by_scene: dict[str, list[Decision]] = {}
    for d in decisions:
        by_scene.setdefault(d.clip.scene_key, []).append(d)

    touched: set[str] = set()
    if mode == "scene":
        for key, ds in by_scene.items():
            drops = sum(1 for d in ds if d.action == "drop")
            hi = sum(1 for d in ds if d.score >= HARD_SCORE)
            if drops or (len(ds) >= SCENE_MIN_CLIPS and hi / len(ds) >= SCENE_HI_FRAC):
                touched.add(key)
        n = 0
        for d in decisions:
            if d.clip.scene_key in touched and d.action != "drop":
                d.action = "drop"
                d.reasons.append("整场敏感，保守剔除")
                n += 1
        if touched:
            log(f"  整场过滤：{len(touched)} 个场景判定为敏感，{n} 句追加剔除")
        return

    # neighbour
    seeds: set[str] = {d.clip.stem for d in decisions if d.action == "drop"}
    for _key, ds in by_scene.items():
        ds.sort(key=lambda d: d.clip.idx)
        run_start = None
        for i, d in enumerate(ds):
            if d.score >= HARD_SCORE:
                if run_start is None:
                    run_start = i
                if i - run_start + 1 >= RUN_MIN:
                    seeds.update(x.clip.stem for x in ds[run_start: i + 1])
            else:
                run_start = None
    changed = True
    guard = 0
    while changed and guard < 20:
        changed = False
        guard += 1
        for key, ds in by_scene.items():
            for i, d in enumerate(ds):
                if d.clip.stem not in seeds:
                    continue
                for j in (i - 1, i + 1):
                    if 0 <= j < len(ds):
                        nb = ds[j]
                        if nb.clip.stem not in seeds and nb.score >= SOFT_SCORE:
                            seeds.add(nb.clip.stem)
                            changed = True
    n = 0
    for d in decisions:
        if d.action != "drop" and d.clip.stem in seeds:
            d.action = "drop"
            d.reasons.append("邻近敏感台词/连续发声，保守剔除")
            n += 1
    if n:
        log(f"  上下文扩散：{n} 句相邻可疑句追加剔除")


def _resolve_decisions_file(args, out_dir: Path) -> Path | None:
    """--decisions 没给时，如果存在 refine 过的最终清单就优先用它。"""
    if getattr(args, "decisions", None):
        return Path(args.decisions)
    verified = out_dir / VERIFIED_REPORT
    if verified.is_file():
        log(f"  使用多轮判别后的清单: {verified.name}")
        return verified
    return None


def _decisions_from_cache(args, clips: list[Clip], out_dir: Path) -> list[Decision]:
    """读转写缓存 -> 逐句判定 -> 上下文扩散。"""
    from . import filter as filter_mod

    cache = out_dir / f"asr_{args.model}.jsonl"
    loaded = _load_cache(clips, cache)
    if loaded == 0:
        raise RuntimeError(
            "还没有转写结果，请先运行 asr（或去掉 --no-filter 直接跑 all）。"
        )
    overrides = None
    decisions_file = _resolve_decisions_file(args, out_dir)
    if decisions_file:
        overrides = filter_mod.read_report(decisions_file)
        log(f"  已读入人工清单 {len(overrides)} 条 ({decisions_file.name})")
    decisions = filter_mod.decide(
        clips,
        drop_scenes=_scene_tokens(args.drop_scenes, clips),
        keep_scenes=_scene_tokens(args.keep_scenes, clips),
        overrides=overrides,
    )
    _apply_cascade(args, decisions)
    return decisions


def _keep_clips(decisions: list[Decision], strict: bool) -> list[Clip]:
    return [
        d.clip for d in decisions if d.action == "keep" or (d.action == "review" and not strict)
    ]


def _suspect_decisions(decisions: list[Decision]) -> list[Decision]:
    """被剔除 + 待人工确认的句子（合成"疑似清单 mp3"用）。"""
    return [d for d in decisions if d.action != "keep"]


def cmd_filter(args) -> int:
    from . import filter as filter_mod

    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    clips = scan_mod.scan_folder(folder)
    decisions = _decisions_from_cache(args, clips, out_dir)
    report = out_dir / "filter_report.csv"
    filter_mod.write_report(decisions, report)
    summary = filter_mod.summarize(decisions)
    (out_dir / "filter_summary.txt").write_text(summary, encoding="utf-8")
    log("\n" + summary)
    log(f"\n清单已写入: {report}  (可人工修改 action 列后用 --decisions 指定)")
    return 0


# --------------------------------------------------------------------------- #
# 多轮判别（refine）
# --------------------------------------------------------------------------- #
def _do_refine(args, clips: list[Clip], out_dir: Path) -> Path | None:
    """对可疑句做多轮独立判别 + 多数表决，写出 filter_report_verified.csv。"""
    from . import filter as filter_mod, verify

    decisions = _decisions_from_cache(args, clips, out_dir)
    base_report = out_dir / "filter_report.csv"
    filter_mod.write_report(decisions, base_report)

    suspects = verify.plan_suspects(decisions)
    log(f"\n[多轮判别] 可疑句 {len(suspects)} 句（剔除 {sum(1 for d in decisions if d.action == 'drop')}"
        f" + 待确认 {sum(1 for d in decisions if d.action == 'review')}）"
        f"，每句 {args.passes} 轮独立识别")
    if not suspects:
        log("  没有可疑句，跳过")
        return base_report

    t0 = time.time()
    state = {"t": t0, "last": 0.0}

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - state["last"] < 5 and done != total:
            return
        state["last"] = now
        _bar(done, total, t0, "复核")

    cache = out_dir / f"verify_{args.model}.jsonl"
    results = verify.verify_suspects(
        suspects,
        model_name=args.model,
        second_model=args.second_model,
        device=args.device,
        passes=args.passes,
        cache_path=cache,
        progress=progress,
    )
    decisions = verify.apply_votes(decisions, results)

    report = out_dir / VERIFIED_REPORT
    filter_mod.write_report(decisions, report)
    summary = filter_mod.summarize(decisions)
    (out_dir / "filter_summary_verified.txt").write_text(summary, encoding="utf-8")
    log("\n" + summary)
    log(f"[多轮判别] 完成，用时 {human_time(time.time() - t0)}；最终清单: {report}")
    return report


def cmd_refine(args) -> int:
    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    clips = scan_mod.scan_folder(folder)
    _print_scan(clips, folder, args.gap)
    report = _do_refine(args, clips, out_dir)
    if report is None:
        return 2
    return 0


# --------------------------------------------------------------------------- #
# 疑似清单 mp3
# --------------------------------------------------------------------------- #
def _write_suspect_mp3(
    args,
    decisions: list[Decision],
    out_dir: Path,
    base: str,
) -> Path | None:
    suspects = _suspect_decisions(decisions)
    if not suspects:
        log("[疑似清单] 没有可疑句，跳过")
        return None
    mp3 = out_dir / f"{base}_suspect.mp3"
    csv_path = out_dir / f"{base}_suspect.csv"
    meta = [
        {
            "action": d.action,
            "score": d.score,
            "votes": getattr(d, "votes", ""),
            "text": d.clip.text,
            "reasons": "; ".join(d.reasons),
        }
        for d in suspects
    ]
    media.build_suspect_mp3(
        [d.clip for d in suspects], mp3, csv_path, meta=meta, gap=args.gap, bitrate=args.bitrate
    )
    log(
        f"[疑似清单] {mp3} ({human_size(mp3.stat().st_size)}, "
        f"{human_time(media.probe_duration(mp3))}, {len(suspects)} 句) + {csv_path.name}"
    )
    return mp3


def cmd_suspect(args) -> int:
    """只根据已有清单重建"疑似句 mp3"，不重新合成主音频（快）。"""
    from . import filter as filter_mod

    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    clips = scan_mod.scan_folder(folder)
    decisions = _decisions_from_cache(args, clips, out_dir)
    filter_mod.write_report(decisions, out_dir / "filter_report.csv")
    base = util.safe_filename(args.title or folder.name)
    return 0 if _write_suspect_mp3(args, decisions, out_dir, base) else 1


# --------------------------------------------------------------------------- #
# 合成
# --------------------------------------------------------------------------- #
def cmd_build(args) -> int:
    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    clips = scan_mod.scan_folder(folder)
    _print_scan(clips, folder, args.gap)
    return _do_build(args, folder, clips, out_dir)


def _render_cover(
    out_dir: Path,
    image: Path,
    title: str,
    n_kept: int,
    total_audio: float,
    cover_mode: str = "crop",
) -> Path:
    from . import cover

    cover_png = out_dir / "cover.png"
    cover.render_cover(
        image,
        cover_png,
        title=title,
        subtitle=f"语音合集 · {n_kept} 句 · {human_time(total_audio)}",
        cover_mode=cover_mode,
    )
    log(f"[封面] {cover_png}  (来自 {image.name}, 模式 {cover_mode})")
    return cover_png


def _do_build(args, folder: Path, clips: list[Clip], out_dir: Path) -> int:
    if args.no_filter:
        keep = clips
        decisions: list[Decision] = []
        log("\n[过滤] 已跳过（--no-filter），全部保留")
    else:
        from . import filter as filter_mod

        decisions = _decisions_from_cache(args, clips, out_dir)
        keep = _keep_clips(decisions, args.strict)
        summary = filter_mod.summarize(decisions)
        (out_dir / "filter_summary.txt").write_text(summary, encoding="utf-8")
        filter_mod.write_report(decisions, out_dir / "filter_report.csv")
        log("\n" + summary)
        log(f"[过滤] 保留 {len(keep)} / {len(clips)} 句，剔除 {len(clips) - len(keep)} 句"
            + ("（strict 模式：待确认也剔除）" if args.strict else ""))

    if not keep:
        log("\n[错误] 过滤后没有剩下任何句子，请检查阈值。")
        return 2

    title = _title_for(folder, clips, args.title)
    base = util.safe_filename(args.title or folder.name)

    # 1) 疑似清单 mp3（先做，快；音频坏不了主产物）
    if not args.no_suspect_mp3 and decisions:
        try:
            _write_suspect_mp3(args, decisions, out_dir, base)
        except Exception as exc:  # noqa: BLE001
            log(f"[疑似清单] 生成失败（不影响主流程）: {exc}")

    # 2) 主音轨
    master = out_dir / "tmp" / "master.flac"
    total_audio = sum(c.dur for c in keep) + max(0, len(keep) - 1) * args.gap

    log(f"\n[音轨] {len(keep)} 句 + {args.gap}s 停顿 ≈ {human_time(total_audio)}")
    t0 = time.time()
    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] < 5:
            return
        last[0] = now
        pct = done * 100 // max(total, 1)
        log(f"  拼接 {done}/{total} ({pct}%)")

    media.build_master(keep, master, gap=args.gap, progress=progress)
    log(f"  无损母带完成: {master} ({human_size(master.stat().st_size)}, 用时 {human_time(time.time() - t0)})")

    split = getattr(args, "split_minutes", 0) or 0
    cover_png = None
    image = None if args.mp3_only else _find_cover_image(folder, args.image)
    if image is not None:
        cover_png = _render_cover(
            out_dir, image, title, len(keep), total_audio, getattr(args, "cover_mode", "crop")
        )
    elif not args.mp3_only:
        log("[封面] 未提供封面图（--image），本次只输出 MP3；提供 CG 图后重跑 build 即可出 MP4")

    if split > 0:
        log(f"\n[切片] 按 {split:g} 分钟分段输出（平台单集时长限制用）")
        for old in list(out_dir.glob(f"{base}_part_*.mp3")) + list(out_dir.glob(f"{base}_part_*.mp4")):
            old.unlink(missing_ok=True)
        parts = media.split_mp3(master, out_dir, base, split, bitrate=args.bitrate)
        for p in parts:
            log(f"[MP3 ] {p} ({human_size(p.stat().st_size)}, {human_time(media.probe_duration(p))})")
        if cover_png is not None:
            vparts = media.split_mp4(
                cover_png, master, out_dir, base, split, fps=args.fps, crf=args.crf
            )
            for p in vparts:
                log(f"[MP4 ] {p} ({human_size(p.stat().st_size)}, {human_time(media.probe_duration(p))})")
    else:
        mp3 = media.to_mp3(master, out_dir / f"{base}.mp3", bitrate=args.bitrate)
        log(f"[MP3 ] {mp3} ({human_size(mp3.stat().st_size)}, {human_time(media.probe_duration(mp3))})")
        if cover_png is not None:
            mp4 = media.to_mp4(cover_png, master, out_dir / f"{base}.mp4", fps=args.fps, crf=args.crf)
            log(f"[MP4 ] {mp4} ({human_size(mp4.stat().st_size)}, {human_time(media.probe_duration(mp4))})")

    if not args.keep_flac:
        try:
            shutil.rmtree(master.parent, ignore_errors=True)
        except Exception:
            pass
    log(f"\n全部完成，输出目录: {out_dir}")
    return 0


def cmd_all(args) -> int:
    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    clips = scan_mod.scan_folder(folder)
    _print_scan(clips, folder, args.gap)

    if not args.no_filter:
        from . import asr

        cache = out_dir / f"asr_{args.model}.jsonl"
        loaded = _load_cache(clips, cache)
        todo = [c for c in clips if not c.asr_done]
        log(f"\n[转写] 模型={args.model} device={args.device}")
        log(f"  缓存命中 {loaded} / {len(clips)}，本次需转写 {len(todo)} 句")
        if todo:
            t0 = time.time()
            asr.transcribe_clips(
                todo,
                model_name=args.model,
                device=args.device,
                workers=args.workers,
                cache_path=cache,
            )
            log(f"  转写完成，用时 {human_time(time.time() - t0)}")

        if args.passes and args.passes > 0:
            _do_refine(args, clips, out_dir)

    return _do_build(args, folder, clips, out_dir)


# --------------------------------------------------------------------------- #
# 远程算力（租 GPU）
# --------------------------------------------------------------------------- #
def cmd_serve(args) -> int:
    from . import remote

    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    cache = out_dir / f"asr_{args.model}.jsonl"
    return int(
        remote.serve(
            folder,
            port=args.port,
            token=args.token,
            host=args.host,
            cache_path=cache,
            model_name=args.model,
        )
        or 0
    )


def cmd_worker(args) -> int:
    from . import remote

    return int(
        remote.run_worker(
            server=args.server,
            token=args.token,
            jobs_dir=args.jobs_dir,
            model_name=args.model,
            device=args.device,
            out_jsonl=args.out_jsonl,
            chunk_sec=args.chunk_sec,
            max_clips=args.max_clips,
            tmp_dir=args.tmp_dir,
        )
        or 0
    )


def cmd_pull(args) -> int:
    from . import remote

    return int(remote.pull_results(args.server, token=args.token, out_jsonl=args.out_jsonl) or 0)


def cmd_export_jobs(args) -> int:
    from . import remote

    folder = Path(args.folder)
    out_dir = Path(args.export_dir) if args.export_dir else folder / "remote_jobs"
    remote.export_jobs(folder, out_dir)
    return 0


def cmd_import_results(args) -> int:
    from . import remote

    folder = Path(args.folder)
    out_dir = _resolve_out(folder, args.out)
    n = remote.import_results(args.results, out_dir)
    log(f"[导入] {n} 条转写结果已合并到 {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", help="输出目录（默认 <文件夹>/voicepack_out）")
    p.add_argument("--gap", type=float, default=0.5, help="每句之间停顿秒数（默认 0.5）")


def _add_filter(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="small", help="Whisper 模型: tiny/base/small/medium/large-v3（默认 small）")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="ASR 设备")
    p.add_argument("--workers", type=int, default=1, help="CPU 转写并行进程数")
    p.add_argument("--no-filter", action="store_true", help="不做内容过滤，全部保留")
    p.add_argument("--strict", action="store_true", help="待确认的句子也一并剔除")
    p.add_argument("--drop-scenes", default="", help="整场剔除，如 12,15 或 noz012")
    p.add_argument("--keep-scenes", default="", help="整场强制保留，如 3,4")
    p.add_argument("--decisions", help="人工修改过的 filter_report.csv")
    p.add_argument("--cascade", choices=CASCADE_MODES, default="neighbour",
                   help="上下文扩散：neighbour=敏感台词前后连带的娇喘一并剔除（默认）；"
                        "scene=整个场景一起剔除（最保险，可能误删剧情）；off=关闭")


def _add_verify(p: argparse.ArgumentParser) -> None:
    p.add_argument("--passes", type=int, default=3,
                   help="可疑句多轮判别轮数（0=关闭，默认 3；轮数越多越准也越慢）")
    p.add_argument("--second-model", default=None,
                   help="多轮判别时额外用另一个模型交叉验证，如 medium / large-v3")


def _add_video(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", help="角色 CG 图路径（出 MP4 需要；不给则只出 MP3）")
    p.add_argument("--cover-mode", choices=["crop", "blur"], default="crop",
                   help="封面适配：crop=居中裁剪填满（默认）；blur=整图居中 + 模糊背景填充（竖图不裁人）")
    p.add_argument("--title", help="标题（角色名）")
    p.add_argument("--fps", type=int, default=5, help="视频帧率（静图，默认 5）")
    p.add_argument("--crf", type=int, default=20, help="x264 质量（默认 20）")
    p.add_argument("--bitrate", default="192k", help="MP3 码率（默认 192k）")
    p.add_argument("--mp3-only", action="store_true", help="只生成 MP3，不生成 MP4")
    p.add_argument("--no-suspect-mp3", action="store_true",
                   help="不生成「疑似清单」mp3（默认会生成 <名字>_suspect.mp3 + .csv）")
    p.add_argument("--split-minutes", type=float, default=0, help="按 N 分钟切片输出多个文件（0=不切）")
    p.add_argument("--keep-flac", action="store_true", help="保留无损母带 flac")


def _add_remote_serve(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    p.add_argument("--token", default="", help="访问令牌（建议设置，防公网裸奔）")
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0；只想本机就 127.0.0.1）")


def _add_remote_worker(p: argparse.ArgumentParser) -> None:
    p.add_argument("--server", default=None, help="本地服务地址，如 http://127.0.0.1:8765")
    p.add_argument("--token", default="", help="访问令牌")
    p.add_argument("--jobs-dir", default=None, help="离线任务包目录（不用 --server 时）")
    p.add_argument("--out-jsonl", default="remote_results.jsonl", help="结果 JSONL 输出路径")
    p.add_argument("--chunk-sec", type=float, default=240.0, help="分块转写的每块目标秒数")
    p.add_argument("--max-clips", type=int, default=32, help="每块句子数上限")
    p.add_argument("--tmp-dir", default=None, help="下载音频的临时目录")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicepack",
        description="柚子社语音合集工具：多轮 AI 过滤敏感内容 → 合成 MP3（+ 疑似清单）→ 生成 MP4",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="只扫描文件夹，统计句数与时长")
    p_scan.add_argument("folder")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_asr = sub.add_parser("asr", help="只做 AI 转写（可断点续跑）")
    p_asr.add_argument("folder")
    _add_common(p_asr)
    _add_filter(p_asr)
    p_asr.set_defaults(func=cmd_asr)

    p_filter = sub.add_parser("filter", help="只做第一轮过滤，生成清单和统计")
    p_filter.add_argument("folder")
    _add_common(p_filter)
    _add_filter(p_filter)
    p_filter.set_defaults(func=cmd_filter)

    p_refine = sub.add_parser("refine", help="对可疑句做多轮判别 + 多数表决，生成最终清单")
    p_refine.add_argument("folder")
    _add_common(p_refine)
    _add_filter(p_refine)
    _add_verify(p_refine)
    p_refine.set_defaults(func=cmd_refine)

    p_suspect = sub.add_parser("suspect", help="只根据已有清单重建「疑似句 mp3」，不重合成主音频")
    p_suspect.add_argument("folder")
    _add_common(p_suspect)
    _add_filter(p_suspect)
    _add_video(p_suspect)
    p_suspect.set_defaults(func=cmd_suspect)

    p_build = sub.add_parser("build", help="用已有转写结果合成 MP3（给 --image 才出 MP4）")
    p_build.add_argument("folder")
    _add_common(p_build)
    _add_filter(p_build)
    _add_video(p_build)
    p_build.set_defaults(func=cmd_build)

    p_all = sub.add_parser("all", help="一键完成：扫描→转写→多轮过滤→MP3→MP4")
    p_all.add_argument("folder")
    _add_common(p_all)
    _add_filter(p_all)
    _add_verify(p_all)
    _add_video(p_all)
    p_all.set_defaults(func=cmd_all)

    p_serve = sub.add_parser("serve", help="[远程算力] 在本机开服务，把语音文件派发给租来的 GPU")
    p_serve.add_argument("folder")
    _add_common(p_serve)
    _add_filter(p_serve)
    _add_remote_serve(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_worker = sub.add_parser("worker", help="[远程算力] 在租来的 GPU 上跑转写，把结果回传")
    _add_filter(p_worker)
    _add_remote_worker(p_worker)
    p_worker.set_defaults(func=cmd_worker)

    p_pull = sub.add_parser("pull", help="[远程算力] 把远端累积的转写结果拉回本地")
    p_pull.add_argument("--server", required=True, help="本地服务地址")
    p_pull.add_argument("--token", default="", help="访问令牌")
    p_pull.add_argument("--out-jsonl", default="remote_results.jsonl", help="保存路径")
    p_pull.set_defaults(func=cmd_pull)

    p_export = sub.add_parser("export-jobs", help="[远程算力] 生成离线任务包（拷到租来的机器上跑）")
    p_export.add_argument("folder")
    p_export.add_argument("--export-dir", default=None, help="任务包输出目录（默认 <文件夹>/remote_jobs）")
    p_export.set_defaults(func=cmd_export_jobs)

    p_import = sub.add_parser("import-results", help="[远程算力] 把远端转写结果合并进本地缓存")
    p_import.add_argument("folder")
    p_import.add_argument("results", help="远端导出的 remote_results.jsonl")
    _add_common(p_import)
    p_import.set_defaults(func=cmd_import_results)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log("\n已中断。转写结果已缓存，重新运行会接着跑。")
        return 130
    except Exception as exc:  # noqa: BLE001
        log(f"\n[错误] {exc}")
        if __import__("os").environ.get("VOICEPACK_DEBUG"):
            raise
        return 1
