r"""Lead 用：对 4 个文件夹做「多轮判别 + 出 MP3 + 疑似清单」。

顺序：逐个文件夹 —— 等 ASR 缓存写满 → refine（GPU，串行）→ build（CPU，后台，最多 2 个并发）。
不生成 MP4（等用户提供 CG 图后再 build --image 补）。

用法:
    python -u tools/_lead_finalize.py E:\mrxyy E:\etyyyyy E:\sjzmyy E:\zyqhyy
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
LOG_DIR = Path(r"E:\voicepack_logs")
PY = sys.executable


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WS / ".deps") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _n_ogg(folder: Path) -> int:
    return sum(1 for p in folder.glob("*.ogg") if "_ref" not in p.stem)


def _n_cache(folder: Path) -> int:
    cache = folder / "voicepack_out" / "asr_small.jsonl"
    if not cache.exists():
        return 0
    with cache.open("r", encoding="utf-8", errors="ignore") as fh:
        return sum(1 for line in fh if line.strip())


def _run(name: str, stage: str, folder: str, extra: list[str], env: dict[str, str],
         log_name: str) -> subprocess.Popen:
    log_path = LOG_DIR / f"{log_name}.log"
    fh = open(log_path, "a", encoding="utf-8")
    fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage} start =====\n")
    fh.flush()
    cmd = [PY, "-u", "-m", "voicepack", stage, folder, *extra]
    print(f"[{name}] {stage} -> {log_path}  {' '.join(extra)}", flush=True)
    return subprocess.Popen(cmd, cwd=str(WS), env=env, stdout=fh, stderr=subprocess.STDOUT)


def main() -> int:
    folders = [Path(a) for a in (sys.argv[1:] or [r"E:\mrxyy"])]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = _env()
    builds: list[tuple[str, subprocess.Popen]] = []
    t_all = time.time()

    for folder in folders:
        name = folder.name
        need = _n_ogg(folder)
        print(f"[{name}] 需要 {need} 句", flush=True)
        # 若别的进程（run_pipeline.py）正在给这个文件夹转写，最多等 15 分钟，之后自己接手
        waited = 0.0
        while _n_cache(folder) < need and waited < 900:
            time.sleep(60)
            waited += 60
            if waited % 300 == 0:
                print(f"[{name}] 等外部 ASR: {_n_cache(folder)}/{need} 句", flush=True)
        if _n_cache(folder) < need:
            # 缓存没满说明 ASR 还没跑过这个文件夹：自己补跑（GPU 串行）
            t0 = time.time()
            proc = _run(name, "asr", str(folder), [], env, f"{name}_finalize")
            rc = proc.wait()
            print(f"[{name}] asr {'OK' if rc == 0 else f'FAILED({rc})'} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if rc != 0 or _n_cache(folder) < need:
                print(f"[{name}] ASR 仍不完整（{_n_cache(folder)}/{need}），跳过", flush=True)
                continue

        t0 = time.time()
        # refine 阶段关掉上下文扩散：先让每句被独立判定，扩散留到 build 时叠加
        proc = _run(name, "refine", str(folder), ["--passes", "3", "--cascade", "off"],
                    env, f"{name}_finalize")
        rc = proc.wait()
        print(f"[{name}] refine {'OK' if rc == 0 else f'FAILED({rc})'} "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            continue

        # build 用验证后的清单（build 会自动优先读 filter_report_verified.csv）
        bproc = _run(name, "build", str(folder), [], env, f"{name}_finalize")
        builds.append((name, bproc))
        while sum(1 for _, p in builds if p.poll() is None) > 2:
            time.sleep(20)

    for name, proc in builds:
        rc = proc.wait()
        print(f"[{name}] build {'OK' if rc == 0 else f'FAILED({rc})'}", flush=True)
    print(f"全部结束，总用时 {(time.time() - t_all) / 60:.1f} 分钟", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
