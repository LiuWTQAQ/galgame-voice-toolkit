"""Batch runner: whole pipeline (scan -> ASR -> filter -> MP3 -> MP4) for every folder.

Usage (from the workspace root):
    python tools/run_all_folders.py                # all four folders, default options
    python tools/run_all_folders.py --strict       # extra options are forwarded to every run

Each folder writes its log to E:\\voicepack_logs\\<folder>.log and its products to
<folder>\\voicepack_out.  ASR results are cached per folder, so an interrupted run
resumes where it stopped.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
LOG_DIR = Path(r"E:\voicepack_logs")
FOLDERS = [r"E:\mrxyy", r"E:\etyyyyy", r"E:\sjzmyy", r"E:\zyqhyy"]


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WS / ".deps") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    extra = sys.argv[1:]
    overall = time.time()
    failed: list[str] = []
    for folder in FOLDERS:
        name = Path(folder).name
        log_path = LOG_DIR / f"{name}.log"
        print(f"\n{'=' * 70}\n[{name}] start -> {log_path}\n{'=' * 70}", flush=True)
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start =====\n")
            fh.flush()
            cmd = [sys.executable, "-u", "-m", "voicepack", "all", folder, *extra]
            proc = subprocess.run(cmd, cwd=str(WS), env=env, stdout=fh, stderr=subprocess.STDOUT)
        ok = proc.returncode == 0
        print(f"[{name}] {'OK' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not ok:
            failed.append(name)
    print(f"\n全部结束，总用时 {(time.time() - overall) / 60:.1f} 分钟")
    if failed:
        print("失败: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
