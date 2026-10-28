r"""Lead 用：批量流水线（ASR 用 GPU 串行，build 用 CPU，两者重叠跑）。

用法（脱离会话后台跑）:
    nohup python -u tools/run_pipeline.py E:\etyyyyy E:\sjzmyy E:\zyqhyy > E:\voicepack_logs\pipeline.log 2>&1 &

每个文件夹：
  1) python -m voicepack asr <folder>        —— 转写（GPU，断点续跑）
  2) python -m voicepack build <folder>      —— 过滤 + 母带 + MP3 + MP4（CPU，后台并行）
ASR 之间串行（GPU 只有一块），build 与下一个 ASR 重叠（互不抢资源）。
每个文件夹的日志写到 E:\\voicepack_logs\\<name>.log。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
LOG_DIR = Path(r"E:\voicepack_logs")
EXTRA = ["--cascade", "neighbour"]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WS / ".deps") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(folder: str, stage: str, extra: list[str], env: dict[str, str],
         log_name: str) -> tuple[subprocess.Popen, object]:
    cmd = [sys.executable, "-u", "-m", "voicepack", stage, folder, *extra]
    log_path = LOG_DIR / f"{log_name}.log"
    fh = open(log_path, "a", encoding="utf-8")
    fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage} start =====\n")
    fh.flush()
    print(f"[{log_name}] {stage} -> {log_path}", flush=True)
    return subprocess.Popen(cmd, cwd=str(WS), env=env, stdout=fh, stderr=subprocess.STDOUT), fh


def main() -> int:
    folders = sys.argv[1:] or [r"E:\etyyyyy", r"E:\sjzmyy", r"E:\zyqhyy"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = _env()
    builds: list[tuple[str, subprocess.Popen, object]] = []
    t_all = time.time()
    for folder in folders:
        name = Path(folder).name
        t0 = time.time()
        proc, fh = _run(folder, "asr", [], env, name)
        rc = proc.wait()
        fh.close()
        print(f"[{name}] asr {'OK' if rc == 0 else f'FAILED({rc})'} "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            continue
        bproc, bfh = _run(folder, "build", EXTRA, env, name)
        builds.append((name, bproc, bfh))
        # 同时最多 2 个 build
        while sum(1 for _, p, _ in builds if p.poll() is None) > 2:
            time.sleep(20)
    for name, proc, fh in builds:
        rc = proc.wait()
        fh.close()
        print(f"[{name}] build {'OK' if rc == 0 else f'FAILED({rc})'}", flush=True)
    print(f"全部结束，总用时 {(time.time() - t_all) / 60:.1f} 分钟", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
