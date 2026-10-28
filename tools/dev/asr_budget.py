"""Micro-benchmark: where does the ASR time go (decode vs inference), and fp16 vs int8_float16."""
import os, sys, time, glob
from pathlib import Path

WS = Path(r"C:\Users\11424\Documents\deepseek-harness\default-workspace")
DEPS = WS / ".deps"
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(WS))
for sub in ("cublas", "cudnn", "cuda_nvrtc"):
    p = DEPS / "nvidia" / sub / "bin"
    if p.is_dir():
        os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]

import subprocess
import numpy as np
from faster_whisper import WhisperModel

files = sorted(glob.glob(r"E:\mrxyy\*.ogg"))
files = [f for f in files if not f.endswith("_ref.ogg")][:40]


def decode(path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


t0 = time.time()
audios = [decode(f) for f in files]
dec = time.time() - t0
total = sum(len(a) for a in audios) / 16000
print(f"decode {len(files)} clips {total:.1f}s audio in {dec:.2f}s -> {total/dec:.0f}x realtime (serial)")

chunk = np.concatenate([np.concatenate([a, np.zeros(4800, dtype=np.float32)]) for a in audios])
print(f"chunk {len(chunk)/16000:.1f}s")

for ct in ("float16", "int8_float16"):
    model = WhisperModel("small", device="cuda", compute_type=ct)
    for run in range(2):
        t0 = time.time()
        segs, _ = model.transcribe(chunk, language="ja", beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False)
        n = len(list(segs))
        dt = time.time() - t0
        print(f"{ct} run{run}: {dt:.2f}s for {len(chunk)/16000:.1f}s audio -> RTF {dt/(len(chunk)/16000):.3f}, {n} segs")
    del model
