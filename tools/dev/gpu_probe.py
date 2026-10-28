import os, sys, time, glob
from pathlib import Path

WS = Path(r"C:\Users\11424\Documents\deepseek-harness\default-workspace")
DEPS = WS / ".deps"
os.environ.setdefault("PYTHONPATH", str(DEPS))
sys.path.insert(0, str(DEPS))
for sub in ("cublas", "cudnn", "cuda_nvrtc"):
    p = DEPS / "nvidia" / sub / "bin"
    if p.is_dir():
        os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]

import ctranslate2
print("ctranslate2", ctranslate2.__version__, "cuda devices:", ctranslate2.get_cuda_device_count())

from faster_whisper import WhisperModel
t0 = time.time()
try:
    model = WhisperModel("small", device="cuda", compute_type="float16")
    print("loaded cuda fp16 in %.1fs" % (time.time() - t0))
    dev = "cuda"
except Exception as e:
    print("cuda load failed:", type(e).__name__, e)
    t0 = time.time()
    model = WhisperModel("small", device="cpu", compute_type="int8")
    print("loaded cpu int8 in %.1fs" % (time.time() - t0))
    dev = "cpu"

import subprocess
import numpy as np

def decode(path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(p.stdout, dtype=np.float32)

files = sorted(glob.glob(r"E:\mrxyy\*.ogg"))
files = [f for f in files if not f.endswith("_ref.ogg")][:12]
total = 0.0
t0 = time.time()
for f in files:
    audio = decode(f)
    total += len(audio) / 48000.0
    segs, _ = model.transcribe(audio, language="ja", beam_size=1, vad_filter=False)
    txt = "".join(s.text for s in segs).strip()
    print(f"{Path(f).name}  {len(audio)/48000:.2f}s  {txt}")
el = time.time() - t0
print(f"\n{len(files)} clips / {total:.1f}s audio in {el:.1f}s -> RTF {el/max(total,1e-9):.3f}")
