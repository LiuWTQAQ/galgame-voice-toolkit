"""Lead scale test: build a 6h master FLAC from a full folder, measure throughput."""
import sys, time
from pathlib import Path

WS = Path(r"C:\Users\11424\Documents\deepseek-harness\default-workspace")
sys.path.insert(0, str(WS))

from voicepack import scan as scan_mod, media
from voicepack.util import human_size, human_time

folder = Path(r"E:\etyyyyy")
tt = time.time()
clips = scan_mod.scan_folder(folder)
print(f"scanned {len(clips)} clips in {time.time()-tt:.1f}s", flush=True)
print(scan_mod.summarize(clips), flush=True)

out = Path(r"E:\_vp_scale") / "master.flac"
t0 = time.time()
media.build_master(clips, out)
dt = time.time() - t0
dur = media.probe_duration(out)
print(
    f"[master] {human_time(dur)} audio from {len(clips)} clips in {dt:.1f}s "
    f"=> {dur/dt:.1f}x realtime, {human_size(out.stat().st_size)}",
    flush=True,
)
