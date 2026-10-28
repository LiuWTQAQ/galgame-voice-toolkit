"""Lead integration test: media pipeline only (no ASR/filter), 200 clips from E:\\mrxyy."""
import sys, time
from pathlib import Path

WS = Path(r"C:\Users\11424\Documents\deepseek-harness\default-workspace")
sys.path.insert(0, str(WS))

from voicepack import scan as scan_mod, media, cover
from voicepack.util import human_size, human_time

folder = Path(r"E:\mrxyy")
out = Path(r"E:\_vp_test")
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
clips = scan_mod.scan_folder(folder)[:200]
print(f"scanned {len(clips)} clips in {time.time()-t0:.1f}s")
print(scan_mod.summarize(clips))

master = out / "tmp" / "master.flac"
if not master.exists():
    t0 = time.time()
    media.build_master(clips, master)
    dt = time.time() - t0
else:
    dt = float("nan")
dur = media.probe_duration(master)
print(f"[master] {human_time(dur)} audio, {human_size(master.stat().st_size)}, encode {dt:.1f}s")

t0 = time.time()
mp3 = media.to_mp3(master, out / "test.mp3")
print(f"[mp3] {human_size(mp3.stat().st_size)} in {time.time()-t0:.1f}s, dur={media.probe_duration(mp3):.1f}s")

t0 = time.time()
png = cover.render_cover(None, out / "cover.png", title="Nozomi", subtitle="语音合集 · 200 句 · 15 分")
print(f"[cover] {png} in {time.time()-t0:.1f}s")

t0 = time.time()
mp4 = media.to_mp4(png, master, out / "test.mp4", fps=5, crf=20)
dt = time.time() - t0
d = media.probe_duration(mp4)
print(f"[mp4] {human_size(mp4.stat().st_size)} in {dt:.1f}s, dur={d:.1f}s -> {d/dt:.1f}x realtime")
print(f"extrapolate 7.0h video: {7.0*3600/(d/dt)/60:.0f} min encode")
