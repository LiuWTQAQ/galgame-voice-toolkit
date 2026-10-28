#!/usr/bin/env python3
"""远程 GPU 模块自测：本机起 serve + 本机 worker（CPU int8）跑通完整闭环。

跑法（在项目根目录）::

    set PYTHONPATH=.deps
    python tools/remote_selftest.py

流程（全部在 127.0.0.1 上，绝不占用 GPU）：

1. ``RemoteServer`` 起在随机端口，token 随机生成；
2. 鉴权：无 token / 错 token 必须 401，``?token=`` 必须放行；
3. ``/api/status`` 应为 60 句全待转写；``/api/jobs`` 带 size/url；``/api/file`` 字节数
   与本地文件一致，``Range`` 请求返回 206 且切片正确；
4. ``run_worker(server=...)``：模型 small / device cpu / int8，把 E:\\_vp_cli 的 60 句
   全部转写并回传（进度日志形如 ``[remote] 已完成 32/60 (53.3%) ...``）；
5. ``pull_results`` 下载结果，``import_results`` 合并成 ``asr_small.jsonl``，
   用 ``scan_folder`` + ``AsrCache.get_valid`` 验证 100% 命中（size/mtime/模型全部对得上）；
6. 与本地已有转写缓存 ``E:\\_vp_cli\\voicepack_out\\asr_small.jsonl`` 对照文本相似度
   （那份缓存是更早版本的 asr.py 跑出来的，分块归属/量化精度都略有差异，不要求逐字一致，
   只检查句子集合一致、平均相似度 >= 0.85）；
7. 离线任务包：``export_jobs`` 3 句 → ``run_worker(jobs_dir=...)`` → 重复跑验证断点续跑
   → ``import_results`` 验证离线结果同样能合并。

退出码：0 = 全部通过；1 = 有失败项或异常。中间产物保留在 ``.remote_selftest/``。
"""
from __future__ import annotations

import difflib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voicepack import asr, remote  # noqa: E402
from voicepack.scan import scan_folder  # noqa: E402
from voicepack.util import human_size, human_time, setup_console  # noqa: E402

SRC = Path(r"E:\_vp_cli")
LOCAL_CACHE = SRC / "voicepack_out" / "asr_small.jsonl"
MODEL = "small"
DEVICE = "cpu"
COMPUTE = "int8"
WORK = ROOT / ".remote_selftest"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """记录一项检查，返回 ok（方便链式判断）。"""
    CHECKS.append((name, bool(ok), detail))
    print(f"[selftest] {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL，坏行跳过。"""
    out: list[dict] = []
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def main() -> int:
    setup_console()
    print("=" * 78)
    print("[selftest] 远程模块自测：serve + worker(cpu/int8/small) + pull + import")
    print(f"[selftest] 数据源: {SRC}")
    print(f"[selftest] 工作目录: {WORK}")
    print("=" * 78)

    if not SRC.is_dir():
        print(f"[selftest] FAIL 测试数据目录不存在: {SRC}")
        return 1
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    clips = scan_folder(str(SRC))
    check("扫描测试数据得到 60 句", len(clips) == 60, f"实际 {len(clips)} 句")

    token = remote.secrets.token_urlsafe(12)
    inbox = WORK / "inbox.jsonl"
    srv = remote.RemoteServer(SRC, port=0, token=token, host="127.0.0.1", cache_path=inbox)
    srv.start()
    url = srv.url
    print(f"[selftest] 服务端已启动: {url}  token={token[:6]}…")
    try:
        # ---- 1. 鉴权 -------------------------------------------------- #
        code, _ = remote._http_once(url + "/api/status")
        check("无 token → 401", code == 401, f"HTTP {code}")
        code, _ = remote._http_once(url + "/api/status", token="wrong-token")
        check("错误 token → 401", code == 401, f"HTTP {code}")
        status = remote.wait_until_ready(url, token=token, timeout=10.0)
        check(
            "status: 60 句全部待转写",
            status.get("total") == 60 and status.get("done") == 0 and status.get("pending") == 60,
            json.dumps({k: status.get(k) for k in ("total", "done", "pending")}, ensure_ascii=False),
        )
        body = remote._http_retry(url + "/api/status?token=" + token, token="", attempts=1)
        check("?token= 查询参数放行", b'"total"' in body, f"{len(body)} 字节")

        # ---- 2. 任务列表 / 文件 / Range ------------------------------- #
        jobs = remote._http_json(url + "/api/jobs?limit=5", token=token)
        check(
            "jobs 返回 5 条且带 name/size/url",
            isinstance(jobs, list) and len(jobs) == 5 and all({"name", "size", "url"} <= set(j) for j in jobs),
            ", ".join(j["name"] for j in jobs),
        )
        first = jobs[0]
        data = remote._http_retry(url + first["url"], token=token, attempts=1)
        local_first = (SRC / first["name"]).read_bytes()
        check(
            "下载文件与本地一致",
            data == local_first and len(data) == first["size"],
            f"{human_size(len(data))} / {first['name']}",
        )
        import urllib.request

        req = urllib.request.Request(url + first["url"], headers={remote.TOKEN_HEADER: token, "Range": "bytes=100-299"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            part = resp.read()
            check(
                "Range 请求 → 206",
                resp.status == 206 and part == local_first[100:300],
                f"HTTP {resp.status}, Content-Range={resp.headers.get('Content-Range')}",
            )

        # ---- 3. worker（在线模式，CPU int8，small） ------------------- #
        print("-" * 78)
        print("[selftest] 开始 worker（server 模式 / device=cpu / compute_type=int8 / model=small）")
        worker_out = WORK / "remote_results.jsonl"
        t0 = time.time()
        summary = remote.run_worker(
            server=url,
            token=token,
            model_name=MODEL,
            device=DEVICE,
            compute_type=COMPUTE,
            out_jsonl=str(worker_out),
            workers=1,
            chunk_sec=240.0,
            max_clips=32,
            tmp_dir=str(WORK / "tmp"),
        )
        worker_elapsed = time.time() - t0
        print("-" * 78)
        check(
            "worker 回传 60 句、无失败",
            summary.get("posted") == 60 and summary.get("failed", 0) == 0,
            f"posted={summary.get('posted')} failed={summary.get('failed')} "
            f"用时={human_time(worker_elapsed)} rate={summary.get('rate')} 句/秒",
        )
        status = remote._http_json(url + "/api/status", token=token)
        check(
            "服务端 status: done=60 / pending=0",
            status.get("done") == 60 and status.get("pending") == 0,
            json.dumps({k: status.get(k) for k in ("total", "done", "pending", "model")}, ensure_ascii=False),
        )
        check(
            "服务端缓存模型记录为 small",
            status.get("model") == MODEL,
            f"model={status.get('model')}",
        )
        check("全部完成后 /api/jobs 为空", remote._http_json(url + "/api/jobs?limit=8", token=token) == [])

        # ---- 4. pull + import ---------------------------------------- #
        pulled = WORK / "pulled.jsonl"
        pulled_count = remote.pull_results(url, token=token, out_jsonl=str(pulled))
        pulled_recs = read_jsonl(pulled)
        check(
            "pull_results 下载 60 条",
            pulled_count == 60 and len(pulled_recs) == 60,
            f"{pulled_count} 条 / {human_size(pulled.stat().st_size)}",
        )
        check(
            "pull 结果与 worker 本地备份同名集合一致",
            {r["name"] for r in pulled_recs} == {r["name"] for r in read_jsonl(worker_out)},
        )

        cache_dir = WORK / "cache"
        added = remote.import_results(pulled, cache_dir)
        cache_path = cache_dir / f"asr_{MODEL}.jsonl"
        cache = asr.AsrCache(cache_path).load()
        check(
            "import 生成 asr_small.jsonl 且 60 条",
            cache_path.is_file() and len(cache.entries) == 60 and added.get(MODEL) == 60,
            f"{cache_path}",
        )
        valid = sum(1 for clip in clips if cache.get_valid(clip, MODEL) is not None)
        check(
            "合并后的缓存可被 AsrCache.get_valid 100% 命中（size/mtime/model 全部匹配）",
            valid == len(clips),
            f"{valid}/{len(clips)}",
        )
        left_tmp = list(cache_dir.glob("*.tmp"))
        check("import 原子写无残留 .tmp", not left_tmp, f"{len(left_tmp)} 个")

        # ---- 5. 与本地转写结果对照 ----------------------------------- #
        local_recs = {r["name"]: r for r in read_jsonl(LOCAL_CACHE)}
        remote_recs = {r["name"]: r for r in pulled_recs}
        if local_recs:
            names = sorted(set(local_recs) & set(remote_recs))
            exact = sum(1 for n in names if local_recs[n].get("text") == remote_recs[n].get("text"))
            sims = [
                difflib.SequenceMatcher(None, remote_recs[n].get("text", ""), local_recs[n].get("text", "")).ratio()
                for n in names
            ]
            mean_sim = sum(sims) / len(sims) if sims else 0.0
            print("[selftest] 前 5 句与本地缓存对照（文本截断到 26 字）：")
            for name in names[:5]:
                ratio = difflib.SequenceMatcher(
                    None, remote_recs[name].get("text", ""), local_recs[name].get("text", "")
                ).ratio()
                print(f"    {name:18s} 相似度={ratio:.3f}")
                print(f"      远端: {remote_recs[name].get('text', '')[:26]}")
                print(f"      本地: {local_recs[name].get('text', '')[:26]}")
            check("结果与本地缓存同名集合一致", set(local_recs) == set(remote_recs))
            check(
                "文本平均相似度 >= 0.85（完全一致句数见 detail）",
                mean_sim >= 0.85,
                f"平均 {mean_sim:.3f}，完全一致 {exact}/{len(names)}",
            )
        else:
            print(f"[selftest] 跳过文本对照：本地缓存不存在 {LOCAL_CACHE}")

        # ---- 6. 离线任务包模式 --------------------------------------- #
        print("-" * 78)
        print("[selftest] 离线任务包模式：export_jobs → worker --jobs-dir → import")
        off_src = WORK / "offline_src"
        off_src.mkdir(parents=True, exist_ok=True)
        for clip in clips[:3]:
            shutil.copy2(clip.path, off_src / clip.name)
        pack = WORK / "taskpack"
        exported = remote.export_jobs(off_src, pack)
        manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
        check(
            "export_jobs 生成 3 条 manifest + ogg 文件",
            exported.get("clips") == 3 and manifest.get("count") == 3
            and len(list((pack / "ogg").glob("*.ogg"))) == 3,
            f"{human_size(exported.get('bytes', 0))}",
        )
        exported2 = remote.export_jobs(off_src, pack)  # 幂等：第二次应全部复用
        check(
            "export_jobs 幂等（第二次全部复用已有文件）",
            exported2.get("reused") == 3 and exported2.get("copied") == 0,
            f"copied={exported2.get('copied')} reused={exported2.get('reused')}",
        )
        off_out = WORK / "offline_results.jsonl"
        off_summary = remote.run_worker(
            jobs_dir=str(pack),
            out_jsonl=str(off_out),
            model_name=MODEL,
            device=DEVICE,
            compute_type=COMPUTE,
            chunk_sec=240.0,
            max_clips=32,
        )
        check(
            "离线 worker 转写 3 句",
            off_summary.get("posted") == 3 and len(read_jsonl(off_out)) == 3,
            f"posted={off_summary.get('posted')}",
        )
        resume = remote.run_worker(
            jobs_dir=str(pack),
            out_jsonl=str(off_out),
            model_name=MODEL,
            device=DEVICE,
            compute_type=COMPUTE,
            chunk_sec=240.0,
            max_clips=32,
        )
        check(
            "离线 worker 断点续跑（第二次 0 句、不重复写）",
            resume.get("posted") == 0 and len(read_jsonl(off_out)) == 3,
            f"posted={resume.get('posted')}",
        )
        off_cache_dir = WORK / "offline_cache"
        remote.import_results(off_out, off_cache_dir)
        off_cache = asr.AsrCache(off_cache_dir / f"asr_{MODEL}.jsonl").load()
        check(
            "离线结果 import 成 3 条兼容缓存",
            len(off_cache.entries) == 3,
            str(off_cache_dir / f"asr_{MODEL}.jsonl"),
        )
    finally:
        srv.shutdown()

    # ---- 汇总 --------------------------------------------------------- #
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    failed = len(CHECKS) - passed
    print("=" * 78)
    print(f"[selftest] 检查项 {passed} 通过 / {failed} 失败（共 {len(CHECKS)}）")
    print(f"[selftest] 中间产物: {WORK}")
    if failed:
        print("[selftest] 失败项：")
        for name, ok, detail in CHECKS:
            if not ok:
                print(f"  - {name}  ({detail})")
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[selftest] 已中断")
        raise SystemExit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SELFTEST FAILED（异常）")
        raise SystemExit(1)
