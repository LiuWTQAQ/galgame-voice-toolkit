"""远程 GPU 转写：本地起 HTTP 服务，租来的 GPU 只负责 faster-whisper 推理。

为什么需要这个模块
------------------
全量 4 个文件夹约 23 小时语音、1.7 万句，本机 RTX 4060 Laptop 只有 8GB 显存且常被
桌面程序/别的任务占用。租一张 4090/A100 按小时计费，把「重」的部分（Whisper 推理）
放到远端，把「必须在本机」的部分（扫描/过滤/合成、以及最终缓存文件）留在本地：

    本地                                              租来的 GPU
    ─────────────────────────────────                ─────────────────────
    scan_folder 得到 clip 列表
    RemoteServer（本模块）  ── GET /api/jobs ────────► worker 拉任务
                            ── GET /api/file/<name> ──► 下载 ogg 到临时目录
                                                         asr.transcribe_clips
                                                         （分块批量，复用 asr.py）
                            ◄─ POST /api/result ─────── 逐条回传结果
    追加写 remote_inbox.jsonl
    pull_results 下载全部结果
    import_results 合并进 asr_<model>.jsonl（与 AsrCache 完全兼容）
    之后照常 python -m voicepack filter / build

两种组网方式
------------
1. **在线**：本机跑 ``serve``，远端 worker 直接访问。本机没有公网 IP 时用 SSH 反向
   隧道（在**本机**执行 ``ssh -R 8765:127.0.0.1:8765 远端``），远端 worker 访问
   ``http://127.0.0.1:8765`` 即可；详见 ``docs/remote_gpu.md``。
2. **离线任务包**：本机 ``export_jobs`` 生成「ogg + manifest.json」，打包上传网盘，
   远端 ``run_worker(jobs_dir=...)`` 跑完下载 results.jsonl，本机 ``import_results``。
   适合完全没有公网入口、或者只想开一次机省事的情况。

安全
----
``serve`` 默认监听 ``0.0.0.0`` 且 token 默认空：**空 token = 不鉴权**，只应在
127.0.0.1/隧道里用。公网直连必须设置 ``--token``（``python -m voicepack.remote serve``
不传 token 时会随机生成并打印）。所有端点都支持 ``X-Token`` 头或 ``?token=``。

断点续跑
--------
服务端以 ``cache_path`` 指定的 JSONL 为「已完成」依据（``AsrCache`` 同名记录后写覆盖），
已完成的 name 不再派发；worker 被杀后重新启动会继续拉剩下的。派发中的任务带
``INFLIGHT_LEASE_SEC``（约 4 分钟）租约，worker 崩溃后过期会被重新派发。
``GET /api/jobs`` 默认只预览不加租约，worker 用 ``?claim=1`` 领取任务。
"""
from __future__ import annotations

import argparse
import hmac
import http.server
import json
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from . import asr
from .scan import parse_name, scan_folder
from .types import Clip
from .util import human_size, human_time, log, ogg_duration, setup_console

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
DEFAULT_PORT = 8765
DEFAULT_HOST = "0.0.0.0"
DEFAULT_CACHE_NAME = "remote_inbox.jsonl"  # 默认放 <folder>/voicepack_out/
DEFAULT_WORKER_MODEL = "large-v3"
API_JOBS_LIMIT = 32  # /api/jobs 单次默认派发数量
MAX_JOBS_LIMIT = 256  # 单次请求上限，防止一个请求把全部任务标记成在途
INFLIGHT_LEASE_SEC = 240.0  # 派发租约：worker 崩溃后约 4 分钟自动重新派发
MAX_RESULT_BYTES = 1 << 20  # POST /api/result 的 body 上限 1MB
HTTP_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 120.0
RETRY_ATTEMPTS = 6
RETRY_BASE_SEC = 1.0
RETRY_MAX_DELAY = 30.0
RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)  # 这些状态码重试，其余 4xx 直接报错
CHUNK_BYTES = 256 * 1024  # 文件分块写 socket 的块大小
EMPTY_WAIT_SEC = 10.0  # pending>0 但任务都在别的 worker 手里时的等待时间
MAX_EMPTY_ROUNDS = 30  # 最多等 5 分钟，防止死循环
TOKEN_HEADER = "X-Token"


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class RemoteError(RuntimeError):
    """远程调用失败（网络、HTTP 5xx、文件大小不符等）。"""


class RemoteAuthError(RemoteError):
    """鉴权失败（HTTP 401/403），不要重试，直接让用户检查 token。"""


# --------------------------------------------------------------------------- #
# 返回值（dict 子类：既方便脚本读字段，也能被 cli.py 的 int()/log 直接消费）
# --------------------------------------------------------------------------- #
class WorkerStats(dict):
    """``run_worker`` 的返回摘要。

    普通 dict 用法不变（``stats["posted"]``），另外 ``int(stats)`` 当退出码用：
    有失败句返回 1，否则 0（``voicepack/cli.py`` 里是 ``int(remote.run_worker(...))``）。
    """

    def __int__(self) -> int:
        return 1 if int(self.get("failed") or 0) else 0


class ImportSummary(dict):
    """``import_results`` 的返回摘要：``{模型名: 本次合并条数}``。

    ``str()`` 形如 ``small=60、large-v3=4129``（cli.py 直接把它拼进日志）。
    """

    def __str__(self) -> str:
        return "、".join(f"{model}={count}" for model, count in self.items()) or "0"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _normalize_server(server: str) -> str:
    """``127.0.0.1:8765`` / ``http://host:8765/`` → ``http://host:8765``。"""
    text = (server or "").strip().rstrip("/")
    if not text:
        raise ValueError("server 不能为空（也可以用环境变量 VOICEPACK_SERVER）")
    if "://" not in text:
        text = "http://" + text
    return text


def _local_ip() -> str:
    """取本机在局域网里的地址，仅用于打印提示（失败返回 ``127.0.0.1``）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def _atomic_write_text(path: Path, text: str) -> None:
    """先写 xxx.tmp 再 ``os.replace``，避免中途失败留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_jsonl(path: Path, records: Sequence[dict]) -> None:
    body = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    _atomic_write_text(path, body)


def _safe_model_tag(model: str) -> str:
    """把模型名变成安全文件名：``large-v3`` → ``large-v3``；本地路径 → 目录名。"""
    base = Path(str(model or "")).name or "unknown"
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base).strip("._")
    return cleaned or "unknown"


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """解析单段 ``Range: bytes=a-b``，返回闭区间 ``(start, end)``；不合法返回 None。

    只支持单段（worker 只需要断点续传/重试时取一段）。``bytes=-N`` = 最后 N 字节，
    ``bytes=a-`` = 从 a 到末尾。多段请求返回 None（调用方回 416，worker 不会发）。
    """
    text = (header or "").strip()
    if not text.startswith("bytes="):
        return None
    spec = text[len("bytes="):]
    if "," in spec:
        return None
    first, _, last = spec.partition("-")
    try:
        if not first:  # bytes=-N
            n = int(last)
            if n <= 0:
                return None
            start = max(0, size - n)
            end = size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1)


def _query_int(query: str, key: str, default: int, low: int = 1, high: int = MAX_JOBS_LIMIT) -> int:
    """从 query string 里取整型参数并夹在 [low, high]。"""
    try:
        values = urllib.parse.parse_qs(query).get(key, [])
        value = int(values[0]) if values else default
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _query_flag(query: str, key: str) -> bool:
    """从 query string 里取布尔参数（``1/true/yes/on`` 都算真）。"""
    values = urllib.parse.parse_qs(query).get(key, [])
    return bool(values) and values[0].strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# 服务端状态：结果缓存 + 在途租约
# --------------------------------------------------------------------------- #
class _ResultStore:
    """线程安全地往 JSONL 结果缓存追加记录（格式与 ``asr.AsrCache`` 完全一致）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache = asr.AsrCache(self.path).load()

    def done_names(self) -> set[str]:
        with self._lock:
            return set(self._cache.entries)

    def count(self) -> int:
        with self._lock:
            return len(self._cache.entries)

    def model(self) -> str:
        """最近一条结果的模型名（服务端没有显式配置模型，从结果里推断）。"""
        with self._lock:
            for rec in reversed(list(self._cache.entries.values())):
                if rec.model:
                    return rec.model
            return ""

    def add(self, rec: asr.CacheRecord) -> int:
        with self._lock:
            self._cache.put(rec)
            return len(self._cache.entries)

    def raw_text(self) -> str:
        with self._lock:
            if not self.path.is_file():
                return ""
            return self.path.read_text(encoding="utf-8", errors="replace")

    def close(self) -> None:
        with self._lock:
            self._cache.close()


# --------------------------------------------------------------------------- #
# 本地 HTTP 服务
# --------------------------------------------------------------------------- #
class RemoteServer:
    """把 ``folder`` 里还没转写的 ogg 分发给远端 worker，并收集结果。

    典型用法（脚本/自测里需要手动控制生命周期时）::

        srv = RemoteServer(folder, port=0, token="abc", cache_path=tmp/"inbox.jsonl")
        srv.start()          # 后台线程跑，srv.url 是实际地址
        ...
        srv.shutdown()

    命令行/阻塞用法直接调用模块级 :func:`serve`。

    Args:
        folder: 语音文件夹（用 ``scan_folder`` 扫描，名字规则与本地流程一致）。
        port: 监听端口，``0`` 表示随机空闲端口（自测用）。
        token: 鉴权 token，空字符串 = 不鉴权（只建议绑定 127.0.0.1 时使用）。
        host: 监听地址，默认 ``0.0.0.0``。
        cache_path: 结果 JSONL 路径，默认 ``<folder>/voicepack_out/remote_inbox.jsonl``。
            已存在的记录视为「已完成」，不再派发（断点续跑）。
        model_name: 期望的 Whisper 模型名（仅用于日志/``/api/status`` 展示；
            服务端本身不跑模型，实际模型由 worker 上报）。
    """

    def __init__(
        self,
        folder: str | Path,
        port: int = DEFAULT_PORT,
        token: str = "",
        host: str = DEFAULT_HOST,
        cache_path: str | Path | None = None,
        model_name: str | None = None,
    ) -> None:
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"文件夹不存在: {self.folder}")
        self.port = int(port)
        self.host = str(host) or DEFAULT_HOST
        self.token = str(token or "")
        self.model_name = str(model_name or "").strip()
        if cache_path is None:
            self.cache_path = self.folder / "voicepack_out" / DEFAULT_CACHE_NAME
        else:
            self.cache_path = Path(cache_path)

        clips = scan_folder(self.folder)
        if not clips:
            raise RuntimeError(f"没有可转写的 ogg 文件: {self.folder}")
        self._clips: list[Clip] = clips
        self._by_name: dict[str, Clip] = {clip.name: clip for clip in clips}
        self._order: list[str] = [clip.name for clip in clips]
        self._store = _ResultStore(self.cache_path)

        self._lock = threading.Lock()  # 保护 _inflight（派发租约）
        self._inflight: dict[str, float] = {}
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started_at = time.time()
        self._actual_port = self.port

    # ---- 状态 ---------------------------------------------------------- #
    @property
    def url(self) -> str:
        """本机访问地址（port=0 时是实际绑定到的端口）。"""
        return f"http://127.0.0.1:{self._actual_port}"

    @property
    def clip_count(self) -> int:
        return len(self._order)

    def status(self) -> dict:
        """``{total,done,pending,model}``，附带缓存路径便于排查。"""
        with self._lock:
            done_names = self._store.done_names()
        total = len(self._order)
        done = sum(1 for name in self._order if name in done_names)
        return {
            "total": total,
            "done": done,
            "pending": total - done,
            "model": self._store.model() or self.model_name,
            "cache": str(self.cache_path),
            "uptime_sec": round(time.time() - self._started_at, 1),
        }

    # ---- 任务派发 ------------------------------------------------------ #
    def jobs(self, limit: int = API_JOBS_LIMIT, claim: bool = False) -> list[dict]:
        """取最多 ``limit`` 个未完成任务。

        Args:
            limit: 本次最多返回多少条（1~256）。
            claim: ``True`` 时把返回的任务标记为「在途」（租约 4 分钟），用于
                worker 真正领取任务、避免多个 worker 重复转写；``False`` 只预览
                （给人工查看/脚本探测用，不占用租约）。

        返回项：``{name, size, mtime, dur, url}``。
        """
        limit = max(1, min(int(limit), MAX_JOBS_LIMIT))
        now = time.time()
        out: list[dict] = []
        with self._lock:
            # 清理过期租约：worker 崩溃后任务能重新派发
            for name in [n for n, stamp in self._inflight.items() if now - stamp > INFLIGHT_LEASE_SEC]:
                self._inflight.pop(name, None)
            done = self._store.done_names()
            for name in self._order:
                if len(out) >= limit:
                    break
                if name in done or name in self._inflight:
                    continue
                clip = self._by_name[name]
                try:
                    stat = clip.path.stat()
                    size, mtime = stat.st_size, stat.st_mtime
                except OSError:
                    size, mtime = -1, 0.0
                if claim:
                    self._inflight[name] = now
                out.append(
                    {
                        "name": name,
                        "size": size,
                        "mtime": round(mtime, 3),
                        "dur": round(clip.dur, 3),
                        "url": "/api/file/" + urllib.parse.quote(name),
                    }
                )
        return out

    # ---- 结果收集 ------------------------------------------------------ #
    def mark_result(self, payload: object) -> dict:
        """校验并写入一条 worker 回传结果；返回最新进度。

        Raises:
            KeyError: name 不在本服务的任务列表里（多半是拼写/换了文件夹）。
            ValueError: body 结构不对或数值字段无法解析。
        """
        if not isinstance(payload, dict):
            raise ValueError("body 必须是 JSON 对象")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("缺少 name 字段")
        clip = self._by_name.get(name)
        if clip is None:
            raise KeyError(name)
        try:
            text = str(payload.get("text") or "")
            avg_logprob = float(payload.get("avg_logprob") or 0.0)
            no_speech_prob = float(payload.get("no_speech_prob") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"字段类型不对: {exc}") from exc
        try:
            stat = clip.path.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:  # 文件在运行中被移走：退而用 worker 回传的值
            size = int(payload.get("size") or -1)
            mtime = float(payload.get("mtime") or 0.0)
        rec = asr.CacheRecord(
            name=name,
            model=str(payload.get("model") or self._store.model() or "unknown"),
            size=size,
            mtime=mtime,
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            vad_empty=bool(payload.get("vad_empty", False)),
            dur=float(payload.get("dur") or clip.dur or 0.0),
        )
        with self._lock:
            self._inflight.pop(name, None)
            count = self._store.add(rec)
        return {"ok": True, "name": name, "done": count, "total": len(self._order)}

    def cache_text(self) -> str:
        """整个结果 JSONL 的文本（``GET /api/cache`` 用）。"""
        return self._store.raw_text()

    def file_path(self, name: str) -> Path | None:
        """名字 → ogg 路径；只允许任务列表里的名字（天然防路径穿越）。"""
        clip = self._by_name.get(name)
        if clip is None:
            return None
        return clip.path if clip.path.is_file() else None

    # ---- 生命周期 ------------------------------------------------------ #
    def start(self) -> "RemoteServer":
        """在后台线程启动 HTTP 服务，立即返回。"""
        if self._httpd is not None:
            return self
        httpd = http.server.ThreadingHTTPServer((self.host, self.port), _build_handler(self))
        httpd.daemon_threads = True
        self._httpd = httpd
        self._actual_port = int(httpd.server_address[1])
        self._thread = threading.Thread(
            target=httpd.serve_forever, name="voicepack-remote-http", daemon=True
        )
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        """阻塞运行直到 Ctrl+C（命令行 ``serve`` 用）。"""
        self.start()
        self._log_banner()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except KeyboardInterrupt:
            log("\n[remote] 收到 Ctrl+C，正在关闭服务…")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """停止 HTTP 服务并关闭缓存文件句柄（可重复调用）。"""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:  # noqa: BLE001 - 关闭阶段不抛
                pass
            try:
                httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._store.close()

    # ---- 启动提示 ------------------------------------------------------ #
    def _log_banner(self) -> None:
        status = self.status()
        shown_token = self.token or "(无鉴权，仅限本机/隧道访问)"
        lan = _local_ip() if self.host in ("0.0.0.0", "::") else self.host
        log("=" * 72)
        log(f"[remote] 服务已启动: {self.url}  (监听 {self.host}:{self._actual_port})")
        log(f"[remote] 任务目录: {self.folder}")
        log(f"[remote] 缓存文件: {self.cache_path} (已有 {self._store.count()} 条)")
        log(f"[remote] 待转写  : {status['pending']} / {status['total']} 句")
        log(f"[remote] 模型    : {status['model'] or '(由 worker 回传时确定)'}")
        log(f"[remote] token   : {shown_token}")
        log(f"[remote] 端点    : /api/status  /api/jobs?limit=N[&claim=1]  /api/file/<name>  /api/result  /api/cache")
        if self.token:
            log(f"[remote] 远端命令: python -m voicepack worker --server http://{lan}:{self._actual_port} --token <token>")
            log(f"[remote] 无公网时: 在本机执行 ssh -R {self._actual_port}:127.0.0.1:{self._actual_port} <远端用户>@<远端主机>")
            log(f"[remote]           远端 worker 用 --server http://127.0.0.1:{self._actual_port}")
        log("=" * 72)


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def _build_handler(server_ref: RemoteServer) -> type[http.server.BaseHTTPRequestHandler]:
    """为某个 :class:`RemoteServer` 生成 handler 类（闭包带引用，避免全局变量）。"""

    class _Handler(http.server.BaseHTTPRequestHandler):
        server_version = "voicepack-remote/1.0"
        protocol_version = "HTTP/1.1"  # keep-alive；所有响应都必须带 Content-Length

        srv = server_ref

        # 请求日志太吵（全量 1.7 万次 POST），只保留显式 log() 的关键信息
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            pass

        # ---- 响应工具 -------------------------------------------------- #
        def _send_json(self, code: int, obj: object) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(code, body, "application/json; charset=utf-8")

        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _read_body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = 0
            if length <= 0:
                return b""
            if length > MAX_RESULT_BYTES:
                raise ValueError(f"body 太大（{length} 字节 > {MAX_RESULT_BYTES}）")
            return self.rfile.read(length)

        def _authorized(self) -> bool:
            token = self.srv.token
            if not token:
                return True
            supplied = self.headers.get(TOKEN_HEADER) or ""
            if not supplied:
                query = urllib.parse.urlsplit(self.path).query
                supplied = urllib.parse.parse_qs(query).get("token", [""])[0]
            try:
                return hmac.compare_digest(str(supplied).encode("utf-8"), token.encode("utf-8"))
            except TypeError:
                return False

        def _unauthorized(self) -> None:
            self._send_json(
                401,
                {
                    "error": "unauthorized",
                    "detail": f"token 不正确：请在请求头 {TOKEN_HEADER} 或 ?token= 里带上正确 token",
                },
            )

        # ---- GET ------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
            try:
                parts = urllib.parse.urlsplit(self.path)
                path = urllib.parse.unquote(parts.path)
                if not self._authorized():
                    self._unauthorized()
                    return
                if path.rstrip("/") == "/api/status":
                    self._send_json(200, self.srv.status())
                    return
                if path.rstrip("/") == "/api/jobs":
                    limit = _query_int(parts.query, "limit", API_JOBS_LIMIT)
                    claim = _query_flag(parts.query, "claim")
                    self._send_json(200, self.srv.jobs(limit, claim=claim))
                    return
                if path.rstrip("/") == "/api/cache":
                    body = self.srv.cache_text().encode("utf-8")
                    self._send_bytes(200, body, "application/x-ndjson; charset=utf-8")
                    return
                if path.startswith("/api/file/"):
                    self._serve_file(path[len("/api/file/"):])
                    return
                if path.rstrip("/") == "":
                    self._send_json(
                        200,
                        {
                            "name": "voicepack-remote",
                            "endpoints": [
                                "GET /api/status",
                                "GET /api/jobs?limit=N[&claim=1]",
                                "GET /api/file/<name>",
                                "POST /api/result",
                                "GET /api/cache",
                            ],
                        },
                    )
                    return
                self._send_json(404, {"error": "not found", "path": path})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001 - 不让单个坏请求打挂服务
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        do_HEAD = do_GET  # noqa: N815 - 文件下载/探测用

        def _serve_file(self, name: str) -> None:
            path = self.srv.file_path(name)
            if path is None:
                self._send_json(404, {"error": "file not found", "name": name})
                return
            try:
                size = path.stat().st_size
            except OSError as exc:
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            start, end = 0, size - 1
            partial = False
            range_header = self.headers.get("Range")
            if range_header:
                parsed = _parse_range(range_header, size)
                if parsed is None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start, end = parsed
                partial = True
            length = max(0, end - start + 1)
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "audio/ogg")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(path, "rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = fh.read(min(CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # 对端提前断开（worker 被杀），不是错误

        # ---- POST ------------------------------------------------------ #
        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._read_body()
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except (BrokenPipeError, ConnectionResetError):
                return
            if not self._authorized():
                self._unauthorized()
                return
            path = urllib.parse.urlsplit(self.path).path
            if path.rstrip("/") != "/api/result":
                self._send_json(404, {"error": "not found", "path": path})
                return
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"body 不是合法 JSON: {exc}"})
                return
            try:
                result = self.srv.mark_result(payload)
            except KeyError as exc:
                self._send_json(404, {"error": f"未知任务 name={exc}"})
                return
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send_json(200, result)

    return _Handler


def serve(
    folder: str | Path,
    port: int = DEFAULT_PORT,
    token: str = "",
    host: str = DEFAULT_HOST,
    cache_path: str | Path | None = None,
    model_name: str | None = None,
) -> None:
    """阻塞启动本地 HTTP 服务（命令行入口，Ctrl+C 退出）。

    Args:
        folder: 语音文件夹。
        port: 监听端口，默认 8765。
        token: 鉴权 token；空字符串 = 不鉴权（只建议本机/SSH 隧道使用）。
            为了不「裸奔」，空 token 时会自动生成一个随机 token 并打印（worker/pull 要用同一个）。
        host: 监听地址，默认 ``0.0.0.0``（局域网可访问）。
        cache_path: 结果 JSONL，默认 ``<folder>/voicepack_out/remote_inbox.jsonl``。
        model_name: 期望模型名（仅日志/状态展示；``python -m voicepack serve`` 会传
            ``--model`` 的值，方便确认 worker 用的模型对不对）。

    Examples:
        本机::

            python -m voicepack serve E:\\_vp_cli --port 8765 --token abc123
    """
    setup_console()
    if not str(token or "").strip():
        token = secrets.token_urlsafe(16)
        log(f"[remote] 未指定 --token，已自动生成: {token}（worker/pull 必须带同一个）")
    server = RemoteServer(
        folder, port=port, token=token, host=host, cache_path=cache_path, model_name=model_name
    )
    server.serve_forever()


# --------------------------------------------------------------------------- #
# HTTP 客户端（worker / pull 用；仅标准库）
# --------------------------------------------------------------------------- #
def _http_once(
    url: str,
    token: str = "",
    method: str = "GET",
    data: bytes | None = None,
    timeout: float = HTTP_TIMEOUT,
) -> tuple[int, bytes]:
    """发一次请求，返回 ``(status, body)``；4xx/5xx 也正常返回而不抛异常。"""
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", "voicepack-remote/1.0")
    if token:
        req.add_header(TOKEN_HEADER, token)
    if data is not None:
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:  # 4xx/5xx：读 body 供报错
        try:
            return int(exc.code), exc.read()
        except Exception:  # noqa: BLE001
            return int(exc.code), b""


def _http_retry(
    url: str,
    token: str = "",
    method: str = "GET",
    data: bytes | None = None,
    attempts: int = RETRY_ATTEMPTS,
    timeout: float = HTTP_TIMEOUT,
) -> bytes:
    """带指数退避的 HTTP 请求，成功返回 body。

    Raises:
        RemoteAuthError: 401/403（不重试，直接提示 token 问题）。
        RemoteError: 重试耗尽或不可重试的 HTTP 错误。
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            code, body = _http_once(url, token=token, method=method, data=data, timeout=timeout)
        except Exception as exc:  # URLError / timeout / ConnectionReset / OSError
            last = RemoteError(f"{type(exc).__name__}: {exc}")
        else:
            if 200 <= code < 300:
                return body
            text = body.decode("utf-8", "replace")[:300]
            if code in (401, 403):
                raise RemoteAuthError(
                    f"鉴权失败 HTTP {code}: {text}（检查 --token 与服务端是否一致）"
                )
            if code not in RETRY_STATUS:
                raise RemoteError(f"HTTP {code}: {text} ({url})")
            last = RemoteError(f"HTTP {code}: {text} ({url})")
        if attempt < attempts:
            delay = min(RETRY_MAX_DELAY, RETRY_BASE_SEC * (2 ** (attempt - 1)))
            log(f"[remote] 请求失败（第 {attempt}/{attempts} 次），{delay:.0f}s 后重试: {url} ({last})")
            time.sleep(delay)
    raise RemoteError(f"请求失败（{attempts} 次重试后放弃）: {url}: {last}")


def _http_json(
    url: str,
    token: str = "",
    method: str = "GET",
    payload: object | None = None,
    attempts: int = RETRY_ATTEMPTS,
    timeout: float = HTTP_TIMEOUT,
) -> Any:
    """请求并解析 JSON 响应。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    body = _http_retry(url, token=token, method=method, data=data, attempts=attempts, timeout=timeout)
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteError(f"响应不是合法 JSON: {url}: {exc}") from exc


def wait_until_ready(server: str, token: str = "", timeout: float = 20.0) -> dict:
    """轮询 ``/api/status`` 直到服务就绪，返回 status（自测/脚本用）。"""
    base = _normalize_server(server)
    deadline = time.time() + max(timeout, 1.0)
    last: Exception | None = None
    while time.time() < deadline:
        try:
            status = _http_json(base + "/api/status", token=token, attempts=1, timeout=5.0)
            if isinstance(status, dict):
                return status
        except Exception as exc:  # noqa: BLE001 - 轮询期间什么错都可能
            last = exc
        time.sleep(0.2)
    raise RemoteError(f"服务在 {timeout:.0f}s 内没有就绪: {base} ({last})")


# --------------------------------------------------------------------------- #
# worker：下载 → 转写 → 回传
# --------------------------------------------------------------------------- #
def _build_clip(name: str, path: Path, dur: float) -> Clip:
    """用文件名还原 Clip（prefix/scene/idx 供过滤阶段使用；解析失败给保守默认值）。"""
    safe_name = Path(name).name
    parsed = parse_name(safe_name)
    if parsed is None:  # 服务端任务列表都来自 scan_folder，正常不会走到这里
        prefix, scene, idx = "remote", 0, 0
    else:
        prefix, scene, idx = parsed[0], parsed[1], parsed[2]
    return Clip(
        path=path,
        name=safe_name,
        stem=Path(safe_name).stem,
        prefix=prefix,
        scene=scene,
        idx=idx,
        dur=float(dur),
    )


def _download_file(
    url: str,
    token: str,
    dest: Path,
    expected_size: int | None = None,
    attempts: int = RETRY_ATTEMPTS,
) -> int:
    """带重试地下载一个小文件，返回字节数。

    Raises:
        RemoteAuthError: 鉴权失败（不重试）。
        RemoteError: 重试耗尽、大小不符、空文件。
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            data = _http_retry(url, token=token, attempts=1, timeout=DOWNLOAD_TIMEOUT)
        except RemoteAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - 单次失败交给下面的退避
            last = exc
        else:
            if not data:
                last = RemoteError("下载得到空文件")
            elif expected_size is not None and len(data) != expected_size:
                last = RemoteError(f"文件大小不符（期望 {expected_size}，实际 {len(data)}）")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return len(data)
        if attempt < attempts:
            delay = min(RETRY_MAX_DELAY, RETRY_BASE_SEC * (2 ** (attempt - 1)))
            log(f"[remote] 下载失败（第 {attempt}/{attempts} 次），{delay:.0f}s 后重试: {url} ({last})")
            time.sleep(delay)
    raise RemoteError(f"下载失败（{attempts} 次重试后放弃）: {url}: {last}")


def _download_batch(
    jobs: Sequence[dict],
    batch_dir: Path,
    base: str,
    token: str,
    workers: int,
) -> tuple[list[Clip], dict[str, dict]]:
    """下载一批任务到 ``batch_dir``，返回 ``(clips, name -> {size,mtime})``。

    下载失败的句子只记日志并跳过（服务端租约过期后会重新派发）。
    """
    if not jobs:
        return [], {}
    info: dict[str, dict] = {}
    clips: list[Clip] = []

    def one(job: dict) -> Clip | None:
        raw_name = str(job.get("name") or "").strip()
        if not raw_name:
            return None
        name = Path(raw_name).name  # 去掉目录成分，防路径穿越
        if name in ("", ".", ".."):
            return None
        url = str(job.get("url") or ("/api/file/" + urllib.parse.quote(name)))
        if not url.startswith("http"):
            url = base + url
        expected = job.get("size")
        dest = batch_dir / name
        try:
            seen = _download_file(url, token, dest, expected_size=int(expected) if expected is not None else None)
        except RemoteAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - 单句失败不打断整批
            log(f"[remote] 跳过下载失败的句子 {name}: {exc}")
            return None
        dur = float(job.get("dur") or 0.0)
        if dur <= 0:
            dur = ogg_duration(dest)
        info[name] = {
            "size": int(expected) if expected is not None else seen,
            "mtime": float(job.get("mtime") or 0.0),
        }
        return _build_clip(name, dest, dur)

    workers = max(1, int(workers))
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for clip in pool.map(one, jobs):
                if clip is not None:
                    clips.append(clip)
    else:
        for job in jobs:
            clip = one(job)
            if clip is not None:
                clips.append(clip)
    return clips, info


def _transcribe_batch(
    clips: Sequence[Clip],
    model_name: str,
    device: str,
    compute_type: str,
    chunk_sec: float,
    max_clips: int,
) -> None:
    """调用 asr.py 的分块批量转写（结果原地写回 clip）。"""
    if not clips:
        return
    asr.transcribe_clips(
        clips,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        workers=1,  # GPU 不能多进程共享模型；并行下载由 workers 控制
        cache_path=None,  # 远端不落 asr 缓存，结果统一走 HTTP 回传
        chunked=True,
        chunk_sec=chunk_sec,
        max_clips_per_chunk=max_clips,
    )


def _record_from_clip(
    clip: Clip,
    model_name: str,
    info: dict | None = None,
) -> asr.CacheRecord:
    """把 Worker 端 Clip 转成一条与 AsrCache 完全兼容的记录（size/mtime 用原始文件的值）。"""
    meta = info or {}
    return asr.CacheRecord(
        name=clip.name,
        model=model_name,
        size=int(meta.get("size", -1)),
        mtime=float(meta.get("mtime", 0.0)),
        text=clip.text,
        avg_logprob=float(clip.avg_logprob),
        no_speech_prob=float(clip.no_speech_prob),
        vad_empty=bool(getattr(clip, "vad_empty", False)),
        dur=float(clip.dur),
    )


def _append_result(fh: Any, record: asr.CacheRecord) -> None:
    """结果追加写入本地 JSONL 并立即 flush（进程被杀最多丢最后一句）。"""
    fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
    fh.flush()


def _log_progress(done: int, total: int, t0: float) -> None:
    """```[remote] 已完成 128/4171 (3.1%) 12.4 句/秒``` 形式的进度。"""
    elapsed = max(time.time() - t0, 1e-6)
    rate = done / elapsed
    pct = done * 100.0 / total if total > 0 else 100.0
    left = (total - done) / rate if rate > 0 else 0.0
    log(f"[remote] 已完成 {done}/{total} ({pct:.1f}%) {rate:.1f} 句/秒  剩余约 {left / 60:.1f} 分钟")


def _load_out_names(path: Path) -> set[str]:
    """读取已写的 out_jsonl 里的 name 集合（离线模式断点续跑用）。"""
    names: set[str] = set()
    if not path.is_file():
        return names
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = str(obj.get("name") or "")
                if name:
                    names.add(name)
    except OSError:
        return names
    return names


# ---- 在线模式 --------------------------------------------------------------- #
def _run_server_mode(
    *,
    base: str,
    token: str,
    model_name: str,
    device: str,
    compute_type: str,
    out_fh: Any,
    out_path: Path,
    tmp_root: Path,
    workers: int,
    chunk_sec: float,
    max_clips: int,
    t0: float,
) -> dict:
    """在线模式主循环：``/api/jobs`` → 下载 → 分块转写 → ``POST /api/result``，直到没任务。"""
    status = _http_json(base + "/api/status", token=token)
    total = int(status.get("total") or 0)
    done_before = int(status.get("done") or 0)
    log(
        f"[remote] 服务端 {base}：共 {total} 句，已完成 {done_before}，"
        f"待转写 {total - done_before}，服务端缓存模型={status.get('model') or '(空)'}"
    )
    posted = 0
    failed_post = 0
    zero_progress_rounds = 0
    empty_rounds = 0
    batch_no = 0

    while True:
        jobs = _http_json(base + f"/api/jobs?limit={max(1, int(max_clips))}&claim=1", token=token)
        if not isinstance(jobs, list):
            raise RemoteError(f"/api/jobs 返回的不是列表: {type(jobs).__name__}")
        if not jobs:
            status = _http_json(base + "/api/status", token=token)
            pending = int(status.get("pending") or 0)
            if pending <= 0:
                break
            # pending>0 说明任务还在别的 worker 手里（或上一批在途租约没到期）
            empty_rounds += 1
            if empty_rounds > MAX_EMPTY_ROUNDS:
                raise RemoteError(
                    f"服务端 pending={pending} 但连续 {MAX_EMPTY_ROUNDS} 次没有任务可派发；"
                    f"可能有任务卡在在途状态，等待 {INFLIGHT_LEASE_SEC:.0f}s 租约过期后重试"
                )
            log(
                f"[remote] 暂时没有可派发任务（pending={pending}，可能在其他 worker 手里），"
                f"{EMPTY_WAIT_SEC:.0f}s 后重试"
            )
            time.sleep(EMPTY_WAIT_SEC)
            continue
        empty_rounds = 0
        batch_no += 1
        batch_dir = tmp_root / f"batch_{batch_no:05d}"
        try:
            clips, info = _download_batch(jobs, batch_dir, base, token, workers)
            if not clips:
                zero_progress_rounds += 1
                if zero_progress_rounds >= 3:
                    raise RemoteError("连续 3 批任务全部下载失败，检查服务端/隧道/网络后重试")
                continue
            try:
                _transcribe_batch(clips, model_name, device, compute_type, chunk_sec, max_clips)
            except Exception as exc:  # noqa: BLE001 - 单批失败不中断（服务端会重新派发）
                zero_progress_rounds += 1
                log(
                    f"[remote] 本批转写失败（{type(exc).__name__}: {exc}），跳过 {len(clips)} 句；"
                    f"租约过期后服务端会重新派发"
                )
                if zero_progress_rounds >= 3:
                    raise
                continue
            zero_progress_rounds = 0

            for clip in clips:
                record = _record_from_clip(clip, model_name, info.get(clip.name))
                payload: dict = record.as_dict()
                error = getattr(clip, "asr_error", None)
                if error:
                    payload["error"] = str(error)
                try:
                    _http_json(base + "/api/result", token=token, method="POST", payload=payload)
                except RemoteAuthError:
                    raise
                except Exception as exc:  # noqa: BLE001 - 回传失败只影响这一句
                    failed_post += 1
                    log(f"[remote] 回传失败 {clip.name}: {exc}；已写入本地 {out_path}，可用 import 兜底")
                else:
                    posted += 1
                _append_result(out_fh, record)
            _log_progress(done_before + posted, total, t0)
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    return {
        "mode": "server",
        "posted": posted,
        "failed": failed_post,
        "total": total,
        "server": base,
    }


# ---- 离线模式 --------------------------------------------------------------- #
def _load_manifest(jobs_dir: Path) -> list[dict]:
    """读取任务包 manifest.json：``[{name,size,mtime,dur}]`` 或 ``{"clips":[...]}``。"""
    path = jobs_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"任务包里没有 manifest.json: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest.json 不是合法 JSON: {exc}") from exc
    raw = obj.get("clips") if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        raise ValueError("manifest.json 结构不对：应为数组或 {\"clips\": [...]}")
    entries: list[dict] = []
    for item in raw:
        if isinstance(item, str):  # 兼容极简写法 ["a.ogg", "b.ogg"]
            entries.append({"name": item})
        elif isinstance(item, dict) and item.get("name"):
            entries.append(dict(item))
    return entries


def _find_job_file(jobs_dir: Path, name: str) -> Path | None:
    """在任务包里找 ogg：支持根目录 / ogg/ / files/ / audio/ 四种布局。"""
    base = Path(str(name)).name
    for candidate in (
        jobs_dir / base,
        jobs_dir / "ogg" / base,
        jobs_dir / "files" / base,
        jobs_dir / "audio" / base,
    ):
        if candidate.is_file():
            return candidate
    return None


def _run_offline_mode(
    *,
    jobs_dir: Path,
    model_name: str,
    device: str,
    compute_type: str,
    out_fh: Any,
    out_path: Path,
    workers: int,
    chunk_sec: float,
    max_clips: int,
    t0: float,
) -> dict:
    """离线任务包模式：读 manifest + 本地 ogg → 转写 → 写 out_jsonl（支持断点续跑）。"""
    entries = _load_manifest(jobs_dir)
    if not entries:
        log(f"[remote] 任务包为空: {jobs_dir}")
        return {"mode": "offline", "posted": 0, "failed": 0, "total": 0, "jobs_dir": str(jobs_dir)}

    missing: list[str] = []
    plan: list[tuple[dict, Path]] = []
    for entry in entries:
        path = _find_job_file(jobs_dir, str(entry["name"]))
        if path is None:
            missing.append(str(entry["name"]))
        else:
            plan.append((entry, path))
    if missing:
        log(f"[remote] 任务包里有 {len(missing)} 个文件找不到，已跳过（前 3 个: {missing[:3]}）")

    already = _load_out_names(out_path)
    if already:
        before = len(plan)
        plan = [(entry, path) for entry, path in plan if Path(str(entry["name"])).name not in already]
        log(f"[remote] 断点续跑：{before - len(plan)} 句已在 {out_path}，跳过")

    total = len(plan)
    if total == 0:
        log("[remote] 没有剩余任务")
        return {"mode": "offline", "posted": 0, "failed": len(missing), "total": 0,
                "jobs_dir": str(jobs_dir)}

    step = max(1, int(max_clips))
    posted = 0
    errors = 0
    for start in range(0, total, step):
        batch = plan[start:start + step]
        clips: list[Clip] = []
        infos: dict[str, dict] = {}
        for entry, path in batch:
            name = Path(str(entry["name"])).name
            dur = float(entry.get("dur") or 0.0) or ogg_duration(path)
            clips.append(_build_clip(name, path, dur))
            infos[name] = {"size": int(entry.get("size") or path.stat().st_size),
                           "mtime": float(entry.get("mtime") or 0.0)}
        try:
            _transcribe_batch(clips, model_name, device, compute_type, chunk_sec, max_clips)
        except Exception as exc:  # noqa: BLE001 - 不写 out_jsonl，下次重跑这批
            log(f"[remote] 本批转写失败（{type(exc).__name__}: {exc}），跳过 {len(clips)} 句不写缓存")
            errors += len(clips)
            continue
        for clip in clips:
            if getattr(clip, "asr_error", None):
                errors += 1
            _append_result(out_fh, _record_from_clip(clip, model_name, infos.get(clip.name)))
            posted += 1
        _log_progress(posted, total, t0)

    return {
        "mode": "offline",
        "posted": posted,
        "failed": len(missing) + errors,
        "total": total,
        "jobs_dir": str(jobs_dir),
    }


# --------------------------------------------------------------------------- #
# 主入口：run_worker / pull_results / export_jobs / import_results
# --------------------------------------------------------------------------- #
def run_worker(
    server: str | None = None,
    token: str = "",
    jobs_dir: str | Path | None = None,
    model_name: str = DEFAULT_WORKER_MODEL,
    device: str = "auto",
    compute_type: str = "auto",
    out_jsonl: str | Path = "remote_results.jsonl",
    workers: int = 1,
    chunk_sec: float = asr.DEFAULT_CHUNK_SEC,
    max_clips: int = asr.DEFAULT_CHUNK_CLIPS,
    tmp_dir: str | Path | None = None,
) -> dict:
    """远端 worker：拉任务 → 下载 → 分块转写 → 回传，直到没有任务。

    两种模式：

    - **在线**（``server`` 非空）：循环 ``GET /api/jobs`` → ``GET /api/file/<name>``
      → ``asr.transcribe_clips``（分块）→ 逐条 ``POST /api/result``。结果同时追加写入
      ``out_jsonl`` 作为本地备份（回传失败的句子还能 import 兜底）。
    - **离线**（``jobs_dir`` 非空）：读 ``<jobs_dir>/manifest.json`` 和同目录/``ogg/``
      下的文件，转写结果只写 ``out_jsonl``（已写过的 name 自动跳过，支持断点续跑）。

    Args:
        server: 服务端地址，如 ``http://127.0.0.1:8765``；``None`` 时读环境变量
            ``VOICEPACK_SERVER``。
        token: 鉴权 token；空时读环境变量 ``VOICEPACK_TOKEN``。
        jobs_dir: 离线任务包目录（与 ``server`` 二选一）。
        model_name: faster-whisper 模型名，默认 ``large-v3``（远端建议 large-v3）。
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``。
        compute_type: ``"auto"`` / ``"float16"`` / ``"int8"`` …
        out_jsonl: 本地结果备份路径（JSONL，格式与 AsrCache 一致）。
        workers: 并行**下载**线程数（转写仍是单模型串行；GPU 建议 1，CPU 机型可 4~8）。
        chunk_sec: 每块目标音频秒数（显存小就调小）。
        max_clips: 每批任务数 / 每块句数上限。
        tmp_dir: 下载临时目录；``None`` 时自动建临时目录并在结束后删除。

    Returns:
        摘要 dict：``{mode, posted, failed, total, elapsed_sec, rate, out_jsonl, ...}``。

    Examples:
        在线（AutoDL 上）::

            python -m voicepack.remote worker --server http://127.0.0.1:8765 \\
                --model large-v3 --device cuda --compute-type float16

        离线::

            python -m voicepack.remote worker --jobs-dir /root/taskpack --out /root/results.jsonl
    """
    setup_console()
    server = (server or os.environ.get("VOICEPACK_SERVER") or "").strip()
    token = token or os.environ.get("VOICEPACK_TOKEN") or ""
    if not server and not jobs_dir:
        raise ValueError("必须指定 server 或 jobs_dir 之一（也可设置环境变量 VOICEPACK_SERVER）")
    if server and jobs_dir:
        raise ValueError("server 与 jobs_dir 只能二选一")
    if int(workers) < 1:
        raise ValueError("workers 必须 >= 1")
    if int(max_clips) < 1:
        raise ValueError("max_clips 必须 >= 1")

    out_path = Path(out_jsonl)
    if str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    auto_tmp = tmp_dir is None
    tmp_root = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.mkdtemp(prefix="voicepack_remote_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    resolved_device, resolved_compute = asr._resolve_device(device, compute_type)
    log(
        f"[remote] worker 启动：模型={model_name} 设备={resolved_device} 量化={resolved_compute} "
        f"批大小={max_clips} chunk={chunk_sec:g}s 并行下载={workers}"
    )
    if server:
        base = _normalize_server(server)
        log(f"[remote] 服务端: {base}")
    else:
        base = ""
        log(f"[remote] 离线任务包: {Path(str(jobs_dir)).resolve()}")
    log(f"[remote] 结果输出: {out_path.resolve()}")
    t0 = time.time()

    try:
        # "a"：在线模式是本地备份，离线模式是断点续跑的依据
        with open(out_path, "a", encoding="utf-8") as out_fh:
            if server:
                summary = _run_server_mode(
                    base=base,
                    token=token,
                    model_name=model_name,
                    device=resolved_device,
                    compute_type=resolved_compute,
                    out_fh=out_fh,
                    out_path=out_path,
                    tmp_root=tmp_root,
                    workers=int(workers),
                    chunk_sec=float(chunk_sec),
                    max_clips=int(max_clips),
                    t0=t0,
                )
            else:
                summary = _run_offline_mode(
                    jobs_dir=Path(str(jobs_dir)),
                    model_name=model_name,
                    device=resolved_device,
                    compute_type=resolved_compute,
                    out_fh=out_fh,
                    out_path=out_path,
                    workers=int(workers),
                    chunk_sec=float(chunk_sec),
                    max_clips=int(max_clips),
                    t0=t0,
                )
    finally:
        if auto_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

    elapsed = time.time() - t0
    posted = int(summary.get("posted", 0))
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["rate"] = round(posted / elapsed, 2) if elapsed > 0 and posted else 0.0
    summary["out_jsonl"] = str(out_path)
    log(
        f"[remote] 全部完成：{summary['mode']} 模式回传 {posted} 句，失败 {summary.get('failed', 0)} 句，"
        f"用时 {human_time(elapsed)}，平均 {summary['rate']:.1f} 句/秒"
    )
    log(f"[remote] 本地结果备份: {out_path.resolve()}")
    if server:
        try:
            final = _http_json(base + "/api/status", token=token, attempts=2)
            log(
                f"[remote] 服务端进度: {final.get('done')}/{final.get('total')} "
                f"(pending={final.get('pending')})"
            )
            summary["server_status"] = final
            if int(final.get("pending") or 0) > 0:
                log("[remote] 注意：服务端仍有 pending 任务，稍后重跑 worker 即可接着拉（已完成的不会重复）")
        except RemoteError as exc:
            log(f"[remote] 拉取最终状态失败（不影响结果）: {exc}")
    return WorkerStats(summary)


def pull_results(
    server: str,
    token: str = "",
    out_jsonl: str | Path = "remote_results.jsonl",
) -> int:
    """下载服务端累积的全部结果到本地 JSONL（覆盖写，原子替换）。

    Args:
        server: 服务端地址；``None``/空时读环境变量 ``VOICEPACK_SERVER``。
        token: 鉴权 token；空时读 ``VOICEPACK_TOKEN``。
        out_jsonl: 本地输出路径。

    Returns:
        下载到的记录条数。
    """
    setup_console()
    base = _normalize_server(server or os.environ.get("VOICEPACK_SERVER") or "")
    token = token or os.environ.get("VOICEPACK_TOKEN") or ""
    body = _http_retry(base + "/api/cache", token=token)
    text = body.decode("utf-8", "replace")
    out = Path(out_jsonl)
    _atomic_write_text(out, text)
    count = sum(1 for line in text.splitlines() if line.strip())
    log(f"[remote] 已下载 {count} 条结果 → {out} ({human_size(len(body))})")
    return count


def export_jobs(folder: str | Path, out_dir: str | Path) -> dict:
    """生成离线任务包：``out_dir/ogg/<name>`` + ``out_dir/manifest.json``。

    已有同名同大小文件会跳过复制（支持中断后续传）。``manifest.json`` 记录
    ``name/size/mtime/dur``，worker 回传时带上 size/mtime，import 后的缓存能直接被
    ``AsrCache.get_valid`` 命中（不用因为 mtime 对不上重跑）。

    Returns:
        摘要 dict：``{out_dir, clips, bytes, copied, reused}``。
    """
    setup_console()
    clips = scan_folder(folder)
    if not clips:
        raise RuntimeError(f"没有可导出的 ogg 文件: {folder}")
    out = Path(out_dir)
    ogg_dir = out / "ogg"
    ogg_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    copied = reused = 0
    total_bytes = 0
    for clip in clips:
        try:
            stat = clip.path.stat()
        except OSError:
            log(f"[remote] 跳过无法读取的文件: {clip.path}")
            continue
        dest = ogg_dir / clip.name
        total_bytes += stat.st_size
        if dest.is_file() and dest.stat().st_size == stat.st_size:
            reused += 1
        else:
            shutil.copy2(clip.path, dest)
            copied += 1
        entries.append(
            {
                "name": clip.name,
                "size": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
                "dur": round(clip.dur, 3),
            }
        )
    manifest = {
        "version": 1,
        "folder": Path(folder).name,
        "count": len(entries),
        "total_bytes": total_bytes,
        "clips": entries,
    }
    _atomic_write_text(out / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    howto = (
        "离线任务包（voicepack remote）\n"
        "============================\n"
        "1. 把整个目录打包上传到远端机器（建议 zip 后传，几万个小文件直接传会很慢）\n"
        "2. 远端装环境: bash deploy/remote_setup.sh\n"
        "3. 远端跑 worker:\n"
        "   python -m voicepack.remote worker --jobs-dir . --out /root/results.jsonl \\\n"
        "       --model large-v3 --device cuda --compute-type float16\n"
        "4. 把 results.jsonl 传回本机，在项目根目录执行:\n"
        "   python -m voicepack.remote import results.jsonl --cache-dir <语音文件夹>/voicepack_out\n"
        "5. 之后照常: python -m voicepack filter <语音文件夹>\n"
    )
    _atomic_write_text(out / "HOWTO.txt", howto)
    log(
        f"[remote] 离线任务包已生成: {out.resolve()}  共 {len(entries)} 句，"
        f"{human_size(total_bytes)}（新复制 {copied}，复用 {reused}）"
    )
    return {
        "out_dir": str(out),
        "clips": len(entries),
        "bytes": total_bytes,
        "copied": copied,
        "reused": reused,
    }


def import_results(results_jsonl: str | Path, cache_dir: str | Path) -> dict[str, int]:
    """把 worker 结果按模型分组合并进 ``cache_dir/asr_<model>.jsonl``。

    与 ``asr.AsrCache`` 完全兼容：读入时用 ``CacheRecord.from_dict``（缺字段取默认值），
    按 name 去重（后写覆盖），与目录里已有缓存合并后**原子写**（先 ``.tmp`` 再 replace）。
    合并后的文件可以直接被 ``python -m voicepack asr/filter/build`` 使用。

    Args:
        results_jsonl: worker 回传/下载的结果 JSONL（``run_worker`` 的 out_jsonl 或
            ``pull_results`` 的产物）。
        cache_dir: 本地缓存目录，通常是 ``<语音文件夹>/voicepack_out``。

    Returns:
        :class:`ImportSummary`（dict 子类）：``{model: 该模型本次合并条数}``。

    Raises:
        FileNotFoundError: 结果文件不存在。
    """
    setup_console()
    src = Path(results_jsonl)
    if not src.is_file():
        raise FileNotFoundError(f"结果文件不存在: {src}")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, asr.CacheRecord]] = {}
    bad = 0
    total = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("not a dict")
                rec = asr.CacheRecord.from_dict(obj)
            except (json.JSONDecodeError, TypeError, ValueError):
                bad += 1
                continue
            if not rec.name:
                bad += 1
                continue
            total += 1
            model = rec.model or "unknown"
            grouped.setdefault(model, {})[rec.name] = rec

    if bad:
        log(f"[remote] 跳过 {bad} 行损坏/缺 name 的记录")
    if not grouped:
        log(f"[remote] 结果文件里没有可用记录: {src}")
        return ImportSummary()

    added: dict[str, int] = {}
    for model, records in grouped.items():
        path = cache_root / f"asr_{_safe_model_tag(model)}.jsonl"
        existing = asr.AsrCache(path).load()
        merged: dict[str, asr.CacheRecord] = dict(existing.entries)
        new_names = [name for name in records if name not in merged]
        merged.update(records)
        _atomic_write_jsonl(path, [rec.as_dict() for rec in merged.values()])
        added[model] = len(records)
        log(
            f"[remote] 合并 → {path}：本次 {len(records)} 条（新增 {len(new_names)}），"
            f"合并后共 {len(merged)} 条"
        )
    log(f"[remote] 共导入 {total} 条结果，覆盖 {len(grouped)} 个模型")
    return ImportSummary(added)


# --------------------------------------------------------------------------- #
# 命令行入口（python -m voicepack.remote ...）
# --------------------------------------------------------------------------- #
def _add_token(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--token",
        default="",
        help="鉴权 token（也可用环境变量 VOICEPACK_TOKEN）；serve 不填会随机生成并打印",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voicepack.remote",
        description="远程 GPU 转写：本地 serve / 远端 worker / 结果拉取与合并",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="本机启动 HTTP 服务，把待转写 ogg 分发给远端 worker")
    p_serve.add_argument("folder", help="语音文件夹")
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"端口（默认 {DEFAULT_PORT}）")
    p_serve.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
    p_serve.add_argument("--cache", default=None, help="结果 JSONL 路径（默认 <folder>/voicepack_out/remote_inbox.jsonl）")
    _add_token(p_serve)

    p_worker = sub.add_parser("worker", help="远端 GPU 上跑：拉任务 → 转写 → 回传")
    p_worker.add_argument("--server", default=None, help="服务端地址，如 http://127.0.0.1:8765")
    p_worker.add_argument("--jobs-dir", default=None, help="离线任务包目录（与 --server 二选一）")
    p_worker.add_argument("--model", default=DEFAULT_WORKER_MODEL, help=f"Whisper 模型（默认 {DEFAULT_WORKER_MODEL}）")
    p_worker.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="设备")
    p_worker.add_argument("--compute-type", default="auto", help="量化类型：auto/float16/int8_float16/int8")
    p_worker.add_argument("--out", default="remote_results.jsonl", help="本地结果备份 JSONL")
    p_worker.add_argument("--workers", type=int, default=1, help="并行下载线程数（GPU 建议 1）")
    p_worker.add_argument("--chunk-sec", type=float, default=asr.DEFAULT_CHUNK_SEC, help="每块目标秒数")
    p_worker.add_argument("--max-clips", type=int, default=asr.DEFAULT_CHUNK_CLIPS, help="每批任务数/每块句数")
    p_worker.add_argument("--tmp-dir", default=None, help="下载临时目录（默认自动建临时目录）")
    _add_token(p_worker)

    p_pull = sub.add_parser("pull", help="把服务端累积的结果下载回本机")
    p_pull.add_argument("--server", required=True, help="服务端地址")
    p_pull.add_argument("--out", default="remote_results.jsonl", help="输出 JSONL")
    _add_token(p_pull)

    p_export = sub.add_parser("export", help="生成离线任务包（ogg + manifest.json）")
    p_export.add_argument("folder", help="语音文件夹")
    p_export.add_argument("--out", required=True, help="任务包输出目录")

    p_import = sub.add_parser("import", help="把结果合并进 asr_<model>.jsonl（与 AsrCache 兼容）")
    p_import.add_argument("results", help="结果 JSONL 文件")
    p_import.add_argument(
        "--cache-dir",
        default=None,
        help="本地缓存目录（默认结果文件所在目录；通常传 <语音文件夹>/voicepack_out）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口，返回退出码。"""
    setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "serve":
            token = args.token or secrets.token_urlsafe(16)
            serve(args.folder, port=args.port, token=token, host=args.host, cache_path=args.cache)
            return 0
        if args.cmd == "worker":
            if not args.server and not args.jobs_dir:
                parser.error("worker 需要 --server 或 --jobs-dir")
            run_worker(
                server=args.server,
                token=args.token,
                jobs_dir=args.jobs_dir,
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                out_jsonl=args.out,
                workers=args.workers,
                chunk_sec=args.chunk_sec,
                max_clips=args.max_clips,
                tmp_dir=args.tmp_dir,
            )
            return 0
        if args.cmd == "pull":
            pull_results(args.server, token=args.token, out_jsonl=args.out)
            return 0
        if args.cmd == "export":
            export_jobs(args.folder, args.out)
            return 0
        if args.cmd == "import":
            cache_dir = args.cache_dir or str(Path(args.results).parent)
            import_results(args.results, cache_dir)
            return 0
        parser.error(f"未知子命令: {args.cmd}")
    except KeyboardInterrupt:
        log("\n已中断。已回传/已写盘的句子不会重复转写，重跑会接着跑。")
        return 130
    except RemoteAuthError as exc:
        log(f"\n[错误] {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001
        log(f"\n[错误] {type(exc).__name__}: {exc}")
        if os.environ.get("VOICEPACK_DEBUG"):
            raise
        return 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
