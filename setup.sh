#!/usr/bin/env bash
# Linux / macOS / 算力平台（AutoDL、RunPod、vast.ai…）环境安装脚本
#
#   bash setup.sh              # 装到 ./.venv，可选装 CUDA 运行库
#   bash setup.sh --check-only # 只检查环境（全就绪返回 0）
#   bash setup.sh --force      # 删掉 .venv 重装
#   bash setup.sh --no-cuda    # 不装 CUDA 相关 wheel（CPU 也能跑）
set -euo pipefail

cd "$(dirname "$0")"

CHECK_ONLY=0
FORCE=0
WANT_CUDA="ask"
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    --force)      FORCE=1 ;;
    --no-cuda)    WANT_CUDA=0 ;;
    --cuda)       WANT_CUDA=1 ;;
    -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

say() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 1. 找 Python 3.10+ ----------
PY=""
for cand in "${VOICEPACK_PYTHON:-}" python3 python python3.12 python3.11 python3.10; do
  [ -n "$cand" ] || continue
  if command -v "$cand" >/dev/null 2>&1; then
    ver=$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)
    if [ -n "$ver" ] && [ "${ver%%.*}" -ge 3 ] && [ "${ver##*.}" -ge 10 ]; then PY="$cand"; break; fi
  fi
done
[ -n "$PY" ] || die "没找到 Python 3.10+，请先安装（apt install python3.11 python3.11-venv）"
say "Python: $("$PY" -V 2>&1)"

# ---------- 2. ffmpeg / ffprobe ----------
missing=""
for bin in ffmpeg ffprobe; do
  command -v "$bin" >/dev/null 2>&1 || missing="$missing $bin"
done
if [ -n "$missing" ]; then
  warn "缺少:$missing"
  warn "Debian/Ubuntu:  sudo apt update && sudo apt install -y ffmpeg"
  warn "CentOS/RHEL:    sudo yum install -y epel-release && sudo yum install -y ffmpeg"
  warn "conda:          conda install -c conda-forge ffmpeg"
  [ "$CHECK_ONLY" = 1 ] && exit 1
else
  say "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -c1-60)"
fi

# ---------- 3. 显卡 ----------
HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_CUDA=1
  say "检测到 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi
if [ "$WANT_CUDA" = "ask" ]; then
  if [ "$HAS_CUDA" = 1 ]; then WANT_CUDA=1; else WANT_CUDA=0; fi
fi

# ---------- 4. 装依赖 ----------
if [ "$CHECK_ONLY" = 1 ]; then
  if [ -x .venv/bin/python ] && .venv/bin/python -c 'import faster_whisper, numpy, PIL' 2>/dev/null; then
    say "依赖已就绪（.venv）"; exit 0
  fi
  warn "还没装依赖，运行 bash setup.sh"; exit 1
fi

if [ "$FORCE" = 1 ]; then rm -rf .venv; fi
if [ ! -x .venv/bin/python ]; then
  say "创建虚拟环境 .venv …"
  "$PY" -m venv .venv || die "venv 创建失败（Debian 需要 python3-venv 包）"
fi
VPY=.venv/bin/python
"$VPY" -m pip install -q --upgrade pip wheel

PIP_ARGS=(--disable-pip-version-check)
[ -n "${PIP_MIRROR:-}" ] && PIP_ARGS+=(-i "$PIP_MIRROR")

say "安装 requirements.txt …"
"$VPY" -m pip install "${PIP_ARGS[@]}" -r requirements.txt

if [ "$WANT_CUDA" = 1 ]; then
  say "安装 CUDA 运行库（cublas / cudnn，约 1GB）…"
  "$VPY" -m pip install "${PIP_ARGS[@]}" nvidia-cublas-cu12 nvidia-cudnn-cu12 || \
    warn "CUDA 库安装失败，仍可用 CPU 模式"
fi

# ---------- 5. 自检 ----------
say "自检 …"
"$VPY" - <<'PYEOF'
import shutil, sys
ok = True
for mod in ("faster_whisper", "numpy", "PIL"):
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except Exception as exc:
        ok = False
        print(f"  [FAIL] {mod}: {exc}")
for exe in ("ffmpeg", "ffprobe"):
    path = shutil.which(exe)
    print(f"  [{'OK' if path else 'FAIL'}] {exe}: {path or '未找到'}")
    ok = ok and bool(path)
sys.exit(0 if ok else 1)
PYEOF

say "完成。下一步："
echo "  bash run.sh --help                       # 看用法"
echo "  bash run.sh all /path/to/voicefolder     # 一键：转写+多轮判别+MP3(+MP4)"
echo "  bash run.sh build /path/to/voicefolder --image cover.png --title 角色名"
