#!/usr/bin/env bash
# 统一启动器：自动挑 Python、把 vendor/ 与 .deps/ 塞进 PYTHONPATH，再转发给 python -m voicepack
#
#   bash run.sh --help
#   bash run.sh scan  "/data/voice/mrxyy"
#   bash run.sh all   "/data/voice/mrxyy" --passes 3
#   bash run.sh build "/data/voice/mrxyy" --image cover.png --title 角色名
#   bash run.sh serve --folder "/data/voice/mrxyy" --port 8765
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

PY=""
for cand in "${VOICEPACK_PYTHON:-}" .venv/bin/python python3 python; do
  [ -n "$cand" ] || continue
  if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { echo "找不到 Python，请先跑 bash setup.sh" >&2; exit 1; }

# setup.bat 装的依赖在 vendor/，旧版在 .deps/；venv 装的直接 import 得到
export PYTHONPATH="$ROOT/vendor:$ROOT/.deps${PYTHONPATH:+:$PYTHONPATH}"

# N 卡运行库（pip 装的 cublas/cudnn 在 .deps/nvidia/*/lib）
while IFS= read -r d; do
  [ -n "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done < <(find "$ROOT/.deps/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null)

exec "$PY" -m voicepack "$@"
