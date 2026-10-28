#!/usr/bin/env bash
# =============================================================================
# voicepack 远程 GPU 环境一键安装（Ubuntu 22.04，AutoDL / RunPod / vast.ai 通用）
#
# 用法（在远端机器上，项目目录里）：
#     bash deploy/remote_setup.sh              # 装到当前项目下 .venv-remote/
#     bash deploy/remote_setup.sh /root/vp     # 指定项目目录
#
# 可选环境变量：
#     PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple   # 国内 pip 镜像
#     HF_ENDPOINT=https://hf-mirror.com                      # 国内 HuggingFace 镜像
#     MODEL=small                                            # smoke test 模型（默认 small）
#     SKIP_SMOKE=1                                           # 只装依赖不跑 smoke test
#     PYTHON_BIN=python3                                     # 指定 Python 解释器
#
# 脚本做的事：
#   1. 检查系统/Python/ffmpeg/磁盘/GPU，缺 ffmpeg 就用 apt 装；
#   2. 建独立 venv（.venv-remote），装 faster-whisper（含 ctranslate2）；
#   3. 有 nvidia-smi 时额外装 nvidia-cublas-cu12 / nvidia-cudnn-cu12==9.*
#      （ctranslate2 4.x 在 CUDA 12 下需要这两个 pip wheel 提供 cuBLAS/cuDNN）；
#   4. 生成 1 秒静音 wav，先用 CPU int8 跑一次 WhisperModel 验证依赖完整，
#      有 GPU 再在 CUDA float16 下跑一次（这一步能提前暴露 cuDNN 版本不对的问题）；
#   5. 打印下一步要执行的 worker 命令。
#
# 注意：smoke test 会从 HuggingFace 下载模型（small 约 480MB）。国内机器务必设置
#       HF_ENDPOINT=https://hf-mirror.com，否则可能卡住。
# =============================================================================
set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-${PROJECT_DIR}/.venv-remote}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_MIRROR="${PIP_MIRROR:-}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
MODEL="${MODEL:-small}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

info()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[setup][警告]\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m[setup][错误]\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 0. 基本信息
# --------------------------------------------------------------------------- #
info "项目目录: ${PROJECT_DIR}"
if [ ! -d "${PROJECT_DIR}/voicepack" ]; then
    fail "在 ${PROJECT_DIR} 下找不到 voicepack/，请把项目目录传上来或检查第一个参数"
fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "找不到 ${PYTHON_BIN}，请先安装 Python 3.10+"
PY_VER="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "Python: ${PYTHON_BIN} (${PY_VER})"
"${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "需要 Python 3.10+（faster-whisper 1.2 要求），当前 ${PY_VER}"

if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    info "系统: ${PRETTY_NAME:-unknown} (${ID:-?} ${VERSION_ID:-?})"
    [ "${ID:-}" = "ubuntu" ] || warn "本脚本按 Ubuntu 22.04 写的，其他发行版可能需要手动装 ffmpeg/venv"
fi

DISK_AVAIL_GB="$(df -Pk "${PROJECT_DIR}" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
info "剩余磁盘: ${DISK_AVAIL_GB} GB"
[ "${DISK_AVAIL_GB}" -ge 8 ] || warn "磁盘偏小：模型 + 下载临时文件建议留 8GB 以上"

command -v nvidia-smi >/dev/null 2>&1 && HAS_GPU=1 || HAS_GPU=0
if [ "${HAS_GPU}" = "1" ]; then
    info "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1)"
    info "CUDA: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n1)"
else
    warn "没有检测到 nvidia-smi：将只装 CPU 版本（worker 可以用 --device cpu，但会很慢）"
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 && SUDO="sudo" || warn "非 root 且没有 sudo，缺系统包时需要手动安装"
fi

# --------------------------------------------------------------------------- #
# 1. 系统依赖：ffmpeg / python3-venv
# --------------------------------------------------------------------------- #
NEED_APT=""
command -v ffmpeg  >/dev/null 2>&1 || NEED_APT="${NEED_APT} ffmpeg"
command -v ffprobe >/dev/null 2>&1 || NEED_APT="${NEED_APT} ffmpeg"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || NEED_APT="${NEED_APT} python3-venv"
if [ -n "${NEED_APT}" ]; then
    info "通过 apt 安装:${NEED_APT}"
    if command -v apt-get >/dev/null 2>&1; then
        ${SUDO} apt-get update -y
        ${SUDO} apt-get install -y ${NEED_APT}
    else
        fail "缺少${NEED_APT}，且当前系统没有 apt-get，请手动安装后再跑本脚本"
    fi
else
    info "ffmpeg / venv 已就绪"
fi
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg 仍不可用"
command -v ffprobe >/dev/null 2>&1 || fail "ffprobe 仍不可用"

# --------------------------------------------------------------------------- #
# 2. venv + pip 依赖
# --------------------------------------------------------------------------- #
if [ ! -x "${VENV}/bin/python" ]; then
    info "创建 venv: ${VENV}"
    "${PYTHON_BIN}" -m venv "${VENV}"
else
    info "复用已有 venv: ${VENV}"
fi

PIP_ARGS=()
if [ -n "${PIP_MIRROR}" ]; then
    PIP_ARGS=(--index-url "${PIP_MIRROR}")
    info "pip 镜像: ${PIP_MIRROR}"
fi

pip_install() {
    "${VENV}/bin/pip" install "${PIP_ARGS[@]}" "$@"
}

info "升级 pip"
pip_install --upgrade pip

info "安装 faster-whisper（含 ctranslate2 / onnxruntime / av）"
pip_install "faster-whisper>=1.2.0,<2" numpy

if [ "${HAS_GPU}" = "1" ]; then
    # ctranslate2 4.x 不再自带 cuBLAS/cuDNN 动态库；CUDA 12 下用 pip wheel 补齐，
    # 否则会报 "Could not locate cudnn_ops.so" / "libcublas.so.12 cannot open"。
    info "安装 CUDA 12 运行库（cuBLAS + cuDNN 9）"
    pip_install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
fi

# --------------------------------------------------------------------------- #
# 3. 环境变量（模型缓存目录、HuggingFace 镜像）
# --------------------------------------------------------------------------- #
if [ -n "${HF_ENDPOINT}" ]; then
    info "HuggingFace 镜像: ${HF_ENDPOINT}"
    export HF_ENDPOINT
fi
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf-cache}"
mkdir -p "${HF_HOME}"
info "模型缓存目录: ${HF_HOME}"

if [ "${SKIP_SMOKE}" = "1" ]; then
    info "SKIP_SMOKE=1，跳过 smoke test"
else
    # ----------------------------------------------------------------------- #
    # 4. smoke test：静音 wav + 一次 CPU 转写（有 GPU 再验证 CUDA）
    # ----------------------------------------------------------------------- #
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "${TMP_DIR}"' EXIT
    SMOKE_WAV="${TMP_DIR}/silence.wav"
    info "生成 1 秒静音 wav 用于 smoke test"
    ffmpeg -v error -f lavfi -i "anullsrc=r=16000:cl=mono" -t 1.0 -y "${SMOKE_WAV}"

    info "smoke test：加载 ${MODEL} 并转写静音（首次会下载模型，可能要几分钟）"
    "${VENV}/bin/python" - "${SMOKE_WAV}" "${MODEL}" <<'PY'
import sys

wav, model_name = sys.argv[1], sys.argv[2]

from faster_whisper import WhisperModel

print(f"[smoke] faster-whisper 导入成功")
model = WhisperModel(model_name, device="cpu", compute_type="int8")
segments, info = model.transcribe(wav, language="ja", beam_size=1, vad_filter=False)
text = "".join(seg.text for seg in segments)
print(f"[smoke] CPU int8 转写静音 OK（language={info.language}, 文本={text!r}）")

try:
    import ctranslate2

    count = ctranslate2.get_cuda_device_count()
except Exception as exc:  # noqa: BLE001
    count = 0
    print(f"[smoke] 读取 CUDA 设备数失败: {exc}")
print(f"[smoke] ctranslate2 CUDA 设备数: {count}")
if count > 0:
    # 这一步会真正加载 cuBLAS/cuDNN，缺库会在这里报错，比等到正式转写才发现好
    cuda_model = WhisperModel(model_name, device="cuda", compute_type="float16")
    cuda_segments, _ = cuda_model.transcribe(wav, language="ja", beam_size=1, vad_filter=False)
    list(cuda_segments)  # 触发实际计算
    print("[smoke] CUDA float16 加载 + 转写 OK（cuBLAS/cuDNN 正常）")
PY
fi

# --------------------------------------------------------------------------- #
# 5. 完成提示
# --------------------------------------------------------------------------- #
cat <<EOF

==============================================================================
[setup] 安装完成 ✅

  venv          : ${VENV}
  Python        : ${VENV}/bin/python
  HF 缓存       : ${HF_HOME}
  ffmpeg        : $(ffmpeg -version 2>/dev/null | head -n1)

接下来：
  # 1) 激活环境（可选，直接写全路径也行）
  source ${VENV}/bin/activate

  # 2) 在线模式：本地已 serve 时，在远端跑 worker
  #    （没有公网入口时，先在本机执行 ssh -R 8765:127.0.0.1:8765 <远端>）
  python -m voicepack.remote worker --server http://127.0.0.1:8765 --token <token> \\
      --model large-v3 --device cuda --compute-type float16

  # 3) 离线任务包模式
  python -m voicepack.remote worker --jobs-dir /root/taskpack \\
      --out /root/results.jsonl --model large-v3 --device cuda --compute-type float16

  # 4) 国内机器记得设置镜像（每开一个新 shell 都要设）
  export HF_ENDPOINT=https://hf-mirror.com

  详细流程见 docs/remote_gpu.md
==============================================================================
EOF
