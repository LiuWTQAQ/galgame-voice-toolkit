#!/usr/bin/env bash
# 等 mrxyy 的 ASR 缓存写满 -> 构建 mrxyy -> 再跑其余三个文件夹的流水线
set -u
cd /c/Users/11424/Documents/deepseek-harness/default-workspace
PY=/c/Users/11424/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe
CACHE=/e/mrxyy/voicepack_out/asr_small.jsonl
LOG=/e/voicepack_logs

export PYTHONIOENCODING=utf-8
export PYTHONPATH=.:.deps

echo "[orchestrator] $(date '+%H:%M:%S') 等待 mrxyy ASR 缓存写满..." >> "$LOG/orchestrator.log"
for i in $(seq 1 180); do
  n=$(wc -l < "$CACHE" 2>/dev/null || echo 0)
  if [ "$n" -ge 4092 ]; then
    echo "[orchestrator] $(date '+%H:%M:%S') ASR 完成 ($n 句)" >> "$LOG/orchestrator.log"
    break
  fi
  sleep 20
done

echo "[orchestrator] $(date '+%H:%M:%S') 开始构建 mrxyy" >> "$LOG/orchestrator.log"
"$PY" -u -m voicepack build E:/mrxyy --cascade neighbour --title mrxyy > "$LOG/mrxyy_build.log" 2>&1
echo "[orchestrator] $(date '+%H:%M:%S') mrxyy 构建结束 exit=$?" >> "$LOG/orchestrator.log"
tail -5 "$LOG/mrxyy_build.log" >> "$LOG/orchestrator.log"

echo "[orchestrator] $(date '+%H:%M:%S') 启动其余三个文件夹流水线" >> "$LOG/orchestrator.log"
"$PY" -u tools/run_pipeline.py E:\\etyyyyy E:\\sjzmyy E:\\zyqhyy > "$LOG/pipeline.log" 2>&1
echo "[orchestrator] $(date '+%H:%M:%S') 全部完成 exit=$?" >> "$LOG/orchestrator.log"
