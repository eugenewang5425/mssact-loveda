#!/bin/bash
# 等待所有 run_queue* python 进程退出后, 启动训练策略(裁剪)队列
LOG="C:/Users/Administrator/.zcode/cli/exec/sess_79382b4e-b5fc-48a3-9f7f-73768fa1f2f2/queue_crop.log"
PY="C:/Users/Administrator/.conda/envs/pytorch/python.exe"
cd "D:/AI点子/遥感transformer分类毕业论文完善/loveda" || exit 1
echo "=== watcher 启动, 等待其他队列结束 $(date '+%F %T') ===" >> "$LOG"
while true; do
  n=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" 2>/dev/null | grep -c "run_queue")
  if [ "$n" -eq 0 ]; then break; fi
  sleep 300
done
echo "=== 其他队列已全部结束, 启动 crop 队列 $(date '+%F %T') ===" >> "$LOG"
nohup "$PY" -u run_queue_crop.py --no-wait >> "$LOG" 2>&1 &
echo "=== crop 队列 已启动 PID $! ===" >> "$LOG"
