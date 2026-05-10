#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
PY="${PYTHON_BIN:-/home/xiaojun/anaconda3/envs/gnn_timing1/bin/python}"
LOG_DIR="$ROOT/copilot_train_logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/v7_main.log"
echo "[$(date '+%F %T')] [start] v7_main"
"$PY" "$ROOT/scripts_proj72/paper_v7/run_tcdp_arch_search_v7_main.py" 2>&1 | tee "$LOG"
echo "[$(date '+%F %T')] [done]  v7_main"
