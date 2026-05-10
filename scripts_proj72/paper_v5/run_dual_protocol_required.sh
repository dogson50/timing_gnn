#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_FILE="$ROOT_DIR/copilot_train_logs/paperv5_dual_required.log"
PY="/home/xiaojun/anaconda3/envs/gnn_timing1/bin/python"
SCRIPT="$ROOT_DIR/scripts_proj72/paper_v5/run_required_experiments_dual_protocol.py"
MON="$ROOT_DIR/copilot_train_logs/arch_queue_monitor.log"

mkdir -p "$ROOT_DIR/copilot_train_logs"
cd "$ROOT_DIR"

{
  echo "[$(date '+%F %T')] continue: start paperv5 dual required queue (table_group + cell_type)."
} | tee -a "$LOG_FILE" "$MON"

"$PY" "$SCRIPT" 2>&1 | tee -a "$LOG_FILE"

{
  echo "[$(date '+%F %T')] continue: paperv5 dual required queue finished."
} | tee -a "$LOG_FILE" "$MON"

