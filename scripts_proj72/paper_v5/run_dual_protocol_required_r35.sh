#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p copilot_train_logs
LOG_FILE="copilot_train_logs/paperv5_dual_required_r35_group_ratio.log"

echo "[launch] $(date '+%F %T') start v5-r35 group-ratio dual required queue" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -u scripts_proj72/paper_v5/run_required_experiments_dual_protocol_r35_best.py | tee -a "$LOG_FILE"
