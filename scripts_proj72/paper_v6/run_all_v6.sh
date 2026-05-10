#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
PY="${PYTHON_BIN:-/home/xiaojun/anaconda3/envs/gnn_timing1/bin/python}"
LOG_DIR="$ROOT/copilot_train_logs"
mkdir -p "$LOG_DIR"

run_suite() {
  local script="$1"
  local name="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] [start] $name"
  "$PY" "$ROOT/scripts_proj72/paper_v6/$script" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] [done]  $name"
}

run_suite "run_tcdp_arch_search_v6_calib_topo.py" "v6_calib_topo"
run_suite "run_tcdp_arch_search_v6_graph_readout.py" "v6_graph_readout"
run_suite "run_tcdp_arch_search_v6_hgat_capacity.py" "v6_hgat_capacity"
run_suite "run_tcdp_arch_search_v6_combo.py" "v6_combo"

echo "[$(date '+%F %T')] all v6 suites completed"
