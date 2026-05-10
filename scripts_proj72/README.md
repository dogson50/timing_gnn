# scripts_proj72

Script layout for the timing-transfer experiments.

## Current Paper Scripts

- `paper_v5/`: main paper protocol and reproducibility scripts.
- `paper_v6/`: architecture-search batches.
- `paper_v7/`: later refinement batches.

## Legacy Scripts

- `reproduce/`: early reproduction scripts kept for traceability.
- `run_step5a_feat_multiseed.ps1`: older multi-seed launcher.
- `run_step5a_msselect_multiseed.ps1`: older model-selection launcher.
- `repro_dac23_ep200_bs_1350.sh`: old DAC23-style reproduction launcher.

## Recommended Entry Points

- `paper_v5/run_required_experiments_dual_protocol_r35_best.py`
- `paper_v5/run_table_group_simple_mix_no_transfer.py`
- `paper_v5/run_table_group_gated_fusion.py`

Keep generated logs and checkpoints outside this directory. Summaries should be written under `ICCAD2026_Changxin/paper_materials/tables_*`.
