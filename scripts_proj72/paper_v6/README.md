# paper_v6 architecture search scripts

These suites keep the current best TCDP training schedule and only perturb architecture-related knobs.

Default base:
- dataset: `dataset_table_group_train_val_no_test.pkl`
- epochs/lr/scheduler/anneal: aligned to the best `r35_02` setting
- 3 quick seeds + top-2 expansion to 5 seeds

Suites:
- `run_tcdp_arch_search_v6_calib_topo.py`
- `run_tcdp_arch_search_v6_graph_readout.py`
- `run_tcdp_arch_search_v6_hgat_capacity.py`
- `run_tcdp_arch_search_v6_combo.py`

One-shot:
- Linux/WSL: `bash scripts_proj72/paper_v6/run_all_v6.sh`
- PowerShell: `scripts_proj72/paper_v6/run_all_v6.ps1`

Outputs:
- logs: `ICCAD2026_Changxin/paper_materials/logs_arch_search_<suite_name>/`
- tables: `ICCAD2026_Changxin/paper_materials/tables_arch_search_<suite_name>/`
