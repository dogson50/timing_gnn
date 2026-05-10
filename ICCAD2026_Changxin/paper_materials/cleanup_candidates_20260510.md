# Cleanup Candidates 2026-05-10

This note separates generated files that are usually safe to remove from files that should be kept for paper reproducibility.

## Keep

- `src/train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py`
- `src/train_balanced_sampling_sep_mlp_shared_calib_step5a_gated_fusion_opt.py`
- `src/options.py`
- `scripts_proj72/paper_v5/`
- `scripts_proj72/paper_v6/`
- `scripts_proj72/paper_v7/`
- `ICCAD2026_Changxin/paper_materials/data/`
- `ICCAD2026_Changxin/paper_materials/tables_ext_v5_r35_group_ratio/`
- `ICCAD2026_Changxin/paper_materials/tables_simple_mix_table_group/`
- `ICCAD2026_Changxin/paper_materials/tables_gated_fusion_table_group/`
- `ICCAD2026_Changxin/paper_materials/tex/*.tex`
- `ICCAD2026_Changxin/paper_materials/group_ratio_experiment_summary.md`

## Safe To Delete After Confirming No Active Run

- Python caches: `__pycache__/`
- LaTeX intermediates: `*.aux`, `*.log`, `*.out`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.xdv`
- Temporary root scripts: `tmp_*.py`
- IDE local state: `.idea/workspace.xml`

## Delete Only After CSV Summary Is Preserved

- Training logs under `copilot_train_logs/`
- Paper experiment logs under `ICCAD2026_Changxin/paper_materials/logs_*`
- Old architecture-search table/log rounds that are not used in the paper narrative:
  - `tables_arch_search_round6/` through older exploratory rounds
  - corresponding `logs_arch_search_round*/`

## High-Risk: Do Not Delete Blindly

- `output/*.pkl` and `output/*.csv`: these may be current datasets.
- `model_*` directories: delete only if the corresponding result has been parsed into a CSV and the checkpoint is not needed.
- `rawdata/` and `rawdata-130/`: source data, not generated scratch.
- Any tracked file reported by `git ls-files`.

## Recommended Cleanup Rule

Do not run broad `git clean -fd` in this repository. Use targeted deletion, or first run:

```powershell
git clean -nd
git clean -ndX
```

Then delete only the confirmed generated categories above.
