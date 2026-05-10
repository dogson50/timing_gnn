# Project Structure

This repository currently keeps three kinds of assets: reusable source code, paper/reproduction scripts, and generated experiment artifacts. The goal is to keep the first two tracked and keep the third either summarized as CSV/Markdown or ignored.

## Core Source

- `src/build_dataset.py`: dataset construction and split logic.
- `src/options.py`: shared CLI options.
- `src/hgat.py`, `src/spi2graph.py`, `src/parse_lib.py`: graph parsing and HGAT utilities.
- `src/train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py`: current stable TCDP training script.
- `src/train_balanced_sampling_sep_mlp_shared_calib_step5a_gated_fusion_opt.py`: gated numeric/graph fusion trial; keep as an experimental negative/diagnostic branch.
- Older `src/train_*.py` files are legacy exploration scripts. Do not use them for paper results unless explicitly referenced by a reproduction script.

## Paper And Reproduction Scripts

- `scripts_proj72/paper_v5/`: current paper protocol scripts, including dual-protocol and group-ratio experiments.
- `scripts_proj72/paper_v6/`: architecture-search experiments.
- `scripts_proj72/paper_v7/`: later refinement experiments.
- `scripts_proj72/reproduce/`: old reproduction shell scripts retained for history.

Recommended paper entry points:

- `scripts_proj72/paper_v5/run_required_experiments_dual_protocol_r35_best.py`
- `scripts_proj72/paper_v5/run_table_group_simple_mix_no_transfer.py`
- `scripts_proj72/paper_v5/run_table_group_gated_fusion.py`

## Paper Materials

- `ICCAD2026_Changxin/paper_materials/data/`: paper datasets and split files.
- `ICCAD2026_Changxin/paper_materials/tables_ext_v5_r35_group_ratio/`: group-level ratio summaries.
- `ICCAD2026_Changxin/paper_materials/tables_simple_mix_table_group/`: simple mix/no-transfer comparison.
- `ICCAD2026_Changxin/paper_materials/tables_gated_fusion_table_group/`: gated-fusion trial summary.
- `ICCAD2026_Changxin/paper_materials/tex/`: paper section drafts.
- `ICCAD2026_Changxin/paper_materials/notes/`: cleanup reports and experiment notes.

## Generated Artifacts

These are ignored and should not be committed unless there is a specific reproduction reason:

- `output/`
- `model*/`, `models/`, `PATH_MODEL/`
- `copilot_train_logs/`
- `ICCAD2026_Changxin/paper_materials/logs*/`
- Python caches: `__pycache__/`, `*.pyc`
- LaTeX intermediates: `*.aux`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.xdv`, `*.out`
- root temporary scripts: `tmp_*.py`

## Cleanup Rule

Do not use broad `git clean -fd` in this repository. It can remove untracked but useful paper tables. Prefer targeted cleanup:

1. Remove caches and LaTeX intermediates.
2. Keep CSV/Markdown summaries before deleting logs.
3. Delete model directories only after the corresponding metrics have been parsed into paper tables.
