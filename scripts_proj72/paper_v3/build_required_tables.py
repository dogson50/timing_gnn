import argparse
import csv
from pathlib import Path


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    summary = load_csv(Path(args.summary_csv).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_key = {(r["exp_name"], r["dataset_pkl_name"]): r for r in summary}
    base_ds = "dataset_table_group_train_val_no_test.pkl"

    # 1) Required comparison table.
    compare_order = [
        ("target_only", "Target-only"),
        ("source_only", "Source-only"),
        ("tcdp_joint", "Joint Source-Target (TCDP)"),
        ("pretrain_finetune", "Pretrain(Source)-Finetune(Target)"),
    ]
    compare_rows = []
    for exp_name, label in compare_order:
        r = by_key.get((exp_name, base_ds))
        if not r:
            continue
        compare_rows.append(
            {
                "method": label,
                "n_seed": r["n_seed"],
                "train_r2_mean": r["train_r2_mean"],
                "train_r2_std": r["train_r2_std"],
                "val_r2_mean": r["val_r2_mean"],
                "val_r2_std": r["val_r2_std"],
                "val_mae_mean": r["val_mae_mean"],
                "val_mae_std": r["val_mae_std"],
                "val_mape_mean": r["val_mape_mean"],
                "val_mape_std": r["val_mape_std"],
            }
        )
    save_csv(
        out_dir / "required_compare_main.csv",
        compare_rows,
        [
            "method",
            "n_seed",
            "train_r2_mean",
            "train_r2_std",
            "val_r2_mean",
            "val_r2_std",
            "val_mae_mean",
            "val_mae_std",
            "val_mape_mean",
            "val_mape_std",
        ],
    )

    # 2) Required ablation table.
    ablate_order = [
        ("tcdp_joint", "TCDP (full)"),
        ("ablate_wo_graph", "w/o graph"),
        ("ablate_wo_domain_calib", "w/o domain calibration"),
        ("ablate_wo_topology_res", "w/o topology residual"),
    ]
    ablate_rows = []
    for exp_name, label in ablate_order:
        r = by_key.get((exp_name, base_ds))
        if not r:
            continue
        ablate_rows.append(
            {
                "method": label,
                "n_seed": r["n_seed"],
                "train_r2_mean": r["train_r2_mean"],
                "train_r2_std": r["train_r2_std"],
                "val_r2_mean": r["val_r2_mean"],
                "val_r2_std": r["val_r2_std"],
                "val_mae_mean": r["val_mae_mean"],
                "val_mae_std": r["val_mae_std"],
                "val_mape_mean": r["val_mape_mean"],
                "val_mape_std": r["val_mape_std"],
            }
        )
    save_csv(
        out_dir / "required_ablation_main.csv",
        ablate_rows,
        [
            "method",
            "n_seed",
            "train_r2_mean",
            "train_r2_std",
            "val_r2_mean",
            "val_r2_std",
            "val_mae_mean",
            "val_mae_std",
            "val_mape_mean",
            "val_mape_std",
        ],
    )

    # 3) Supervision-ratio curve table.
    ratio_rows = []
    for ratio_tag in ["01p", "05p", "10p", "20p"]:
        ds = f"dataset_table_group_train_val_no_test_sup{ratio_tag}.pkl"
        r_joint = by_key.get((f"ratio_tcdp_joint_{ratio_tag}", ds))
        r_tgt = by_key.get((f"ratio_target_only_{ratio_tag}", ds))
        ratio = float(ratio_tag[:2]) / 100.0
        row = {
            "ratio": ratio,
            "ratio_tag": ratio_tag,
            "tcdp_val_r2_mean": r_joint["val_r2_mean"] if r_joint else "",
            "tcdp_val_r2_std": r_joint["val_r2_std"] if r_joint else "",
            "tcdp_val_mae_mean": r_joint["val_mae_mean"] if r_joint else "",
            "tcdp_val_mae_std": r_joint["val_mae_std"] if r_joint else "",
            "tcdp_val_mape_mean": r_joint["val_mape_mean"] if r_joint else "",
            "tcdp_val_mape_std": r_joint["val_mape_std"] if r_joint else "",
            "target_only_val_r2_mean": r_tgt["val_r2_mean"] if r_tgt else "",
            "target_only_val_r2_std": r_tgt["val_r2_std"] if r_tgt else "",
            "target_only_val_mae_mean": r_tgt["val_mae_mean"] if r_tgt else "",
            "target_only_val_mae_std": r_tgt["val_mae_std"] if r_tgt else "",
            "target_only_val_mape_mean": r_tgt["val_mape_mean"] if r_tgt else "",
            "target_only_val_mape_std": r_tgt["val_mape_std"] if r_tgt else "",
        }
        if r_joint and r_tgt:
            row["delta_r2_tcdp_minus_target_only"] = fnum(r_joint["val_r2_mean"]) - fnum(r_tgt["val_r2_mean"])
            row["delta_mae_tcdp_minus_target_only"] = fnum(r_joint["val_mae_mean"]) - fnum(r_tgt["val_mae_mean"])
            row["delta_mape_tcdp_minus_target_only"] = fnum(r_joint["val_mape_mean"]) - fnum(r_tgt["val_mape_mean"])
        else:
            row["delta_r2_tcdp_minus_target_only"] = ""
            row["delta_mae_tcdp_minus_target_only"] = ""
            row["delta_mape_tcdp_minus_target_only"] = ""
        ratio_rows.append(row)
    save_csv(
        out_dir / "required_supervision_ratio_curve.csv",
        ratio_rows,
        [
            "ratio",
            "ratio_tag",
            "tcdp_val_r2_mean",
            "tcdp_val_r2_std",
            "tcdp_val_mae_mean",
            "tcdp_val_mae_std",
            "tcdp_val_mape_mean",
            "tcdp_val_mape_std",
            "target_only_val_r2_mean",
            "target_only_val_r2_std",
            "target_only_val_mae_mean",
            "target_only_val_mae_std",
            "target_only_val_mape_mean",
            "target_only_val_mape_std",
            "delta_r2_tcdp_minus_target_only",
            "delta_mae_tcdp_minus_target_only",
            "delta_mape_tcdp_minus_target_only",
        ],
    )

    # 4) bonus appendix table.
    bonus_order = [
        "bonus_parasitic_replace",
        "bonus_dualmean",
        "bonus_enrich_off_auto",
        "bonus_t16x96_mainline",
        "bonus_shared_backbone_only",
        "bonus_joint_no_src_anneal",
    ]
    bonus_rows = []
    for exp_name in bonus_order:
        r = by_key.get((exp_name, base_ds))
        if not r:
            continue
        bonus_rows.append(r)
    save_csv(
        out_dir / "bonus_variants_summary.csv",
        bonus_rows,
        list(bonus_rows[0].keys()) if bonus_rows else [
            "exp_name",
            "dataset_pkl_name",
            "n_seed",
            "train_r2_mean",
            "train_r2_std",
            "val_r2_mean",
            "val_r2_std",
            "val_mae_mean",
            "val_mae_std",
            "val_mape_mean",
            "val_mape_std",
        ],
    )

    print(f"[Saved] {out_dir / 'required_compare_main.csv'}")
    print(f"[Saved] {out_dir / 'required_ablation_main.csv'}")
    print(f"[Saved] {out_dir / 'required_supervision_ratio_curve.csv'}")
    print(f"[Saved] {out_dir / 'bonus_variants_summary.csv'}")


if __name__ == "__main__":
    main()
