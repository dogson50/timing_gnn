import argparse
import copy
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "slew",
    "cap",
    "voltage",
    "temp",
    "wp_over_wn",
    "wp_sum",
    "wn_sum",
    "is_inv",
    "log_slew",
    "log_cap",
    "req_p",
    "req_n",
    "rc_p",
    "rc_n",
    "rc_eff",
    "req_eff",
    "inv_v",
    "inv_temp",
    "pn_balance",
    "pol_bit",
]
TARGET_COL = "delay"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_pkl", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--prefix",
        type=str,
        default="dataset_table_group_group_ratio_val50_no_test",
        help="output file prefix",
    )
    p.add_argument("--group_col", type=str, default="group_id")
    p.add_argument("--seed", type=int, default=9294)
    p.add_argument("--val_group_frac", type=float, default=0.50)
    p.add_argument(
        "--candidate_fracs",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 1.0 / 3.0, 0.2, 0.1],
        help="fractions of candidate groups used as labeled target-train groups",
    )
    p.add_argument(
        "--candidate_tags",
        type=str,
        nargs="+",
        default=["g100", "g050", "g033", "g020", "g010"],
        help="tags matching candidate_fracs",
    )
    return p.parse_args()


def ensure_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_COLS:
        if c not in out.columns:
            out[c] = 0.0
    return out


def compute_scalers(df_tgt_train: pd.DataFrame):
    dfx = ensure_numeric_cols(df_tgt_train)
    means = {}
    stds = {}
    for c in NUMERIC_COLS:
        v = pd.to_numeric(dfx[c], errors="coerce").fillna(0.0).astype(float).values
        m = float(np.mean(v)) if len(v) > 0 else 0.0
        s = float(np.std(v)) if len(v) > 0 else 1.0
        if abs(s) < 1e-12:
            s = 1.0
        means[c] = m
        stds[c] = s
    y = pd.to_numeric(df_tgt_train[TARGET_COL], errors="coerce").fillna(0.0).astype(float).values
    y_mean = float(np.mean(y)) if len(y) > 0 else 0.0
    y_std = float(np.std(y)) if len(y) > 0 else 1.0
    if abs(y_std) < 1e-12:
        y_std = 1.0
    return {"mean": means, "std": stds}, {"mean": y_mean, "std": y_std}


def safe_ratio(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return float(a) / float(b)


def main():
    args = parse_args()
    if len(args.candidate_fracs) != len(args.candidate_tags):
        raise ValueError("--candidate_fracs and --candidate_tags must have the same length")

    base_pkl = Path(args.base_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_pkl.open("rb") as f:
        base_obj = pickle.load(f)

    src_df = base_obj.get("src_df", pd.DataFrame()).copy()
    tgt_train = base_obj.get("tgt_train_df", pd.DataFrame()).copy()
    tgt_val = base_obj.get("tgt_val_df", pd.DataFrame()).copy()
    tgt_test = base_obj.get("tgt_test_df", pd.DataFrame()).copy()
    df_tgt_all = pd.concat([tgt_train, tgt_val, tgt_test], ignore_index=True)

    if len(df_tgt_all) == 0:
        raise RuntimeError("target dataframe is empty")
    if args.group_col not in df_tgt_all.columns:
        raise KeyError(f"group column not found: {args.group_col}")

    groups = df_tgt_all[args.group_col].dropna().astype(str).unique().tolist()
    if len(groups) < 2:
        raise RuntimeError(f"need at least 2 target groups, got {len(groups)}")

    rng = np.random.RandomState(args.seed)
    rng.shuffle(groups)

    n_total_groups = len(groups)
    n_val_groups = int(round(n_total_groups * float(args.val_group_frac)))
    n_val_groups = max(1, min(n_total_groups - 1, n_val_groups))
    val_groups = groups[:n_val_groups]
    candidate_groups = groups[n_val_groups:]

    val_set = set(val_groups)
    cand_set = set(candidate_groups)
    df_val_fixed = df_tgt_all[df_tgt_all[args.group_col].astype(str).isin(val_set)].copy().reset_index(drop=True)

    rows = []
    for frac, tag in zip(args.candidate_fracs, args.candidate_tags):
        if frac <= 0 or frac > 1:
            raise ValueError(f"candidate fraction must be in (0,1], got {frac}")

        n_candidate = len(candidate_groups)
        n_pick = int(round(n_candidate * float(frac)))
        n_pick = max(1, min(n_candidate, n_pick))
        sel_groups = candidate_groups[:n_pick]
        sel_set = set(sel_groups)
        unused_groups = candidate_groups[n_pick:]

        df_train = (
            df_tgt_all[df_tgt_all[args.group_col].astype(str).isin(sel_set)]
            .copy()
            .reset_index(drop=True)
        )
        df_test = df_tgt_all.iloc[0:0].copy()

        df_train["is_labeled"] = 1
        df_val = df_val_fixed.copy()
        df_val["is_labeled"] = 1
        df_test["is_labeled"] = 1

        scaler_stats, y_scaler = compute_scalers(df_train)

        meta = copy.deepcopy(base_obj.get("meta", {}))
        split_info = {
            "split_mode": "group_level_target_source_ratio_fixed_val",
            "group_col": args.group_col,
            "seed": int(args.seed),
            "val_group_frac": float(args.val_group_frac),
            "candidate_group_frac": float(frac),
            "tag": str(tag),
            "num_total_groups": int(n_total_groups),
            "num_val_groups": int(len(val_groups)),
            "num_candidate_groups": int(len(candidate_groups)),
            "num_train_groups": int(len(sel_groups)),
            "num_unused_groups": int(len(unused_groups)),
            "val_groups": list(val_groups),
            "train_groups": list(sel_groups),
            "unused_groups": list(unused_groups),
        }
        meta["split_info"] = split_info
        meta["target_group_candidate_frac"] = float(frac)
        meta["target_group_ratio_tag"] = str(tag)
        stats = copy.deepcopy(meta.get("stats", {}))
        stats["num_src"] = int(len(src_df))
        stats["num_tgt_train_labeled"] = int(len(df_train))
        stats["num_tgt_train_unlabeled"] = 0
        stats["num_tgt_val"] = int(len(df_val))
        stats["num_tgt_test"] = 0
        stats["num_tgt_train_groups"] = int(len(sel_groups))
        stats["num_tgt_val_groups"] = int(len(val_groups))
        stats["num_tgt_unused_groups"] = int(len(unused_groups))
        meta["stats"] = stats

        dataset_obj = {
            "meta": meta,
            "scaler_stats": scaler_stats,
            "y_scaler": y_scaler,
            "tgt_train_df": df_train,
            "tgt_val_df": df_val,
            "tgt_test_df": df_test,
            "src_df": src_df.reset_index(drop=True),
        }

        out_name = f"{args.prefix}_{tag}.pkl"
        out_pkl = out_dir / out_name
        with out_pkl.open("wb") as f:
            pickle.dump(dataset_obj, f)

        train_frac_all = safe_ratio(len(sel_groups), n_total_groups)
        train_val_group_ratio = safe_ratio(len(sel_groups), len(val_groups))
        src_groups = (
            src_df[args.group_col].dropna().astype(str).nunique()
            if args.group_col in src_df.columns
            else 0
        )
        tgt_src_group_ratio = safe_ratio(len(sel_groups), src_groups) if src_groups > 0 else 0.0

        rows.append(
            {
                "tag": tag,
                "candidate_frac": float(frac),
                "dataset_pkl_name": out_name,
                "num_total_groups": int(n_total_groups),
                "num_val_groups": int(len(val_groups)),
                "num_candidate_groups": int(len(candidate_groups)),
                "num_train_groups": int(len(sel_groups)),
                "num_unused_groups": int(len(unused_groups)),
                "train_group_frac_of_all": train_frac_all,
                "train_val_group_ratio": train_val_group_ratio,
                "target_source_group_ratio_approx": tgt_src_group_ratio,
                "num_tgt_train_rows": int(len(df_train)),
                "num_tgt_val_rows": int(len(df_val)),
                "num_src_rows": int(len(src_df)),
            }
        )

    out_csv = out_dir / f"{args.prefix}_datasets.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "candidate_frac",
                "dataset_pkl_name",
                "num_total_groups",
                "num_val_groups",
                "num_candidate_groups",
                "num_train_groups",
                "num_unused_groups",
                "train_group_frac_of_all",
                "train_val_group_ratio",
                "target_source_group_ratio_approx",
                "num_tgt_train_rows",
                "num_tgt_val_rows",
                "num_src_rows",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[Saved] {out_csv}")
    for r in rows:
        print(
            f"[Built] {r['tag']}: cand_frac={r['candidate_frac']:.4f}, "
            f"train_groups={r['num_train_groups']}/{r['num_candidate_groups']}, "
            f"train_val_group_ratio={r['train_val_group_ratio']:.3f}, "
            f"dataset={r['dataset_pkl_name']}"
        )


if __name__ == "__main__":
    main()
