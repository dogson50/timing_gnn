import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit",
]
TARGET_COL = "delay"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_pkl", type=str, required=True,
                   help="base dataset pkl (usually table_group train/val no-test)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--out_name", type=str, default="dataset_cell_type_train_val_no_test.pkl")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=1.0 / 6.0,
                   help="cell_type-level train ratio; val uses all remaining cells; test is empty")
    p.add_argument("--target_label_ratio", type=float, default=1.0)
    return p.parse_args()


def ensure_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_COLS:
        if c not in out.columns:
            out[c] = 0.0
    return out


def compute_scalers(df_train_pool: pd.DataFrame):
    dfx = ensure_numeric_cols(df_train_pool)
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
    y = pd.to_numeric(df_train_pool[TARGET_COL], errors="coerce").fillna(0.0).astype(float).values
    y_mean = float(np.mean(y)) if len(y) > 0 else 0.0
    y_std = float(np.std(y)) if len(y) > 0 else 1.0
    if abs(y_std) < 1e-12:
        y_std = 1.0
    return {"mean": means, "std": stds}, {"mean": y_mean, "std": y_std}


def main():
    args = parse_args()
    base_pkl = Path(args.base_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_pkl.open("rb") as f:
        base_obj = pickle.load(f)

    tgt_train = base_obj.get("tgt_train_df", pd.DataFrame()).copy()
    tgt_val = base_obj.get("tgt_val_df", pd.DataFrame()).copy()
    tgt_test = base_obj.get("tgt_test_df", pd.DataFrame()).copy()
    src_df = base_obj.get("src_df", pd.DataFrame()).copy()

    if len(tgt_train) == 0 and len(tgt_val) == 0 and len(tgt_test) == 0:
        raise RuntimeError("Base dataset has empty target data.")

    df_tgt_all = pd.concat([tgt_train, tgt_val, tgt_test], ignore_index=True)
    if "cell_type" not in df_tgt_all.columns:
        raise KeyError("target dataframe missing column: cell_type")

    cell_types = df_tgt_all["cell_type"].dropna().astype(str).unique().tolist()
    rng = np.random.RandomState(args.split_seed)
    rng.shuffle(cell_types)
    n_cell = len(cell_types)
    if n_cell < 2:
        raise RuntimeError(f"Need at least 2 cell types for train/val split, got {n_cell}")
    n_train_cell = int(np.floor(args.train_ratio * n_cell))
    n_train_cell = max(1, min(n_train_cell, n_cell - 1))

    train_cells = set(cell_types[:n_train_cell])
    val_cells = set(cell_types[n_train_cell:])

    df_tgt_train_pool = df_tgt_all[df_tgt_all["cell_type"].astype(str).isin(train_cells)].copy()
    df_tgt_val = df_tgt_all[df_tgt_all["cell_type"].astype(str).isin(val_cells)].copy()
    df_tgt_test = df_tgt_all.iloc[0:0].copy()

    # labeled/unlabeled split inside target-train pool
    df_tgt_train_pool = df_tgt_train_pool.sample(frac=1, random_state=args.split_seed).reset_index(drop=True)
    n_train_total = len(df_tgt_train_pool)
    n_train_labeled = int(round(n_train_total * float(args.target_label_ratio)))
    n_train_labeled = max(1, min(n_train_total, n_train_labeled))

    df_tgt_train = df_tgt_train_pool.iloc[:n_train_labeled].copy()
    df_tgt_unlabeled = df_tgt_train_pool.iloc[n_train_labeled:].copy()

    df_tgt_train["is_labeled"] = 1
    df_tgt_unlabeled["is_labeled"] = 0
    df_tgt_val["is_labeled"] = 1
    df_tgt_test["is_labeled"] = 1

    scaler_stats, y_scaler = compute_scalers(df_tgt_train_pool)

    base_meta = copy.deepcopy(base_obj.get("meta", {}))
    split_info = {
        "split_mode": "cell_type_train_val_no_test",
        "ratios": [float(args.train_ratio), 1.0 - float(args.train_ratio), 0.0],
        "seed": int(args.split_seed),
        "train_cells": sorted(train_cells),
        "val_cells": sorted(val_cells),
        "test_cells": [],
        "num_tgt_train_pool": int(len(df_tgt_train_pool)),
        "num_tgt_val": int(len(df_tgt_val)),
        "num_tgt_test": 0,
    }
    base_meta["split_info"] = split_info
    base_meta["target_supervision_ratio"] = float(args.target_label_ratio)
    base_meta["target_supervision_keep"] = int(n_train_labeled)
    base_meta["target_supervision_seed"] = int(args.split_seed)

    stats = copy.deepcopy(base_meta.get("stats", {}))
    stats["num_src"] = int(len(src_df))
    stats["num_tgt_train_labeled"] = int(len(df_tgt_train))
    stats["num_tgt_train_unlabeled"] = int(len(df_tgt_unlabeled))
    stats["num_tgt_val"] = int(len(df_tgt_val))
    stats["num_tgt_test"] = int(len(df_tgt_test))
    base_meta["stats"] = stats

    dataset_obj = {
        "meta": base_meta,
        "scaler_stats": scaler_stats,
        "y_scaler": y_scaler,
        "tgt_train_df": df_tgt_train.reset_index(drop=True),
        "tgt_val_df": df_tgt_val.reset_index(drop=True),
        "tgt_test_df": df_tgt_test.reset_index(drop=True),
        "src_df": src_df.reset_index(drop=True),
    }

    out_pkl = out_dir / args.out_name
    with out_pkl.open("wb") as f:
        pickle.dump(dataset_obj, f)

    # side outputs for quick inspection
    with (out_dir / "dataset_cell_type_split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {out_pkl}")
    print(
        f"[Stats] src={len(src_df)}, "
        f"tgt_train_labeled={len(df_tgt_train)}, tgt_train_unlabeled={len(df_tgt_unlabeled)}, "
        f"tgt_val={len(df_tgt_val)}, tgt_test={len(df_tgt_test)}"
    )
    print(f"[Cells] train={len(train_cells)}, val={len(val_cells)}, test=0")


if __name__ == "__main__":
    main()

