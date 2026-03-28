r"""Diagnose arc-conditioning signal quality from dataset.pkl.

Outputs are written to output/arc_diag by default.
"""

import argparse
import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "delay"
ARC_COLS = ["cell_type", "from_pin", "to_pin", "pol"]


def _norm_pin(v) -> str:
    if pd.isna(v):
        return "<UNK>"
    s = str(v).strip()
    return s if s else "<UNK>"


def _norm_pol(v) -> str:
    if pd.isna(v):
        return "fall"
    s = str(v).strip().lower()
    return "rise" if s == "rise" else "fall"


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=ARC_COLS + [TARGET_COL])

    out = df.copy()
    if "cell_type" not in out.columns:
        out["cell_type"] = "<UNK_CELL>"
    if "from_pin" not in out.columns:
        out["from_pin"] = "<UNK>"
    if "to_pin" not in out.columns:
        out["to_pin"] = "<UNK>"
    if "pol" not in out.columns:
        out["pol"] = "fall"
    if TARGET_COL not in out.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    out["cell_type"] = out["cell_type"].astype(str)
    out["from_pin"] = out["from_pin"].map(_norm_pin)
    out["to_pin"] = out["to_pin"].map(_norm_pin)
    out["pol"] = out["pol"].map(_norm_pol)
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return out


def _load_dataset_pkl(data_dir: str, pkl_name: str) -> Dict:
    pkl_path = os.path.join(data_dir, pkl_name)
    if not os.path.exists(pkl_path) and pkl_name != "dataset.pkl":
        pkl_path = os.path.join(data_dir, "dataset.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cannot find dataset pkl under: {data_dir}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _arc_key_df(df: pd.DataFrame) -> pd.Series:
    return (
        df["cell_type"].astype(str)
        + "|"
        + df["from_pin"].astype(str)
        + "|"
        + df["to_pin"].astype(str)
        + "|"
        + df["pol"].astype(str)
    )


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true)) if y_true.size > 0 else 0.0
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-18:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _split_basic_stats(name: str, df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return {
            "split": name,
            "rows": 0,
            "unique_cell_types": 0,
            "unique_arc_keys": 0,
            "singleton_arc_ratio": 0.0,
            "top1_arc_share": 0.0,
        }

    arc_key = _arc_key_df(df)
    cnt = arc_key.value_counts()
    singleton_ratio = float((cnt == 1).sum() / max(1, len(cnt)))
    top1_share = float(cnt.iloc[0] / max(1, len(df))) if len(cnt) > 0 else 0.0
    return {
        "split": name,
        "rows": int(len(df)),
        "unique_cell_types": int(df["cell_type"].nunique()),
        "unique_arc_keys": int(len(cnt)),
        "singleton_arc_ratio": singleton_ratio,
        "top1_arc_share": top1_share,
    }


def _cell_arc_stats(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "n_samples",
                "n_unique_arcs",
                "samples_per_arc",
                "singleton_arc_ratio",
                "max_arc_share",
            ]
        )

    g = df.groupby("cell_type", sort=False)
    rows = []
    for cell_type, part in g:
        arc_count = _arc_key_df(part).value_counts()
        n_samples = int(len(part))
        n_arc = int(len(arc_count))
        rows.append(
            {
                "cell_type": cell_type,
                "n_samples": n_samples,
                "n_unique_arcs": n_arc,
                "samples_per_arc": float(n_samples / max(1, n_arc)),
                "singleton_arc_ratio": float((arc_count == 1).sum() / max(1, n_arc)),
                "max_arc_share": float(arc_count.iloc[0] / max(1, n_samples)) if n_arc > 0 else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["n_samples", "samples_per_arc"], ascending=[False, True]).reset_index(drop=True)


def _top_arcs(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["cell_type", "from_pin", "to_pin", "pol", "count", "share"])

    cnt = (
        df.groupby(ARC_COLS, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    cnt["share"] = cnt["count"] / max(1, len(df))
    return cnt.head(topk)


def _overlap_stats(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    if len(val_df) == 0:
        return {
            "val_rows": 0,
            "arc_seen_ratio": 0.0,
            "arc_unseen_ratio": 0.0,
            "cell_seen_ratio": 0.0,
            "cell_unseen_ratio": 0.0,
        }

    train_arc = set(_arc_key_df(train_df).tolist())
    val_arc = _arc_key_df(val_df)
    arc_seen = val_arc.isin(train_arc)

    train_cell = set(train_df["cell_type"].astype(str).tolist())
    val_cell = val_df["cell_type"].astype(str)
    cell_seen = val_cell.isin(train_cell)

    return {
        "val_rows": int(len(val_df)),
        "arc_seen_ratio": float(arc_seen.mean()),
        "arc_unseen_ratio": float(1.0 - arc_seen.mean()),
        "cell_seen_ratio": float(cell_seen.mean()),
        "cell_unseen_ratio": float(1.0 - cell_seen.mean()),
    }


def _group_mean_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    keys: Tuple[str, ...],
    fallback_series: pd.Series,
    global_mean: float,
) -> np.ndarray:
    if len(eval_df) == 0:
        return np.zeros((0,), dtype=np.float64)

    if len(train_df) == 0:
        return np.full((len(eval_df),), global_mean, dtype=np.float64)

    gmean = train_df.groupby(list(keys))[TARGET_COL].mean()
    idx = [eval_df[k].values for k in keys]
    pred = gmean.reindex(pd.MultiIndex.from_arrays(idx)).values

    if fallback_series is not None:
        fallback = fallback_series.reindex(eval_df["cell_type"].values).values
        pred = np.where(np.isnan(pred), fallback, pred)

    pred = np.where(np.isnan(pred), global_mean, pred)
    return np.asarray(pred, dtype=np.float64)


def _signal_gain_report(tgt_train: pd.DataFrame, tgt_val: pd.DataFrame) -> Dict:
    if len(tgt_train) == 0:
        return {
            "train": {},
            "val": {},
        }

    global_mean = float(tgt_train[TARGET_COL].mean())
    cell_mean = tgt_train.groupby("cell_type")[TARGET_COL].mean()

    y_tr = tgt_train[TARGET_COL].values.astype(np.float64)
    y_va = tgt_val[TARGET_COL].values.astype(np.float64) if len(tgt_val) > 0 else np.zeros((0,), dtype=np.float64)

    pred_global_tr = np.full_like(y_tr, global_mean)
    pred_cell_tr = _group_mean_predict(tgt_train, tgt_train, ("cell_type",), None, global_mean)
    pred_arc_tr = _group_mean_predict(
        tgt_train,
        tgt_train,
        ("cell_type", "from_pin", "to_pin", "pol"),
        fallback_series=cell_mean,
        global_mean=global_mean,
    )

    report = {
        "train": {
            "r2_global": _safe_r2(y_tr, pred_global_tr),
            "r2_cell_mean": _safe_r2(y_tr, pred_cell_tr),
            "r2_arc_mean": _safe_r2(y_tr, pred_arc_tr),
        },
        "val": {},
    }
    report["train"]["arc_gain_over_cell"] = report["train"]["r2_arc_mean"] - report["train"]["r2_cell_mean"]

    if len(tgt_val) > 0:
        pred_global_va = np.full_like(y_va, global_mean)
        pred_cell_va = _group_mean_predict(tgt_train, tgt_val, ("cell_type",), None, global_mean)
        pred_arc_va = _group_mean_predict(
            tgt_train,
            tgt_val,
            ("cell_type", "from_pin", "to_pin", "pol"),
            fallback_series=cell_mean,
            global_mean=global_mean,
        )
        report["val"] = {
            "r2_global": _safe_r2(y_va, pred_global_va),
            "r2_cell_mean": _safe_r2(y_va, pred_cell_va),
            "r2_arc_mean": _safe_r2(y_va, pred_arc_va),
        }
        report["val"]["arc_gain_over_cell"] = report["val"]["r2_arc_mean"] - report["val"]["r2_cell_mean"]

    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose arc-conditioning signal in dataset.pkl")
    parser.add_argument("--data_save_path", type=str, default="output", help="Path containing dataset.pkl")
    parser.add_argument("--dataset_pkl_name", type=str, default="dataset.pkl", help="dataset pkl filename")
    parser.add_argument("--out_dir", type=str, default="output/arc_diag", help="Directory to write reports")
    parser.add_argument("--topk", type=int, default=200, help="Top K arc keys to save")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    obj = _load_dataset_pkl(args.data_save_path, args.dataset_pkl_name)
    src_df = _prepare_df(obj.get("src_df"))
    tgt_train_df = _prepare_df(obj.get("tgt_train_df"))
    tgt_val_df = _prepare_df(obj.get("tgt_val_df"))

    split_stats = [
        _split_basic_stats("src", src_df),
        _split_basic_stats("tgt_train", tgt_train_df),
        _split_basic_stats("tgt_val", tgt_val_df),
    ]

    overlap = _overlap_stats(tgt_train_df, tgt_val_df)
    signal_gain = _signal_gain_report(tgt_train_df, tgt_val_df)

    summary = {
        "split_stats": split_stats,
        "tgt_val_overlap_vs_tgt_train": overlap,
        "signal_gain_group_mean": signal_gain,
        "quick_read": {
            "arc_is_long_tail_hint": float(split_stats[1]["singleton_arc_ratio"]) > 0.5 if len(split_stats) > 1 else False,
            "val_arc_oov_high_hint": float(overlap.get("arc_unseen_ratio", 0.0)) > 0.3,
            "arc_adds_signal_hint": float(signal_gain.get("val", {}).get("arc_gain_over_cell", 0.0)) > 0.01,
        },
    }

    pd.DataFrame(split_stats).to_csv(os.path.join(args.out_dir, "split_stats.csv"), index=False)
    _cell_arc_stats(tgt_train_df).to_csv(os.path.join(args.out_dir, "tgt_train_cell_arc_stats.csv"), index=False)
    _top_arcs(tgt_train_df, args.topk).to_csv(os.path.join(args.out_dir, "tgt_train_top_arcs.csv"), index=False)

    if len(tgt_val_df) > 0:
        val_arc = _arc_key_df(tgt_val_df)
        train_arc_set = set(_arc_key_df(tgt_train_df).tolist())
        val_detail = tgt_val_df[ARC_COLS + [TARGET_COL]].copy()
        val_detail["arc_seen_in_train"] = val_arc.isin(train_arc_set).astype(int)
        val_detail.to_csv(os.path.join(args.out_dir, "tgt_val_arc_seen_detail.csv"), index=False)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[ArcDiag] Done")
    print(f"[ArcDiag] Reports written to: {args.out_dir}")
    print("[ArcDiag] Key summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
