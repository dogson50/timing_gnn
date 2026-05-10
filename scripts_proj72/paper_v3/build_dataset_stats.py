import argparse
import csv
import pickle
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_md", type=str, required=True)
    return p.parse_args()


def arc_count(df: pd.DataFrame) -> int:
    if df is None or len(df) == 0:
        return 0
    cols = [c for c in ["cell_type", "from_pin", "to_pin", "timing_sense"] if c in df.columns]
    if not cols:
        return int(len(df))
    return int(df[cols].astype(str).drop_duplicates().shape[0])


def uniq_count(df: pd.DataFrame, col: str) -> int:
    if df is None or len(df) == 0 or col not in df.columns:
        return 0
    return int(df[col].astype(str).nunique())


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pkl_path in sorted(data_dir.glob("dataset_table_group_train_val_no_test*.pkl")):
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        src = obj.get("src_df")
        tr = obj.get("tgt_train_df")
        va = obj.get("tgt_val_df")
        te = obj.get("tgt_test_df")
        tgt_all = pd.concat([x for x in [tr, va, te] if x is not None and len(x) > 0], ignore_index=True) if any(
            x is not None and len(x) > 0 for x in [tr, va, te]
        ) else pd.DataFrame()

        rows.append(
            {
                "dataset_pkl_name": pkl_path.name,
                "src_num": int(len(src)) if src is not None else 0,
                "src_cells": uniq_count(src, "cell_type"),
                "src_arcs": arc_count(src),
                "src_groups": uniq_count(src, "group_id"),
                "tgt_cells": uniq_count(tgt_all, "cell_type"),
                "tgt_arcs": arc_count(tgt_all),
                "tgt_groups": uniq_count(tgt_all, "group_id"),
                "num_train": int(len(tr)) if tr is not None else 0,
                "num_val": int(len(va)) if va is not None else 0,
                "num_test": int(len(te)) if te is not None else 0,
                "train_cells": uniq_count(tr, "cell_type"),
                "train_arcs": arc_count(tr),
                "train_groups": uniq_count(tr, "group_id"),
                "val_cells": uniq_count(va, "cell_type"),
                "val_arcs": arc_count(va),
                "val_groups": uniq_count(va, "group_id"),
                "test_cells": uniq_count(te, "cell_type"),
                "test_arcs": arc_count(te),
                "test_groups": uniq_count(te, "group_id"),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_pkl_name",
                "src_num",
                "src_cells",
                "src_arcs",
                "src_groups",
                "tgt_cells",
                "tgt_arcs",
                "tgt_groups",
                "num_train",
                "num_val",
                "num_test",
                "train_cells",
                "train_arcs",
                "train_groups",
                "val_cells",
                "val_arcs",
                "val_groups",
                "test_cells",
                "test_arcs",
                "test_groups",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    lines = [
        "| Dataset | Src Num | Src Cells | Src Arcs | Src Groups | Tgt Cells | Tgt Arcs | Tgt Groups | Train | Val | Test | Train Cells/Arcs/Groups | Val Cells/Arcs/Groups | Test Cells/Arcs/Groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset_pkl_name']} | {r['src_num']} | {r['src_cells']} | {r['src_arcs']} | {r['src_groups']} | "
            f"{r['tgt_cells']} | {r['tgt_arcs']} | {r['tgt_groups']} | {r['num_train']} | {r['num_val']} | {r['num_test']} | "
            f"{r['train_cells']}/{r['train_arcs']}/{r['train_groups']} | "
            f"{r['val_cells']}/{r['val_arcs']}/{r['val_groups']} | "
            f"{r['test_cells']}/{r['test_arcs']}/{r['test_groups']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
