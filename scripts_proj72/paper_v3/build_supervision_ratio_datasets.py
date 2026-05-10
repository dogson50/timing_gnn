import argparse
import copy
import csv
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_pkl", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    p.add_argument("--seed", type=int, default=9294)
    return p.parse_args()


def fmt_ratio_tag(r: float) -> str:
    return f"{int(round(r * 100)):02d}p"


def main():
    args = parse_args()
    base_pkl = Path(args.base_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_pkl.open("rb") as f:
        base_obj = pickle.load(f)

    if "tgt_train_df" not in base_obj or "tgt_val_df" not in base_obj:
        raise RuntimeError(f"Invalid dataset pkl: {base_pkl}")

    base_train = base_obj["tgt_train_df"].reset_index(drop=True)
    n_total = len(base_train)
    rng = np.random.RandomState(args.seed)

    rows = []
    for r in args.ratios:
        if r <= 0 or r > 1:
            raise ValueError(f"ratio must be in (0,1], got {r}")
        n_keep = max(1, int(round(n_total * r)))
        idx = np.arange(n_total)
        rng.shuffle(idx)
        keep = np.sort(idx[:n_keep])

        obj = copy.deepcopy(base_obj)
        obj["tgt_train_df"] = base_train.iloc[keep].copy().reset_index(drop=True)

        meta = copy.deepcopy(obj.get("meta", {}))
        stats = copy.deepcopy(meta.get("stats", {}))
        stats["num_tgt_train_labeled"] = int(len(obj["tgt_train_df"]))
        stats["num_tgt_val"] = int(len(obj.get("tgt_val_df", [])))
        stats["num_tgt_test"] = int(len(obj.get("tgt_test_df", [])))
        meta["stats"] = stats
        meta["target_supervision_ratio"] = float(r)
        meta["target_supervision_keep"] = int(n_keep)
        meta["target_supervision_seed"] = int(args.seed)
        obj["meta"] = meta

        tag = fmt_ratio_tag(r)
        out_name = f"dataset_table_group_train_val_no_test_sup{tag}.pkl"
        out_pkl = out_dir / out_name
        with out_pkl.open("wb") as f:
            pickle.dump(obj, f)

        rows.append(
            {
                "ratio": r,
                "ratio_tag": tag,
                "n_train_total": n_total,
                "n_train_keep": n_keep,
                "dataset_pkl_name": out_name,
            }
        )

    out_csv = out_dir / "supervision_ratio_datasets.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ratio", "ratio_tag", "n_train_total", "n_train_keep", "dataset_pkl_name"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[Saved] {out_csv}")
    for row in rows:
        print(
            f"[Built] ratio={row['ratio']:.2%}, keep={row['n_train_keep']}/{row['n_train_total']}, "
            f"pkl={row['dataset_pkl_name']}"
        )


if __name__ == "__main__":
    main()

