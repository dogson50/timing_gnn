import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


RE_EPOCH_VAL = re.compile(
    r"^e(?P<epoch>\d+),\s*train_loss:(?P<train_loss>[-+0-9.eE]+),\s*r2:(?P<train_r2>[-+0-9.eE]+),\s*"
    r"val_loss:(?P<val_loss>[-+0-9.eE]+),\s*val_r2:(?P<val_r2>[-+0-9.eE]+),\s*"
    r"val_mae:(?P<val_mae>[-+0-9.eE]+),\s*val_mape:(?P<val_mape>[-+0-9.eE]+)"
)
RE_BEST = re.compile(
    r"^\[Best\]\s*epoch:(?P<best_epoch>\d+),\s*val_r2:(?P<val_r2>[-+0-9.eE]+),\s*val_loss:(?P<val_loss>[-+0-9.eE]+)"
    r"(?:,\s*val_mae:(?P<val_mae>[-+0-9.eE]+),\s*val_mape:(?P<val_mape>[-+0-9.eE]+))?"
)
RE_FINAL = re.compile(
    r"^\[FinalTest\]\s*loss:(?P<test_loss>[-+0-9.eE]+),\s*r2:(?P<test_r2>[-+0-9.eE]+)"
    r"(?:,\s*mae:(?P<test_mae>[-+0-9.eE]+),\s*mape:(?P<test_mape>[-+0-9.eE]+))?"
)


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(max(0.0, var))


def parse_log(path: Path):
    epoch_map = {}
    best = None
    final = None
    if not path.exists():
        return epoch_map, best, final
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_EPOCH_VAL.match(line.strip())
        if m:
            ep = int(m.group("epoch")) + 1
            epoch_map[ep] = {
                "train_r2": fnum(m.group("train_r2")),
                "train_loss": fnum(m.group("train_loss")),
                "val_r2": fnum(m.group("val_r2")),
                "val_loss": fnum(m.group("val_loss")),
                "val_mae": fnum(m.group("val_mae")),
                "val_mape": fnum(m.group("val_mape")),
            }
            continue
        m = RE_BEST.match(line.strip())
        if m:
            best = {
                "best_epoch": int(m.group("best_epoch")),
                "val_r2": fnum(m.group("val_r2")),
                "val_loss": fnum(m.group("val_loss")),
                "val_mae": fnum(m.group("val_mae")),
                "val_mape": fnum(m.group("val_mape")),
            }
            continue
        m = RE_FINAL.match(line.strip())
        if m:
            final = {
                "test_r2": fnum(m.group("test_r2")),
                "test_loss": fnum(m.group("test_loss")),
                "test_mae": fnum(m.group("test_mae")),
                "test_mape": fnum(m.group("test_mape")),
            }
    return epoch_map, best, final


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_csv", type=str, required=True)
    p.add_argument("--runs_csv", type=str, required=True)
    p.add_argument("--summary_csv", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    manifest_csv = Path(args.manifest_csv).resolve()
    runs_csv = Path(args.runs_csv).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    runs_csv.parent.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    run_rows = []
    for row in manifest_rows:
        log_path = Path(row["log_path"])
        epoch_map, best, final = parse_log(log_path)
        best_epoch = best["best_epoch"] if best is not None else None
        ep = epoch_map.get(best_epoch, {}) if best_epoch is not None else {}
        run_rows.append(
            {
                **row,
                "best_epoch": best_epoch,
                "train_r2_at_best": ep.get("train_r2"),
                "train_loss_at_best": ep.get("train_loss"),
                "val_r2": (best or {}).get("val_r2"),
                "val_loss": (best or {}).get("val_loss"),
                "val_mae": (best or {}).get("val_mae", ep.get("val_mae")),
                "val_mape": (best or {}).get("val_mape", ep.get("val_mape")),
                "test_r2": (final or {}).get("test_r2"),
                "test_loss": (final or {}).get("test_loss"),
                "test_mae": (final or {}).get("test_mae"),
                "test_mape": (final or {}).get("test_mape"),
            }
        )

    run_fields = list(run_rows[0].keys()) if run_rows else []
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        if run_fields:
            w.writeheader()
            w.writerows(run_rows)

    groups = defaultdict(list)
    for r in run_rows:
        if r.get("status") != "done":
            continue
        groups[(r.get("exp_name"), r.get("dataset_pkl_name"))].append(r)

    summary_rows = []
    for (exp_name, dataset_name), rows in groups.items():
        summary_rows.append(
            {
                "exp_name": exp_name,
                "dataset_pkl_name": dataset_name,
                "n_seed": len(rows),
                "train_r2_mean": mean_std([fnum(x.get("train_r2_at_best")) for x in rows])[0],
                "train_r2_std": mean_std([fnum(x.get("train_r2_at_best")) for x in rows])[1],
                "val_r2_mean": mean_std([fnum(x.get("val_r2")) for x in rows])[0],
                "val_r2_std": mean_std([fnum(x.get("val_r2")) for x in rows])[1],
                "val_mae_mean": mean_std([fnum(x.get("val_mae")) for x in rows])[0],
                "val_mae_std": mean_std([fnum(x.get("val_mae")) for x in rows])[1],
                "val_mape_mean": mean_std([fnum(x.get("val_mape")) for x in rows])[0],
                "val_mape_std": mean_std([fnum(x.get("val_mape")) for x in rows])[1],
            }
        )

    summary_fields = [
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
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[Saved] {runs_csv}")
    print(f"[Saved] {summary_csv}")


if __name__ == "__main__":
    main()

