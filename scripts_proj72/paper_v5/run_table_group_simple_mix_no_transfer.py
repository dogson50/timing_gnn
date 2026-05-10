import csv
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


BEST_RE = re.compile(
    r"\[Best\]\s*epoch:(?P<epoch>\d+),\s*"
    r"val_r2:(?P<val_r2>[-+0-9.eE]+),\s*"
    r"val_loss:(?P<val_loss>[-+0-9.eE]+),\s*"
    r"val_mae:(?P<val_mae>[-+0-9.eE]+),\s*"
    r"val_mape:(?P<val_mape>[-+0-9.eE]+)"
)


@dataclass
class RunResult:
    seed: int
    epoch: int
    val_r2: float
    val_loss: float
    val_mae: float
    val_mape: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_best_from_log(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m_all = list(BEST_RE.finditer(text))
    if not m_all:
        return None
    m = m_all[-1]
    return RunResult(
        seed=-1,
        epoch=int(m.group("epoch")),
        val_r2=float(m.group("val_r2")),
        val_loss=float(m.group("val_loss")),
        val_mae=float(m.group("val_mae")),
        val_mape=float(m.group("val_mape")),
    )


def build_cmd(py: Path, train_py: Path, model_dir: str, seed: int):
    return [
        str(py),
        str(train_py),
        "--model_saving_dir",
        model_dir,
        "--data_save_path",
        "ICCAD2026_Changxin/paper_materials/data",
        "--dataset_pkl_name",
        "dataset_table_group_train_val_no_test.pkl",
        "--gpu",
        "0",
        "--seed",
        str(seed),
        "--freeze_hgat",
        "--num_epoch",
        "320",
        "--batch_size",
        "2048",
        "--learning_rate",
        "1.0e-3",
        "--enc_lr_scale",
        "0.05",
        "--weight_decay",
        "5e-5",
        "--mlp_dropout",
        "0.2",
        "--hgat_l2_norm",
        "--z_noise_std",
        "0.01",
        "--enrich_parasitic_net_feat",
        "--hgat_net_feat_mode",
        "base",
        "--hgat_par_cap_weight",
        "0.6",
        "--disable_domain_calibration",
        "--src_loss_anneal_start",
        "0",
        "--src_loss_anneal_end",
        "0",
        "--src_loss_final_scale",
        "1.0",
        "--target_loss_weight",
        "1.0",
        "--loss_weight_45",
        "1.0",
        "--lr_scheduler",
        "plateau",
        "--plateau_factor",
        "0.5",
        "--plateau_patience",
        "4",
        "--plateau_threshold",
        "3e-4",
        "--plateau_min_lr",
        "1e-6",
        "--early_stop_patience",
        "22",
        "--early_stop_min_delta",
        "8e-4",
        "--num_workers",
        "0",
        "--val_eval_interval",
        "5",
        "--epoch_log_interval",
        "10",
        "--skip_train_r2",
        "--fast_eval_loss_only",
        "--skip_test_eval",
    ]


def run_with_log(cmd, cwd: Path, log_path: Path):
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("[Cmd] " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            lf.write(line)
            lf.flush()
        return p.wait()


def safe_mean_std(vals):
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def load_tcdp_reference(summary_csv: Path):
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    for r in rows:
        if r.get("exp_name") == "tcdp_joint" and r.get("dataset_pkl_name") == "dataset_table_group_train_val_no_test.pkl":
            return {
                "val_r2_mean": float(r["val_r2_mean"]),
                "val_r2_std": float(r["val_r2_std"]),
                "val_mae_mean": float(r["val_mae_mean"]),
                "val_mae_std": float(r["val_mae_std"]),
                "val_mape_mean": float(r["val_mape_mean"]),
                "val_mape_std": float(r["val_mape_std"]),
                "n_seed": int(r["n_seed"]),
            }
    return None


def main():
    root = project_root()
    py = Path(sys.executable)
    train_py = root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py"

    logs_dir = root / "ICCAD2026_Changxin" / "paper_materials" / "logs_simple_mix_table_group"
    out_dir = root / "ICCAD2026_Changxin" / "paper_materials" / "tables_simple_mix_table_group"
    ensure_dir(logs_dir)
    ensure_dir(out_dir)

    seeds = [9294, 9295, 9296, 9297, 9298]
    per_seed = []

    for s in seeds:
        run_id = f"simple_mix_no_transfer_table_group_s{s}"
        model_dir = f"model_paperv5_{run_id}"
        log_path = logs_dir / f"{run_id}.log"

        parsed = parse_best_from_log(log_path)
        if parsed is None:
            cmd = build_cmd(py, train_py, model_dir, s)
            print(f"[Run] seed={s}")
            rc = run_with_log(cmd, root, log_path)
            if rc != 0:
                raise RuntimeError(f"run failed for seed={s}, rc={rc}")
            parsed = parse_best_from_log(log_path)
            if parsed is None:
                raise RuntimeError(f"[Best] not found in log: {log_path}")
        parsed.seed = s
        per_seed.append(parsed)

    r2_vals = [x.val_r2 for x in per_seed]
    mae_vals = [x.val_mae for x in per_seed]
    mape_vals = [x.val_mape for x in per_seed]

    r2_m, r2_s = safe_mean_std(r2_vals)
    mae_m, mae_s = safe_mean_std(mae_vals)
    mape_m, mape_s = safe_mean_std(mape_vals)

    ref_csv = root / "ICCAD2026_Changxin" / "paper_materials" / "tables_ext_v5_r35_group_ratio" / "exp_summary_dual.csv"
    ref = load_tcdp_reference(ref_csv)

    per_seed_csv = out_dir / "simple_mix_no_transfer_per_seed.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_epoch", "val_r2", "val_loss", "val_mae", "val_mape"])
        for x in per_seed:
            w.writerow([x.seed, x.epoch, f"{x.val_r2:.6f}", f"{x.val_loss:.6f}", f"{x.val_mae:.6f}", f"{x.val_mape:.6f}"])

    compare_csv = out_dir / "simple_mix_vs_tcdp_table_group.csv"
    with compare_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method",
            "n_seed",
            "val_r2_mean",
            "val_r2_std",
            "val_mae_mean",
            "val_mae_std",
            "val_mape_mean",
            "val_mape_std",
            "delta_r2_vs_tcdp",
            "delta_mae_vs_tcdp",
            "delta_mape_vs_tcdp",
        ])
        delta_r2 = delta_mae = delta_mape = math.nan
        if ref is not None:
            delta_r2 = r2_m - ref["val_r2_mean"]
            delta_mae = mae_m - ref["val_mae_mean"]
            delta_mape = mape_m - ref["val_mape_mean"]

        w.writerow([
            "simple_mix_no_transfer",
            len(per_seed),
            f"{r2_m:.6f}",
            f"{r2_s:.6f}",
            f"{mae_m:.6f}",
            f"{mae_s:.6f}",
            f"{mape_m:.6f}",
            f"{mape_s:.6f}",
            f"{delta_r2:.6f}" if not math.isnan(delta_r2) else "",
            f"{delta_mae:.6f}" if not math.isnan(delta_mae) else "",
            f"{delta_mape:.6f}" if not math.isnan(delta_mape) else "",
        ])

        if ref is not None:
            w.writerow([
                "tcdp_joint_reference",
                ref["n_seed"],
                f"{ref['val_r2_mean']:.6f}",
                f"{ref['val_r2_std']:.6f}",
                f"{ref['val_mae_mean']:.6f}",
                f"{ref['val_mae_std']:.6f}",
                f"{ref['val_mape_mean']:.6f}",
                f"{ref['val_mape_std']:.6f}",
                "",
                "",
                "",
            ])

    print("[Done] simple-mix no-transfer results")
    print(f"[Out] {per_seed_csv}")
    print(f"[Out] {compare_csv}")


if __name__ == "__main__":
    main()
