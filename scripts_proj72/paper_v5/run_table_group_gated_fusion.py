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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_best(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(BEST_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return RunResult(
        seed=-1,
        epoch=int(m.group("epoch")),
        val_r2=float(m.group("val_r2")),
        val_loss=float(m.group("val_loss")),
        val_mae=float(m.group("val_mae")),
        val_mape=float(m.group("val_mape")),
    )


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


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
        "--use_topology_expert",
        "--topology_expert_dim",
        "32",
        "--topology_expert_hidden",
        "64",
        "--use_gated_fusion",
        "--fusion_gate_mode",
        "channel",
        "--fusion_gate_hidden",
        "128",
        "--src_loss_anneal_start",
        "120",
        "--src_loss_anneal_end",
        "320",
        "--src_loss_final_scale",
        "0.8",
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
        "--auto_transfer_by_sup",
        "--auto_src_w_low",
        "1.0",
        "--auto_src_w_high",
        "1.0",
        "--auto_src_final_scale_low",
        "0.8",
        "--auto_src_final_scale_high",
        "0.8",
        "--auto_tgt_w_low",
        "1.0",
        "--auto_tgt_w_high",
        "1.0",
        "--auto_high_sup_cutoff",
        "0.99",
        "--auto_high_sup_unfreeze_hgat",
        "--auto_high_sup_enc_lr_scale",
        "0.03",
        "--enc_update_interval",
        "4",
    ]


def run_cmd(cmd, cwd: Path, log_path: Path):
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[Cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()


def load_tcdp_reference(root: Path):
    path = root / "ICCAD2026_Changxin" / "paper_materials" / "tables_ext_v5_r35_group_ratio" / "exp_summary_dual.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("exp_name") == "tcdp_joint" and row.get("dataset_pkl_name") == "dataset_table_group_train_val_no_test.pkl":
                return {
                    "n_seed": int(row["n_seed"]),
                    "val_r2_mean": float(row["val_r2_mean"]),
                    "val_r2_std": float(row["val_r2_std"]),
                    "val_mae_mean": float(row["val_mae_mean"]),
                    "val_mae_std": float(row["val_mae_std"]),
                    "val_mape_mean": float(row["val_mape_mean"]),
                    "val_mape_std": float(row["val_mape_std"]),
                }
    return None


def main():
    root = Path(__file__).resolve().parents[2]
    py = Path(sys.executable)
    train_py = root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_gated_fusion_opt.py"
    logs_dir = root / "ICCAD2026_Changxin" / "paper_materials" / "logs_gated_fusion_table_group"
    tables_dir = root / "ICCAD2026_Changxin" / "paper_materials" / "tables_gated_fusion_table_group"
    ensure_dir(logs_dir)
    ensure_dir(tables_dir)

    results = []
    for seed in [9294, 9295, 9296, 9297, 9298]:
        run_id = f"gated_fusion_table_group_s{seed}"
        log_path = logs_dir / f"{run_id}.log"
        result = parse_best(log_path)
        if result is None:
            print(f"[Run] {run_id}")
            cmd = build_cmd(py, train_py, f"model_paperv5_{run_id}", seed)
            rc = run_cmd(cmd, root, log_path)
            if rc != 0:
                raise RuntimeError(f"{run_id} failed with exit code {rc}")
            result = parse_best(log_path)
            if result is None:
                raise RuntimeError(f"[Best] line not found in {log_path}")
        result.seed = seed
        results.append(result)

    r2_m, r2_s = mean_std([x.val_r2 for x in results])
    mae_m, mae_s = mean_std([x.val_mae for x in results])
    mape_m, mape_s = mean_std([x.val_mape for x in results])
    ref = load_tcdp_reference(root)

    with (tables_dir / "gated_fusion_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "best_epoch", "val_r2", "val_loss", "val_mae", "val_mape"])
        for x in results:
            writer.writerow([x.seed, x.epoch, f"{x.val_r2:.6f}", f"{x.val_loss:.6f}", f"{x.val_mae:.6f}", f"{x.val_mape:.6f}"])

    with (tables_dir / "gated_fusion_vs_tcdp.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
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
        if ref:
            delta_r2 = r2_m - ref["val_r2_mean"]
            delta_mae = mae_m - ref["val_mae_mean"]
            delta_mape = mape_m - ref["val_mape_mean"]
        else:
            delta_r2 = delta_mae = delta_mape = math.nan
        writer.writerow([
            "gated_fusion_tcdp",
            len(results),
            f"{r2_m:.6f}",
            f"{r2_s:.6f}",
            f"{mae_m:.6f}",
            f"{mae_s:.6f}",
            f"{mape_m:.6f}",
            f"{mape_s:.6f}",
            "" if math.isnan(delta_r2) else f"{delta_r2:.6f}",
            "" if math.isnan(delta_mae) else f"{delta_mae:.6f}",
            "" if math.isnan(delta_mape) else f"{delta_mape:.6f}",
        ])
        if ref:
            writer.writerow([
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

    print("[Done] gated fusion table_group run finished")
    print(tables_dir / "gated_fusion_vs_tcdp.csv")


if __name__ == "__main__":
    main()
