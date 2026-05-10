import csv
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


RE_BEST = re.compile(
    r"^\[Best\]\s*epoch:(?P<best_epoch>\d+),\s*val_r2:(?P<val_r2>[-+0-9.eE]+),\s*val_loss:(?P<val_loss>[-+0-9.eE]+),\s*val_mae:(?P<val_mae>[-+0-9.eE]+),\s*val_mape:(?P<val_mape>[-+0-9.eE]+)"
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_best(log_path: Path):
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_BEST.match(line.strip())
        if m:
            best = {
                "best_epoch": int(m.group("best_epoch")),
                "val_r2": float(m.group("val_r2")),
                "val_loss": float(m.group("val_loss")),
                "val_mae": float(m.group("val_mae")),
                "val_mape": float(m.group("val_mape")),
            }
    return best


def run_cmd(cmd, cwd: Path, log_path: Path):
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("[Cmd] " + " ".join(cmd) + "\n")
        lf.flush()
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
            lf.write(line)
            lf.flush()
        return proc.wait()


def fmt_mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def load_full_baseline(summary_csv: Path):
    if not summary_csv.exists():
        return None
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if (
            r.get("exp_name") == "tcdp_joint"
            and r.get("dataset_pkl_name") == "dataset_table_group_train_val_no_test.pkl"
        ):
            return {
                "r2": float(r["val_r2_mean"]),
                "mae": float(r["val_mae_mean"]),
                "mape": float(r["val_mape_mean"]),
            }
    return None


def base_args(train_py: Path, model_dir: str, seed: int):
    return [
        str(Path(sys.executable)),
        str(train_py),
        "--model_saving_dir", model_dir,
        "--data_save_path", "ICCAD2026_Changxin/paper_materials/data",
        "--dataset_pkl_name", "dataset_table_group_train_val_no_test.pkl",
        "--gpu", "0",
        "--seed", str(seed),
        "--freeze_hgat",
        "--num_epoch", "320",
        "--batch_size", "2048",
        "--learning_rate", "1.0e-3",
        "--enc_lr_scale", "0.05",
        "--weight_decay", "5e-5",
        "--mlp_dropout", "0.2",
        "--hgat_l2_norm",
        "--z_noise_std", "0.01",
        "--enrich_parasitic_net_feat",
        "--hgat_net_feat_mode", "base",
        "--hgat_par_cap_weight", "0.6",
        "--use_topology_expert",
        "--topology_expert_dim", "32",
        "--topology_expert_hidden", "64",
        "--src_loss_anneal_start", "120",
        "--src_loss_anneal_end", "320",
        "--src_loss_final_scale", "0.8",
        "--target_loss_weight", "1.0",
        "--loss_weight_45", "1.0",
        "--lr_scheduler", "plateau",
        "--plateau_factor", "0.5",
        "--plateau_patience", "4",
        "--plateau_threshold", "3e-4",
        "--plateau_min_lr", "1e-6",
        "--early_stop_patience", "22",
        "--early_stop_min_delta", "8e-4",
        "--num_workers", "0",
        "--val_eval_interval", "5",
        "--epoch_log_interval", "10",
        "--skip_train_r2",
        "--fast_eval_loss_only",
        "--skip_test_eval",
        "--auto_transfer_by_sup",
        "--auto_src_w_low", "1.0",
        "--auto_src_w_high", "1.0",
        "--auto_src_final_scale_low", "0.8",
        "--auto_src_final_scale_high", "0.8",
        "--auto_tgt_w_low", "1.0",
        "--auto_tgt_w_high", "1.0",
        "--auto_high_sup_cutoff", "0.99",
        "--auto_high_sup_unfreeze_hgat",
        "--auto_high_sup_enc_lr_scale", "0.03",
        "--enc_update_interval", "4",
    ]


def run_full_only_search(
    *,
    project_root: Path,
    candidate_defs,
    suite_name: str,
    seeds_quick=None,
    seeds_extra=None,
):
    if seeds_quick is None:
        seeds_quick = [9294, 9295, 9296]
    if seeds_extra is None:
        seeds_extra = [9297, 9298]

    train_py = project_root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py"
    logs_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / f"logs_arch_search_{suite_name}"
    tables_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / f"tables_arch_search_{suite_name}"
    ensure_dir(logs_dir)
    ensure_dir(tables_dir)

    baseline = load_full_baseline(
        project_root / "ICCAD2026_Changxin" / "paper_materials" / "tables_ext_v5_r35_group_ratio" / "exp_summary_dual.csv"
    )

    all_rows = []
    for c in candidate_defs:
        name = c["name"]
        extra = list(c.get("extra", []))
        print(f"[Candidate] {name}")
        for seed in seeds_quick:
            run_id = f"{name}_s{seed}_full"
            model_dir = f"model_{suite_name}_{run_id}"
            log_path = logs_dir / f"{run_id}.log"
            best = parse_best(log_path)
            if best is None:
                cmd = base_args(train_py, model_dir, seed) + extra
                rc = run_cmd(cmd, project_root, log_path)
                if rc != 0:
                    print(f"[Warn] non-zero exit for {run_id}: rc={rc}")
                best = parse_best(log_path)
            row = {
                "candidate": name,
                "seed": seed,
                "dataset_pkl_name": "dataset_table_group_train_val_no_test.pkl",
            }
            if best:
                row.update(best)
            all_rows.append(row)

    quick_summary = summarize_and_save(all_rows, tables_dir, baseline, "quick")
    top_names = [x["candidate"] for x in quick_summary[:2]]
    print(f"[ExpandTop2] {top_names}")

    for c in candidate_defs:
        name = c["name"]
        if name not in top_names:
            continue
        extra = list(c.get("extra", []))
        for seed in seeds_extra:
            run_id = f"{name}_s{seed}_full"
            model_dir = f"model_{suite_name}_{run_id}"
            log_path = logs_dir / f"{run_id}.log"
            best = parse_best(log_path)
            if best is None:
                cmd = base_args(train_py, model_dir, seed) + extra
                rc = run_cmd(cmd, project_root, log_path)
                if rc != 0:
                    print(f"[Warn] non-zero exit for {run_id}: rc={rc}")
                best = parse_best(log_path)
            row = {
                "candidate": name,
                "seed": seed,
                "dataset_pkl_name": "dataset_table_group_train_val_no_test.pkl",
            }
            if best:
                row.update(best)
            all_rows.append(row)

    final_summary = summarize_and_save(all_rows, tables_dir, baseline, "final")
    if final_summary:
        top = final_summary[0]
        print(
            f"[Done] best={top['candidate']}, "
            f"val_r2={top['val_r2_mean']:.6f}, "
            f"delta_vs_r35={top.get('delta_r2_vs_r35_tcdp', 0.0):+.6f}"
        )


def summarize_and_save(rows, tables_dir: Path, baseline, stage_tag: str):
    ensure_dir(tables_dir)
    runs_csv = tables_dir / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        fn = ["candidate", "seed", "dataset_pkl_name", "best_epoch", "val_r2", "val_loss", "val_mae", "val_mape"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)

    grp = {}
    for r in rows:
        if "val_r2" not in r:
            continue
        grp.setdefault(r["candidate"], []).append(r)

    out = []
    for cand, rr in grp.items():
        r2m, r2s = fmt_mean_std([float(x["val_r2"]) for x in rr])
        maem, maes = fmt_mean_std([float(x["val_mae"]) for x in rr])
        mapem, mapes = fmt_mean_std([float(x["val_mape"]) for x in rr])
        item = {
            "candidate": cand,
            "n_seed": len(rr),
            "val_r2_mean": r2m,
            "val_r2_std": r2s,
            "val_mae_mean": maem,
            "val_mae_std": maes,
            "val_mape_mean": mapem,
            "val_mape_std": mapes,
            "delta_r2_vs_r35_tcdp": "",
            "delta_mae_vs_r35_tcdp": "",
            "delta_mape_vs_r35_tcdp": "",
        }
        if baseline is not None:
            item["delta_r2_vs_r35_tcdp"] = r2m - baseline["r2"]
            item["delta_mae_vs_r35_tcdp"] = maem - baseline["mae"]
            item["delta_mape_vs_r35_tcdp"] = mapem - baseline["mape"]
        out.append(item)

    out.sort(key=lambda x: x["val_r2_mean"], reverse=True)
    summary_csv = tables_dir / "summary_full.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fn = [
            "candidate", "n_seed",
            "val_r2_mean", "val_r2_std",
            "val_mae_mean", "val_mae_std",
            "val_mape_mean", "val_mape_std",
            "delta_r2_vs_r35_tcdp",
            "delta_mae_vs_r35_tcdp",
            "delta_mape_vs_r35_tcdp",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(out)

    top_csv = tables_dir / f"top_{stage_tag}.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        fn = [
            "candidate", "n_seed",
            "val_r2_mean", "val_r2_std",
            "val_mae_mean", "val_mae_std",
            "val_mape_mean", "val_mape_std",
            "delta_r2_vs_r35_tcdp",
            "delta_mae_vs_r35_tcdp",
            "delta_mape_vs_r35_tcdp",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(out[:10])
    return out
