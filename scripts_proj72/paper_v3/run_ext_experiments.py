import csv
import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class RunItem:
    run_id: str
    exp_name: str
    seed: int
    dataset_pkl_name: str
    model_dir: str
    log_path: str
    status: str = "pending"
    exit_code: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_sec: str = ""
    cmd: str = ""


def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def before_cutoff(cutoff_hour=7):
    return dt.datetime.now().hour < cutoff_hour


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["run_id"]: r for r in rows}


def write_manifest(path: Path, rows: List[dict]):
    fields = [
        "run_id",
        "exp_name",
        "seed",
        "dataset_pkl_name",
        "model_dir",
        "log_path",
        "status",
        "exit_code",
        "start_time",
        "end_time",
        "duration_sec",
        "cmd",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def is_log_finished(log_path: Path):
    if not log_path.exists():
        return False
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return "[Best] epoch:" in txt


def run_cmd(cmd: List[str], cwd: Path, log_path: Path):
    start = dt.datetime.now()
    ensure_dirs(log_path.parent)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[RunStart] {now_str()}\n")
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
        rc = proc.wait()
        lf.write(f"\n[RunEnd] {now_str()} exit={rc}\n")
    dur = (dt.datetime.now() - start).total_seconds()
    return rc, dur


def build_common_args(model_dir: str, dataset_pkl_name: str, seed: int):
    return [
        "--model_saving_dir", model_dir,
        "--data_save_path", "ICCAD2026_Changxin/paper_materials/data",
        "--dataset_pkl_name", dataset_pkl_name,
        "--gpu", "0",
        "--seed", str(seed),
        "--freeze_hgat",
        "--num_epoch", "220",
        "--batch_size", "2048",
        "--learning_rate", "2e-3",
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
        "--src_loss_anneal_start", "80",
        "--src_loss_anneal_end", "220",
        "--src_loss_final_scale", "0.8",
        "--target_loss_weight", "1.0",
        "--loss_weight_45", "1.0",
        "--lr_scheduler", "plateau",
        "--plateau_factor", "0.5",
        "--plateau_patience", "3",
        "--plateau_threshold", "3e-4",
        "--plateau_min_lr", "1e-6",
        "--early_stop_patience", "12",
        "--early_stop_min_delta", "8e-4",
        "--num_workers", "0",
        "--val_eval_interval", "5",
        "--skip_test_eval",
    ]


def main():
    project_root = Path(__file__).resolve().parents[2]
    train_py = project_root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py"
    py = Path(sys.executable)

    logs_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / "logs_ext"
    tables_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / "tables_ext"
    data_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / "data"
    ensure_dirs(logs_dir, tables_dir, data_dir)

    # Build supervision-ratio datasets.
    build_ds_py = project_root / "scripts_proj72" / "paper_v3" / "build_supervision_ratio_datasets.py"
    subprocess.check_call(
        [
            str(py),
            str(build_ds_py),
            "--base_pkl",
            str(data_dir / "dataset_table_group_train_val_no_test.pkl"),
            "--out_dir",
            str(data_dir),
            "--ratios",
            "0.01",
            "0.05",
            "0.10",
            "0.20",
            "--seed",
            "9294",
        ],
        cwd=str(project_root),
    )

    stats_py = project_root / "scripts_proj72" / "paper_v3" / "build_dataset_stats.py"
    subprocess.check_call(
        [
            str(py),
            str(stats_py),
            "--data_dir",
            str(data_dir),
            "--out_csv",
            str(tables_dir / "dataset_stats.csv"),
            "--out_md",
            str(tables_dir / "dataset_stats.md"),
        ],
        cwd=str(project_root),
    )

    manifest_path = tables_dir / "exp_manifest.csv"
    manifest_old = read_manifest(manifest_path)

    runs: List[RunItem] = []
    primary_seeds = [9294, 9295, 9296]
    extra_seeds = [9297, 9298]
    ratios = ["01p", "05p", "10p", "20p"]
    all_seeds = primary_seeds + extra_seeds

    def add_run(exp_name: str, seed: int, dataset_pkl: str, args_extra: List[str]):
        run_id = f"{exp_name}_s{seed}_{dataset_pkl.replace('.pkl', '')}"
        model_dir = f"model_paperv3_{exp_name}_s{seed}_{dataset_pkl.replace('.pkl', '')}"
        log_path = logs_dir / f"{run_id}.log"
        cmd = [str(py), str(train_py)] + build_common_args(model_dir, dataset_pkl, seed) + args_extra
        runs.append(
            RunItem(
                run_id=run_id,
                exp_name=exp_name,
                seed=seed,
                dataset_pkl_name=dataset_pkl,
                model_dir=model_dir,
                log_path=str(log_path),
                cmd=" ".join(cmd),
            )
        )

    # Priority-1: required set (multi-seed).
    for s in primary_seeds:
        add_run("tcdp_joint", s, "dataset_table_group_train_val_no_test.pkl", [])
        add_run("target_only", s, "dataset_table_group_train_val_no_test.pkl", ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0"])
        add_run("source_only", s, "dataset_table_group_train_val_no_test.pkl", ["--target_loss_weight", "0.0", "--loss_weight_45", "1.0"])
        add_run("ablate_wo_graph", s, "dataset_table_group_train_val_no_test.pkl", ["--disable_graph_feature"])
        add_run("ablate_wo_domain_calib", s, "dataset_table_group_train_val_no_test.pkl", ["--disable_domain_calibration"])
        add_run("ablate_wo_topology_res", s, "dataset_table_group_train_val_no_test.pkl", ["--loss_weight_45", "1.0", "--topology_expert_dim", "32", "--topology_expert_hidden", "64"])
        # Remove topology expert by disabling flag effect: no use_topology_expert.
        runs[-1].cmd = runs[-1].cmd.replace("--use_topology_expert ", "")
        for tag in ratios:
            ds = f"dataset_table_group_train_val_no_test_sup{tag}.pkl"
            add_run(f"ratio_tcdp_joint_{tag}", s, ds, [])
            add_run(f"ratio_target_only_{tag}", s, ds, ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0"])

    # Priority-1: pretrain + finetune (source-only then target-only).
    for s in primary_seeds:
        pre_id = f"pretrain_source_only_s{s}"
        pre_model = f"model_paperv3_{pre_id}"
        pre_log = logs_dir / f"{pre_id}.log"
        pre_cmd = [str(py), str(train_py)] + build_common_args(pre_model, "dataset_table_group_train_val_no_test.pkl", s) + [
            "--target_loss_weight", "0.0",
            "--loss_weight_45", "1.0",
            "--num_epoch", "140",
        ]
        runs.append(
            RunItem(
                run_id=pre_id,
                exp_name="pretrain_stage1_source_only",
                seed=s,
                dataset_pkl_name="dataset_table_group_train_val_no_test.pkl",
                model_dir=pre_model,
                log_path=str(pre_log),
                cmd=" ".join(pre_cmd),
            )
        )
        ckpt = project_root / pre_model / "ckpt_best.pt"
        add_run(
            "pretrain_finetune",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0", "--init_full_ckpt_path", str(ckpt), "--num_epoch", "180"],
        )

    # Priority-2: extra seeds for full set if time allows.
    for s in extra_seeds:
        add_run("tcdp_joint", s, "dataset_table_group_train_val_no_test.pkl", [])
        add_run("target_only", s, "dataset_table_group_train_val_no_test.pkl", ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0"])
        add_run("source_only", s, "dataset_table_group_train_val_no_test.pkl", ["--target_loss_weight", "0.0", "--loss_weight_45", "1.0"])
        add_run("ablate_wo_graph", s, "dataset_table_group_train_val_no_test.pkl", ["--disable_graph_feature"])
        add_run("ablate_wo_domain_calib", s, "dataset_table_group_train_val_no_test.pkl", ["--disable_domain_calibration"])
        add_run("ablate_wo_topology_res", s, "dataset_table_group_train_val_no_test.pkl", [])
        runs[-1].cmd = runs[-1].cmd.replace("--use_topology_expert ", "")
        for tag in ratios:
            ds = f"dataset_table_group_train_val_no_test_sup{tag}.pkl"
            add_run(f"ratio_tcdp_joint_{tag}", s, ds, [])
            add_run(f"ratio_target_only_{tag}", s, ds, ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0"])

    # Priority-2: pretrain + finetune for extra seeds.
    for s in extra_seeds:
        pre_id = f"pretrain_source_only_s{s}"
        pre_model = f"model_paperv3_{pre_id}"
        pre_log = logs_dir / f"{pre_id}.log"
        pre_cmd = [str(py), str(train_py)] + build_common_args(pre_model, "dataset_table_group_train_val_no_test.pkl", s) + [
            "--target_loss_weight", "0.0",
            "--loss_weight_45", "1.0",
            "--num_epoch", "140",
        ]
        runs.append(
            RunItem(
                run_id=pre_id,
                exp_name="pretrain_stage1_source_only",
                seed=s,
                dataset_pkl_name="dataset_table_group_train_val_no_test.pkl",
                model_dir=pre_model,
                log_path=str(pre_log),
                cmd=" ".join(pre_cmd),
            )
        )
        ckpt = project_root / pre_model / "ckpt_best.pt"
        add_run(
            "pretrain_finetune",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0", "--init_full_ckpt_path", str(ckpt), "--num_epoch", "180"],
        )

    # Priority-3: bonus variants for stronger paper appendices (5 seeds).
    for s in all_seeds:
        add_run(
            "bonus_parasitic_replace",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--hgat_net_feat_mode", "parasitic_replace"],
        )
        add_run(
            "bonus_dualmean",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--hgat_dual_readout", "--hgat_dual_merge", "mean"],
        )
        add_run(
            "bonus_enrich_off_auto",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--hgat_net_feat_mode", "auto"],
        )
        runs[-1].cmd = (
            runs[-1].cmd.replace("--enrich_parasitic_net_feat ", "")
            .replace("--hgat_net_feat_mode base", "--hgat_net_feat_mode auto")
        )
        add_run(
            "bonus_t16x96_mainline",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--topology_expert_dim", "16", "--topology_expert_hidden", "96"],
        )
        add_run(
            "bonus_shared_backbone_only",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--disable_domain_calibration"],
        )
        runs[-1].cmd = runs[-1].cmd.replace("--use_topology_expert ", "")
        add_run(
            "bonus_joint_no_src_anneal",
            s,
            "dataset_table_group_train_val_no_test.pkl",
            ["--src_loss_anneal_start", "9999", "--src_loss_anneal_end", "10000", "--src_loss_final_scale", "1.0"],
        )

    all_rows = []
    for r in runs:
        row = asdict(r)
        prev = manifest_old.get(r.run_id)
        if prev is not None and prev.get("status") == "done" and is_log_finished(Path(prev.get("log_path", ""))):
            row.update(prev)
        all_rows.append(row)
    write_manifest(manifest_path, all_rows)

    # Execute runs until cutoff.
    for row in all_rows:
        if row.get("status") == "done" and is_log_finished(Path(row["log_path"])):
            print(f"[SkipDone] {row['run_id']}")
            continue
        if not before_cutoff(7):
            print("[Stop] cutoff hour reached (>=07:00).")
            break

        cmd = row["cmd"].split(" ")
        log_path = Path(row["log_path"])
        print(f"[Run] {row['run_id']}")
        row["status"] = "running"
        row["start_time"] = now_str()
        write_manifest(manifest_path, all_rows)

        rc, dur = run_cmd(cmd, project_root, log_path)
        row["exit_code"] = str(rc)
        row["end_time"] = now_str()
        row["duration_sec"] = f"{dur:.1f}"
        row["status"] = "done" if rc == 0 and is_log_finished(log_path) else "failed"
        write_manifest(manifest_path, all_rows)

        parse_py = project_root / "scripts_proj72" / "paper_v3" / "parse_experiment_logs.py"
        subprocess.call(
            [
                str(py),
                str(parse_py),
                "--manifest_csv",
                str(manifest_path),
                "--runs_csv",
                str(tables_dir / "exp_runs.csv"),
                "--summary_csv",
                str(tables_dir / "exp_summary.csv"),
            ],
            cwd=str(project_root),
        )

    print("[Done] runner finished or reached cutoff.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
