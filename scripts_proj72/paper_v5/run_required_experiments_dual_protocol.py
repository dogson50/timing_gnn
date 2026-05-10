import csv
import datetime as dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List


@dataclass
class RunItem:
    run_id: str
    protocol: str
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
        "run_id", "protocol", "exp_name", "seed", "dataset_pkl_name",
        "model_dir", "log_path", "status", "exit_code", "start_time", "end_time",
        "duration_sec", "cmd",
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
            lf.flush()
        rc = proc.wait()
        lf.write(f"\n[RunEnd] {now_str()} exit={rc}\n")
    dur = (dt.datetime.now() - start).total_seconds()
    return rc, dur


def log_monitor(project_root: Path, msg: str):
    mon = project_root / "copilot_train_logs" / "arch_queue_monitor.log"
    ensure_dirs(mon.parent)
    with mon.open("a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {msg}\n")


def quoted_cmd(parts: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in parts)


def prepare_celltype_datasets(project_root: Path, py: Path):
    data_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / "data"
    ensure_dirs(data_dir)
    base_pkl = data_dir / "dataset_table_group_train_val_no_test.pkl"
    cell_pkl = data_dir / "dataset_cell_type_train_val_no_test.pkl"

    prep_py = project_root / "scripts_proj72" / "paper_v5" / "prepare_celltype_train_val_no_test_dataset.py"
    sup_py = project_root / "scripts_proj72" / "paper_v5" / "build_supervision_ratio_datasets_generic.py"

    if not cell_pkl.exists():
        subprocess.check_call(
            [
                str(py), str(prep_py),
                "--base_pkl", str(base_pkl),
                "--out_dir", str(data_dir),
                "--out_name", "dataset_cell_type_train_val_no_test.pkl",
                "--split_seed", "42",
                "--train_ratio", str(1.0 / 6.0),
                "--target_label_ratio", "1.0",
            ],
            cwd=str(project_root),
        )

    # Always rebuild ratio subsets to guarantee consistency with base dataset.
    subprocess.check_call(
        [
            str(py), str(sup_py),
            "--base_pkl", str(cell_pkl),
            "--out_dir", str(data_dir),
            "--prefix", "dataset_cell_type_train_val_no_test",
            "--ratios", "0.01", "0.05", "0.10", "0.20",
            "--seed", "9294",
        ],
        cwd=str(project_root),
    )


def build_protocols() -> Dict[str, dict]:
    # Protocol-A: best current TCDP variant direction (r39_01 style).
    tg_common = [
        "--freeze_hgat",
        "--num_epoch", "360",
        "--batch_size", "2048",
        "--learning_rate", "9e-4",
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
        "--src_loss_anneal_start", "140",
        "--src_loss_anneal_end", "360",
        "--src_loss_final_scale", "0.8",
        "--target_loss_weight", "1.0",
        "--loss_weight_45", "1.0",
        "--lr_scheduler", "plateau",
        "--plateau_factor", "0.5",
        "--plateau_patience", "5",
        "--plateau_threshold", "3e-4",
        "--plateau_min_lr", "1e-6",
        "--early_stop_patience", "26",
        "--early_stop_min_delta", "8e-4",
        "--num_workers", "0",
        "--val_eval_interval", "5",
        "--epoch_log_interval", "10",
        "--skip_train_r2",
        "--fast_eval_loss_only",
        "--skip_test_eval",
    ]
    tg_ratio_base = [
        "--auto_transfer_by_sup",
        "--auto_sup_high", "0.20",
        "--auto_sup_curve_power", "0.80",
        "--auto_tgt_w_high", "1.10",
        "--auto_src_w_high", "0.35",
        "--auto_src_final_scale_high", "0.25",
        "--auto_src_w_low", "1.0",
        "--auto_src_final_scale_low", "0.8",
        "--auto_tgt_w_low", "1.0",
    ]

    def tg_ratio_extra(tag: str):
        if tag == "05p":
            return [
                *tg_ratio_base,
                "--auto_high_sup_cutoff", "0.05",
                "--auto_high_sup_src_w", "0.03",
                "--auto_high_sup_src_final_scale", "0.02",
                "--auto_high_sup_tgt_w", "1.45",
                "--tgt_group_reweight",
                "--tgt_group_reweight_power", "0.2",
            ]
        return list(tg_ratio_base)

    # Protocol-B: historical cell_type best-line (legacy hyper-parameters + TCDP heads).
    ct_common = [
        "--freeze_hgat",
        "--num_epoch", "240",
        "--batch_size", "3072",
        "--learning_rate", "1.2e-3",
        "--enc_lr_scale", "0.05",
        "--weight_decay", "5e-5",
        "--mlp_dropout", "0.2",
        "--hgat_l2_norm",
        "--z_noise_std", "0.01",
        "--enrich_parasitic_net_feat",
        "--hgat_net_feat_mode", "parasitic_replace",
        "--hgat_par_cap_weight", "0.7",
        "--use_topology_expert",
        "--topology_expert_dim", "32",
        "--topology_expert_hidden", "64",
        "--src_loss_anneal_start", "25",
        "--src_loss_anneal_end", "180",
        "--src_loss_final_scale", "0.55",
        "--target_loss_weight", "1.0",
        "--loss_weight_45", "1.0",
        "--lr_scheduler", "plateau",
        "--plateau_factor", "0.5",
        "--plateau_patience", "4",
        "--plateau_threshold", "4e-4",
        "--plateau_min_lr", "5e-6",
        "--early_stop_patience", "18",
        "--early_stop_min_delta", "6e-4",
        "--num_workers", "0",
        "--val_eval_interval", "1",
        "--epoch_log_interval", "10",
        "--skip_train_r2",
        "--fast_eval_loss_only",
        "--skip_test_eval",
    ]

    return {
        "tg_bestnow_r39_01": {
            "train_script": "step5a",
            "base_dataset": "dataset_table_group_train_val_no_test.pkl",
            "ratio_prefix": "dataset_table_group_train_val_no_test",
            "common_args": tg_common,
            "ratio_joint_extra_fn": tg_ratio_extra,
        },
        "celltype_legacy_best": {
            "train_script": "step5a",
            "base_dataset": "dataset_cell_type_train_val_no_test.pkl",
            "ratio_prefix": "dataset_cell_type_train_val_no_test",
            "common_args": ct_common,
            "ratio_joint_extra_fn": lambda tag: [],
        },
    }


def build_common_args(model_dir: str, dataset_pkl_name: str, seed: int, protocol_cfg: dict):
    return [
        "--model_saving_dir", model_dir,
        "--data_save_path", "ICCAD2026_Changxin/paper_materials/data",
        "--dataset_pkl_name", dataset_pkl_name,
        "--gpu", "0",
        "--seed", str(seed),
    ] + list(protocol_cfg["common_args"])


def maybe_remove_topology(args_extra: List[str]):
    out = []
    i = 0
    while i < len(args_extra):
        cur = args_extra[i]
        if cur == "--use_topology_expert":
            i += 1
            continue
        out.append(cur)
        i += 1
    return out


def main():
    project_root = Path(__file__).resolve().parents[2]
    py = Path(sys.executable)
    train_step5a = project_root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py"
    parse_py = project_root / "scripts_proj72" / "paper_v3" / "parse_experiment_logs.py"

    logs_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / "logs_ext_v5"
    tables_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / "tables_ext_v5"
    ensure_dirs(logs_dir, tables_dir)

    log_monitor(project_root, "continue: start v5 dual-protocol required experiments (table_group bestnow + cell_type legacy)")

    # 1) Prepare protocol-specific datasets.
    prepare_celltype_datasets(project_root, py)

    protocols = build_protocols()
    ratios = ["01p", "05p", "10p", "20p"]
    seeds = [9294, 9295, 9296, 9297, 9298]

    manifest_path = tables_dir / "exp_manifest_dual.csv"
    runs_csv = tables_dir / "exp_runs_dual.csv"
    summary_csv = tables_dir / "exp_summary_dual.csv"
    manifest_old = read_manifest(manifest_path)

    runs: List[RunItem] = []

    def add_run(
        protocol: str,
        exp_name: str,
        seed: int,
        dataset_pkl: str,
        args_extra: List[str],
        train_script_key: str = "step5a",
    ):
        cfg = protocols[protocol]
        train_py = train_step5a if train_script_key == "step5a" else train_step5a
        run_id = f"{protocol}__{exp_name}_s{seed}_{dataset_pkl.replace('.pkl', '')}"
        model_dir = f"model_paperv5_{run_id}"
        log_path = logs_dir / f"{run_id}.log"
        cmd = [str(py), str(train_py)] + build_common_args(model_dir, dataset_pkl, seed, cfg) + list(args_extra)
        runs.append(
            RunItem(
                run_id=run_id,
                protocol=protocol,
                exp_name=exp_name,
                seed=seed,
                dataset_pkl_name=dataset_pkl,
                model_dir=model_dir,
                log_path=str(log_path),
                cmd=quoted_cmd(cmd),
            )
        )

    # 2) Build run queue for each protocol.
    for protocol, cfg in protocols.items():
        base_ds = cfg["base_dataset"]
        ratio_prefix = cfg["ratio_prefix"]

        for s in seeds:
            # Compare set
            add_run(protocol, "tcdp_joint", s, base_ds, [])
            add_run(protocol, "target_only", s, base_ds, ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0"])
            add_run(protocol, "source_only", s, base_ds, ["--target_loss_weight", "0.0", "--loss_weight_45", "1.0"])

            # Pretrain + finetune
            pre_exp = "pretrain_stage1_source_only"
            pre_ds = base_ds
            pre_model_dir = f"model_paperv5_{protocol}__{pre_exp}_s{s}_{pre_ds.replace('.pkl', '')}"
            pre_ckpt = project_root / pre_model_dir / "ckpt_best.pt"
            add_run(protocol, pre_exp, s, pre_ds, ["--target_loss_weight", "0.0", "--loss_weight_45", "1.0", "--num_epoch", "140"])
            add_run(
                protocol,
                "pretrain_finetune",
                s,
                base_ds,
                ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0", "--init_full_ckpt_path", str(pre_ckpt), "--num_epoch", "180"],
            )

            # Ablation set
            add_run(protocol, "ablate_wo_graph", s, base_ds, ["--disable_graph_feature"])
            add_run(protocol, "ablate_wo_domain_calib", s, base_ds, ["--disable_domain_calibration"])
            add_run(
                protocol,
                "ablate_wo_topology_res",
                s,
                base_ds,
                maybe_remove_topology([]),  # topology removed from common args in execution phase below
            )

            # Ratio curve set
            for tag in ratios:
                ds = f"{ratio_prefix}_sup{tag}.pkl"
                add_run(protocol, f"ratio_tcdp_joint_{tag}", s, ds, cfg["ratio_joint_extra_fn"](tag))
                add_run(
                    protocol,
                    f"ratio_target_only_{tag}",
                    s,
                    ds,
                    ["--loss_weight_45", "0.0", "--src_loss_final_scale", "0.0"],
                )

    # 3) Resume from manifest.
    all_rows = []
    for r in runs:
        row = asdict(r)
        prev = manifest_old.get(r.run_id)
        if prev is not None and prev.get("status") == "done" and is_log_finished(Path(prev.get("log_path", ""))):
            row.update(prev)
        all_rows.append(row)
    write_manifest(manifest_path, all_rows)

    # 4) Execute queue until all done.
    for row in all_rows:
        log_path = Path(row["log_path"])
        if row.get("status") == "done" and is_log_finished(log_path):
            print(f"[SkipDone] {row['run_id']}")
            continue

        cmd = shlex.split(row["cmd"])
        # Special handling for topology ablation: remove topology flag from command.
        if row["exp_name"] == "ablate_wo_topology_res":
            cmd = [x for x in cmd if x != "--use_topology_expert"]

        print(f"[Run] {row['run_id']}")
        log_monitor(project_root, f"v5 running: {row['run_id']}")

        row["status"] = "running"
        row["start_time"] = now_str()
        write_manifest(manifest_path, all_rows)

        rc, dur = run_cmd(cmd, project_root, log_path)
        row["exit_code"] = str(rc)
        row["end_time"] = now_str()
        row["duration_sec"] = f"{dur:.1f}"
        row["status"] = "done" if rc == 0 and is_log_finished(log_path) else "failed"
        write_manifest(manifest_path, all_rows)

        subprocess.call(
            [
                str(py),
                str(parse_py),
                "--manifest_csv",
                str(manifest_path),
                "--runs_csv",
                str(runs_csv),
                "--summary_csv",
                str(summary_csv),
            ],
            cwd=str(project_root),
        )

    log_monitor(project_root, "v5 queue finished: all dual-protocol runs completed")
    print("[Done] all queued runs completed.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

