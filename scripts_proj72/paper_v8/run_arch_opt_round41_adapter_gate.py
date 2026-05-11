import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


BEST_RE = re.compile(
    r"^\[Best\]\s*epoch:(?P<best_epoch>\d+),\s*"
    r"val_r2:(?P<val_r2>[-+0-9.eE]+),\s*"
    r"val_loss:(?P<val_loss>[-+0-9.eE]+),\s*"
    r"val_mae:(?P<val_mae>[-+0-9.eE]+),\s*"
    r"val_mape:(?P<val_mape>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    extra: tuple[str, ...]
    note: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_best(log_path: Path):
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = BEST_RE.match(line.strip())
        if m:
            best = {
                "best_epoch": int(m.group("best_epoch")),
                "val_r2": float(m.group("val_r2")),
                "val_loss": float(m.group("val_loss")),
                "val_mae": float(m.group("val_mae")),
                "val_mape": float(m.group("val_mape")),
            }
    return best


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
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


def fmt(vals: list[float]):
    if not vals:
        return "", ""
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def load_ref(project_root: Path):
    # Prefer the current best round40 result, then fall back to paper TCDP.
    r40 = project_root / "ICCAD2026_Changxin/paper_materials/tables_arch_search_round40_20plus/summary.csv"
    if r40.exists():
        with r40.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r.get("candidate") == "r40_drop025" and int(r.get("n_seed", "0")) >= 5:
                return {
                    "name": "r40_drop025",
                    "val_r2": float(r["val_r2_mean"]),
                    "val_mae": float(r["val_mae_mean"]),
                    "val_mape": float(r["val_mape_mean"]),
                }
    base = project_root / "ICCAD2026_Changxin/paper_materials/tables_ext_v5_r35_group_ratio/exp_summary_dual.csv"
    if base.exists():
        with base.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("exp_name") == "tcdp_joint" and r.get("dataset_pkl_name") == "dataset_table_group_train_val_no_test.pkl":
                    return {
                        "name": "r35_tcdp",
                        "val_r2": float(r["val_r2_mean"]),
                        "val_mae": float(r["val_mae_mean"]),
                        "val_mape": float(r["val_mape_mean"]),
                    }
    return None


def base_args(project_root: Path, model_dir: str, seed: int) -> list[str]:
    train_py = project_root / "src/train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_v8_opt.py"
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
        "--mlp_dropout", "0.25",
        "--hgat_l2_norm",
        "--z_noise_std", "0.005",
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
        "--v7_adaptive_src_gate",
        "--v7_gate_temp", "0.70",
        "--v7_gate_floor", "0.30",
        "--v7_calib_scale", "0.90",
        "--v7_topo_scale", "1.10",
    ]


def candidates() -> list[Candidate]:
    return [
        Candidate("r41_anchor_drop025", (), "round40 best baseline re-run in v8 script"),
        Candidate("r41_ga16_s025", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "16", "--v8_graph_adapter_scale", "0.25"), "small graph adapter"),
        Candidate("r41_ga16_s050", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "16", "--v8_graph_adapter_scale", "0.50"), "small graph adapter stronger"),
        Candidate("r41_ga32_s025", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.25"), "medium graph adapter"),
        Candidate("r41_ga32_s050", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50"), "medium graph adapter stronger"),
        Candidate("r41_ga32_s100", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "1.00"), "full graph adapter"),
        Candidate("r41_ga64_s025", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "64", "--v8_graph_adapter_scale", "0.25"), "wide graph adapter mild"),
        Candidate("r41_ga64_s050", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "64", "--v8_graph_adapter_scale", "0.50"), "wide graph adapter"),
        Candidate("r41_ga32_do05", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50", "--v8_graph_adapter_dropout", "0.05"), "regularized graph adapter"),
        Candidate("r41_ga32_do10", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50", "--v8_graph_adapter_dropout", "0.10"), "stronger adapter dropout"),
        Candidate("r41_tgate_linear", ("--v8_topo_fusion_gate", "--v8_topo_fusion_gate_prior", "2.0"), "topology-conditioned scalar graph gate"),
        Candidate("r41_tgate_h16", ("--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "2.0"), "nonlinear topology graph gate"),
        Candidate("r41_tgate_h32", ("--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "32", "--v8_topo_fusion_gate_prior", "2.0"), "wider topology graph gate"),
        Candidate("r41_tgate_prior15", ("--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "1.5"), "conservative graph gate"),
        Candidate("r41_tgate_prior25", ("--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "2.5"), "graph-amplifying gate"),
        Candidate("r41_cellres_d4_s03", ("--v8_cell_variant_residual", "--v8_cell_variant_dim", "4", "--v8_cell_variant_hidden", "16", "--v8_cell_variant_scale", "0.30"), "drive/cell variant residual mild"),
        Candidate("r41_cellres_d8_s05", ("--v8_cell_variant_residual", "--v8_cell_variant_dim", "8", "--v8_cell_variant_hidden", "32", "--v8_cell_variant_scale", "0.50"), "drive/cell variant residual"),
        Candidate("r41_cellres_d8_s10", ("--v8_cell_variant_residual", "--v8_cell_variant_dim", "8", "--v8_cell_variant_hidden", "32", "--v8_cell_variant_scale", "1.00"), "strong cell variant residual"),
        Candidate("r41_cellres_d16_s05", ("--v8_cell_variant_residual", "--v8_cell_variant_dim", "16", "--v8_cell_variant_hidden", "64", "--v8_cell_variant_scale", "0.50"), "wide cell variant residual"),
        Candidate("r41_ga32_tgate", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50", "--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "2.0"), "adapter plus topology gate"),
        Candidate("r41_ga32_cellres", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50", "--v8_cell_variant_residual", "--v8_cell_variant_dim", "8", "--v8_cell_variant_hidden", "32", "--v8_cell_variant_scale", "0.50"), "adapter plus cell residual"),
        Candidate("r41_tgate_cellres", ("--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "2.0", "--v8_cell_variant_residual", "--v8_cell_variant_dim", "8", "--v8_cell_variant_hidden", "32", "--v8_cell_variant_scale", "0.50"), "topology gate plus cell residual"),
        Candidate("r41_all_mild", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.25", "--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "1.5", "--v8_cell_variant_residual", "--v8_cell_variant_dim", "4", "--v8_cell_variant_hidden", "16", "--v8_cell_variant_scale", "0.30"), "all new modules, conservative"),
        Candidate("r41_all_mid", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50", "--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_prior", "2.0", "--v8_cell_variant_residual", "--v8_cell_variant_dim", "8", "--v8_cell_variant_hidden", "32", "--v8_cell_variant_scale", "0.50"), "all new modules, medium"),
        Candidate("r41_all_reg", ("--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.50", "--v8_graph_adapter_dropout", "0.10", "--v8_topo_fusion_gate", "--v8_topo_fusion_gate_hidden", "16", "--v8_topo_fusion_gate_dropout", "0.10", "--v8_topo_fusion_gate_prior", "2.0", "--v8_cell_variant_residual", "--v8_cell_variant_dim", "8", "--v8_cell_variant_hidden", "32", "--v8_cell_variant_scale", "0.30"), "all modules with regularization"),
        Candidate("r41_drop028_ga32", ("--mlp_dropout", "0.28", "--v8_graph_adapter", "--v8_graph_adapter_hidden", "32", "--v8_graph_adapter_scale", "0.25"), "dropout refinement plus adapter"),
        Candidate("r41_drop030_anchor", ("--mlp_dropout", "0.30"), "dropout refinement"),
    ]


def summarize(rows: list[dict], tables_dir: Path, ref: dict | None) -> list[dict]:
    ensure_dir(tables_dir)
    with (tables_dir / "runs.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["candidate", "seed", "note", "best_epoch", "val_r2", "val_loss", "val_mae", "val_mape", "status", "log_path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    grouped = {}
    for r in rows:
        if r.get("status") == "ok":
            grouped.setdefault(r["candidate"], []).append(r)
    out = []
    for name, rr in grouped.items():
        r2m, r2s = fmt([float(x["val_r2"]) for x in rr])
        maem, maes = fmt([float(x["val_mae"]) for x in rr])
        mapem, mapes = fmt([float(x["val_mape"]) for x in rr])
        item = {
            "candidate": name,
            "n_seed": len(rr),
            "val_r2_mean": r2m,
            "val_r2_std": r2s,
            "val_mae_mean": maem,
            "val_mae_std": maes,
            "val_mape_mean": mapem,
            "val_mape_std": mapes,
            "delta_r2_vs_ref": "",
            "delta_mae_vs_ref": "",
            "delta_mape_vs_ref": "",
            "ref_name": ref["name"] if ref else "",
            "note": rr[0]["note"],
        }
        if ref:
            item["delta_r2_vs_ref"] = r2m - ref["val_r2"]
            item["delta_mae_vs_ref"] = maem - ref["val_mae"]
            item["delta_mape_vs_ref"] = mapem - ref["val_mape"]
        out.append(item)
    out.sort(key=lambda x: (float(x["val_r2_mean"]), -float(x["val_mae_mean"])), reverse=True)
    with (tables_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "candidate", "n_seed", "val_r2_mean", "val_r2_std", "val_mae_mean", "val_mae_std",
            "val_mape_mean", "val_mape_std", "delta_r2_vs_ref", "delta_mae_vs_ref",
            "delta_mape_vs_ref", "ref_name", "note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    suite = "round41_adapter_gate"
    logs_dir = project_root / "ICCAD2026_Changxin/paper_materials" / f"logs_arch_search_{suite}"
    tables_dir = project_root / "ICCAD2026_Changxin/paper_materials" / f"tables_arch_search_{suite}"
    ensure_dir(logs_dir)
    ensure_dir(tables_dir)
    ref = load_ref(project_root)
    cands = candidates()
    rows: list[dict] = []

    def run_one(c: Candidate, seed: int) -> dict:
        run_id = f"{c.name}_s{seed}"
        log_path = logs_dir / f"{run_id}.log"
        best = parse_best(log_path)
        if best is None:
            cmd = base_args(project_root, f"model_{suite}_{run_id}", seed) + list(c.extra)
            rc = run_cmd(cmd, project_root, log_path)
            if rc != 0:
                print(f"[Warn] {run_id} failed rc={rc}")
            best = parse_best(log_path)
        row = {
            "candidate": c.name,
            "seed": seed,
            "note": c.note,
            "status": "ok" if best else "missing_best",
            "log_path": str(log_path.relative_to(project_root)),
        }
        if best:
            row.update(best)
        return row

    print(f"[Suite] {suite}; candidates={len(cands)}; ref={ref}")
    for c in cands:
        print(f"[Stage1] {c.name}: {c.note}")
        rows.append(run_one(c, 9294))
        top = summarize(rows, tables_dir, ref)[:3]
        print("[Top3]", [(x["candidate"], x["val_r2_mean"]) for x in top])

    summary = summarize(rows, tables_dir, ref)
    top8 = {x["candidate"] for x in summary[:8]}
    print("[Stage2Top8]", sorted(top8))
    for c in cands:
        if c.name not in top8:
            continue
        for seed in [9295, 9296]:
            rows.append(run_one(c, seed))
        summarize(rows, tables_dir, ref)

    summary = summarize(rows, tables_dir, ref)
    top4 = {x["candidate"] for x in summary[:4]}
    print("[Stage3Top4]", sorted(top4))
    for c in cands:
        if c.name not in top4:
            continue
        for seed in [9297, 9298]:
            rows.append(run_one(c, seed))
        summarize(rows, tables_dir, ref)

    summary = summarize(rows, tables_dir, ref)
    print("[Done]")
    for x in summary[:10]:
        print(
            f"{x['candidate']} n={x['n_seed']} r2={float(x['val_r2_mean']):.6f} "
            f"mae={float(x['val_mae_mean']):.4f} mape={float(x['val_mape_mean']):.4f} "
            f"dr={x['delta_r2_vs_ref']}"
        )


if __name__ == "__main__":
    main()
