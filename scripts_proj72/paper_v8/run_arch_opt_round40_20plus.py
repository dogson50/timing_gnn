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
    script: str
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


def fmt_mean_std(vals: list[float]):
    if not vals:
        return "", ""
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def load_reference(project_root: Path):
    ref_csv = (
        project_root
        / "ICCAD2026_Changxin"
        / "paper_materials"
        / "tables_ext_v5_r35_group_ratio"
        / "exp_summary_dual.csv"
    )
    if not ref_csv.exists():
        return None
    with ref_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (
                r.get("exp_name") == "tcdp_joint"
                and r.get("dataset_pkl_name") == "dataset_table_group_train_val_no_test.pkl"
            ):
                return {
                    "val_r2": float(r["val_r2_mean"]),
                    "val_mae": float(r["val_mae_mean"]),
                    "val_mape": float(r["val_mape_mean"]),
                }
    return None


def common_args(project_root: Path, train_py: Path, model_dir: str, seed: int) -> list[str]:
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


def candidates() -> list[Candidate]:
    core = (
        "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
        "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
        "--z_noise_std", "0.005",
    )
    return [
        Candidate("r40_anchor_v7_core", "v7", core, "Current best-v7 style anchor."),
        Candidate("r40_wd8e5", "v7", core + ("--weight_decay", "8e-5"), "Slightly stronger regularization."),
        Candidate("r40_wd1e4", "v7", core + ("--weight_decay", "1e-4"), "Stronger regularization."),
        Candidate("r40_lr8e4", "v7", core + ("--learning_rate", "8e-4"), "Lower LR for stability."),
        Candidate("r40_lr12e4", "v7", core + ("--learning_rate", "1.2e-3"), "Higher LR for escape."),
        Candidate("r40_drop015", "v7", core + ("--mlp_dropout", "0.15"), "Less MLP dropout."),
        Candidate("r40_drop025", "v7", core + ("--mlp_dropout", "0.25"), "More MLP dropout."),
        Candidate("r40_topo115", "v7", core + ("--v7_topo_scale", "1.15"), "More topology residual."),
        Candidate("r40_topo120", "v7", core + ("--v7_topo_scale", "1.20"), "Aggressive topology residual."),
        Candidate("r40_cal085_top115", "v7", core + ("--v7_calib_scale", "0.85", "--v7_topo_scale", "1.15"), "Shift from calibration to topology."),
        Candidate("r40_cal075_top125", "v7", core + ("--v7_calib_scale", "0.75", "--v7_topo_scale", "1.25"), "Strong topology-biased decomposition."),
        Candidate("r40_gate060_f025", "v7", ("--v7_adaptive_src_gate", "--v7_gate_temp", "0.60", "--v7_gate_floor", "0.25", "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10", "--z_noise_std", "0.005"), "Sharper source gate."),
        Candidate("r40_gate080_f035", "v7", ("--v7_adaptive_src_gate", "--v7_gate_temp", "0.80", "--v7_gate_floor", "0.35", "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10", "--z_noise_std", "0.005"), "Smoother source gate."),
        Candidate("r40_gate090_f040", "v7", ("--v7_adaptive_src_gate", "--v7_gate_temp", "0.90", "--v7_gate_floor", "0.40", "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10", "--z_noise_std", "0.005"), "Conservative source gate."),
        Candidate("r40_src070_tgt110", "v7", core + ("--auto_src_w_high", "0.70", "--auto_src_final_scale_high", "0.70", "--auto_tgt_w_high", "1.10"), "Weaker source at high-supervision."),
        Candidate("r40_src055_tgt120", "v7", core + ("--auto_src_w_high", "0.55", "--auto_src_final_scale_high", "0.55", "--auto_tgt_w_high", "1.20"), "More target-biased high-supervision."),
        Candidate("r40_src080_tgt105", "v7", core + ("--auto_src_w_high", "0.80", "--auto_src_final_scale_high", "0.80", "--auto_tgt_w_high", "1.05"), "Mild source retention."),
        Candidate("r40_rw_p02", "v7", core + ("--tgt_group_reweight", "--tgt_group_reweight_power", "0.2", "--tgt_group_reweight_min", "0.5", "--tgt_group_reweight_max", "1.8"), "Mild group reweighting."),
        Candidate("r40_rw_p04", "v7", core + ("--tgt_group_reweight", "--tgt_group_reweight_power", "0.4", "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.2"), "Medium group reweighting."),
        Candidate("r40_rw_p07", "v7", core + ("--tgt_group_reweight", "--tgt_group_reweight_power", "0.7", "--tgt_group_reweight_min", "0.3", "--tgt_group_reweight_max", "3.0"), "Strong group reweighting."),
        Candidate("r40_par_w04", "v7", core + ("--hgat_par_cap_weight", "0.4"), "Less parasitic-cap influence."),
        Candidate("r40_par_w08", "v7", core + ("--hgat_par_cap_weight", "0.8"), "More parasitic-cap influence."),
        Candidate("r40_mode_replace", "v7", core + ("--hgat_net_feat_mode", "parasitic_replace"), "Parasitic replacement mode."),
        Candidate("r40_mode_append_split", "v7", core + ("--hgat_net_feat_mode", "parasitic_append_split"), "Parasitic append split mode."),
        Candidate("r40_dual_concat", "v7", core + ("--hgat_dual_readout", "--hgat_dual_merge", "concat"), "Dual graph readout concat."),
        Candidate("r40_dual_mean", "v7", core + ("--hgat_dual_readout", "--hgat_dual_merge", "mean"), "Dual graph readout mean."),
        Candidate("r40_type_attn", "v7", core + ("--hgat_type_attn_readout",), "Type-attention readout."),
        Candidate("r40_net_readout", "v7", core + ("--hgat_use_net_readout",), "Net-only readout."),
        Candidate("r40_topo_dim64", "v7", core + ("--topology_expert_dim", "64", "--topology_expert_hidden", "64"), "Wider topology embedding."),
        Candidate("r40_topo_hidden128", "v7", core + ("--topology_expert_dim", "32", "--topology_expert_hidden", "128"), "Deeper topology residual."),
        Candidate("r40_moe2", "v7", core + ("--v7_topo_moe_k", "2", "--v7_topo_moe_temp", "0.8"), "Topology residual MoE-2."),
        Candidate("r40_moe4", "v7", core + ("--v7_topo_moe_k", "4", "--v7_topo_moe_temp", "0.8"), "Topology residual MoE-4."),
        Candidate("r40_dis_weak", "v7", core + ("--v7_disentangle", "--v7_disentangle_hidden", "64", "--v7_disentangle_orth_w", "1e-4"), "Weak domain-invariant/domain-specific decoupling."),
        Candidate("r40_dis_mid", "v7", core + ("--v7_disentangle", "--v7_disentangle_hidden", "96", "--v7_disentangle_orth_w", "5e-4"), "Medium decoupling."),
        Candidate("r40_znoise0", "v7", core + ("--z_noise_std", "0.0"), "No graph embedding noise."),
        Candidate("r40_znoise02", "v7", core + ("--z_noise_std", "0.02"), "Stronger graph embedding noise."),
        Candidate("r40_gated_scalar", "gated", ("--use_gated_fusion", "--fusion_gate_mode", "scalar", "--fusion_gate_hidden", "0", "--fusion_gate_dropout", "0.0"), "Scalar graph/table fusion gate."),
        Candidate("r40_gated_ch64_drop05", "gated", ("--use_gated_fusion", "--fusion_gate_mode", "channel", "--fusion_gate_hidden", "64", "--fusion_gate_dropout", "0.05"), "Channel fusion gate with small hidden."),
        Candidate("r40_gated_ch32_drop10", "gated", ("--use_gated_fusion", "--fusion_gate_mode", "channel", "--fusion_gate_hidden", "32", "--fusion_gate_dropout", "0.10"), "More regularized channel fusion gate."),
    ]


def train_script(project_root: Path, script: str) -> Path:
    if script == "v7":
        return project_root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_v7_opt.py"
    if script == "gated":
        return project_root / "src" / "train_balanced_sampling_sep_mlp_shared_calib_step5a_gated_fusion_opt.py"
    raise ValueError(f"unknown script kind: {script}")


def summarize(rows: list[dict], tables_dir: Path, ref: dict | None) -> list[dict]:
    ensure_dir(tables_dir)
    runs_csv = tables_dir / "runs.csv"
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "candidate", "script", "seed", "note",
            "best_epoch", "val_r2", "val_loss", "val_mae", "val_mape",
            "status", "log_path",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    grouped = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        grouped.setdefault(r["candidate"], []).append(r)

    out = []
    for name, rr in grouped.items():
        r2m, r2s = fmt_mean_std([float(x["val_r2"]) for x in rr])
        maem, maes = fmt_mean_std([float(x["val_mae"]) for x in rr])
        mapem, mapes = fmt_mean_std([float(x["val_mape"]) for x in rr])
        item = {
            "candidate": name,
            "script": rr[0]["script"],
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
            "note": rr[0]["note"],
        }
        if ref is not None:
            item["delta_r2_vs_ref"] = r2m - ref["val_r2"]
            item["delta_mae_vs_ref"] = maem - ref["val_mae"]
            item["delta_mape_vs_ref"] = mapem - ref["val_mape"]
        out.append(item)

    out.sort(key=lambda x: (float(x["val_r2_mean"]), -float(x["val_mae_mean"])), reverse=True)
    summary_csv = tables_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "candidate", "script", "n_seed",
            "val_r2_mean", "val_r2_std",
            "val_mae_mean", "val_mae_std",
            "val_mape_mean", "val_mape_std",
            "delta_r2_vs_ref", "delta_mae_vs_ref", "delta_mape_vs_ref",
            "note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    suite = "round40_20plus"
    logs_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / f"logs_arch_search_{suite}"
    tables_dir = project_root / "ICCAD2026_Changxin" / "paper_materials" / f"tables_arch_search_{suite}"
    ensure_dir(logs_dir)
    ensure_dir(tables_dir)
    ref = load_reference(project_root)

    cands = candidates()
    rows: list[dict] = []
    quick_seeds = [9294]
    expand1_seeds = [9295, 9296]
    expand2_seeds = [9297, 9298]

    def run_one(c: Candidate, seed: int) -> dict:
        run_id = f"{c.name}_s{seed}"
        log_path = logs_dir / f"{run_id}.log"
        best = parse_best(log_path)
        if best is None:
            model_dir = f"model_{suite}_{run_id}"
            cmd = common_args(project_root, train_script(project_root, c.script), model_dir, seed) + list(c.extra)
            rc = run_cmd(cmd, project_root, log_path)
            if rc != 0:
                print(f"[Warn] {run_id} failed rc={rc}")
            best = parse_best(log_path)

        row = {
            "candidate": c.name,
            "script": c.script,
            "seed": seed,
            "note": c.note,
            "status": "ok" if best else "missing_best",
            "log_path": str(log_path.relative_to(project_root)),
        }
        if best:
            row.update(best)
        return row

    print(f"[Suite] {suite}; candidates={len(cands)}")

    # Stage 1: run all candidates with one seed, enough to cover 20+ architecture trials.
    for c in cands:
        print(f"[Stage1] {c.name}: {c.note}")
        for seed in quick_seeds:
            rows.append(run_one(c, seed))
        summary = summarize(rows, tables_dir, ref)
        if summary:
            print(f"[Stage1Top] {summary[0]['candidate']} r2={float(summary[0]['val_r2_mean']):.6f}")

    # Stage 2: expand top 8 candidates to 3 seeds.
    summary = summarize(rows, tables_dir, ref)
    top8 = {x["candidate"] for x in summary[:8]}
    print(f"[Stage2Top8] {sorted(top8)}")
    for c in cands:
        if c.name not in top8:
            continue
        for seed in expand1_seeds:
            rows.append(run_one(c, seed))
        summarize(rows, tables_dir, ref)

    # Stage 3: expand top 4 candidates to 5 seeds.
    summary = summarize(rows, tables_dir, ref)
    top4 = {x["candidate"] for x in summary[:4]}
    print(f"[Stage3Top4] {sorted(top4)}")
    for c in cands:
        if c.name not in top4:
            continue
        for seed in expand2_seeds:
            rows.append(run_one(c, seed))
        summarize(rows, tables_dir, ref)

    summary = summarize(rows, tables_dir, ref)
    print("[Done] top candidates:")
    for x in summary[:10]:
        dr = x["delta_r2_vs_ref"]
        dr_s = f"{float(dr):+.6f}" if dr != "" else "NA"
        print(
            f"  {x['candidate']}: n={x['n_seed']}, "
            f"r2={float(x['val_r2_mean']):.6f}, "
            f"mae={float(x['val_mae_mean']):.4f}, "
            f"mape={float(x['val_mape_mean']):.4f}, "
            f"delta_r2={dr_s}"
        )


if __name__ == "__main__":
    main()
