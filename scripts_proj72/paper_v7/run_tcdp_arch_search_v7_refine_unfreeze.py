from pathlib import Path

from _runner_utils_v7 import run_v7_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    core = [
        "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
        "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
        "--z_noise_std", "0.005",
        "--weight_decay", "8e-5",
        "--learning_rate", "8e-4",
        "--num_epoch", "420",
        "--src_loss_anneal_end", "420",
        "--early_stop_patience", "40",
        "--early_stop_min_delta", "3e-4",
        "--plateau_patience", "6",
        "--plateau_threshold", "1e-4",
    ]

    candidates = [
        {"name": "v7u_anchor_lr8e4", "extra": list(core)},

        # Soft unfreeze via auto-high-sup hook.
        {"name": "v7u_unfreeze_e4_lr003", "extra": list(core) + [
            "--auto_high_sup_cutoff", "0.15",
            "--auto_high_sup_unfreeze_hgat",
            "--auto_high_sup_enc_lr_scale", "0.03",
            "--enc_update_interval", "4",
        ]},
        {"name": "v7u_unfreeze_e2_lr004", "extra": list(core) + [
            "--auto_high_sup_cutoff", "0.15",
            "--auto_high_sup_unfreeze_hgat",
            "--auto_high_sup_enc_lr_scale", "0.04",
            "--enc_update_interval", "2",
        ]},
        {"name": "v7u_unfreeze_e2_lr005", "extra": list(core) + [
            "--auto_high_sup_cutoff", "0.15",
            "--auto_high_sup_unfreeze_hgat",
            "--auto_high_sup_enc_lr_scale", "0.05",
            "--enc_update_interval", "2",
        ]},

        # Unfreeze + target group reweight.
        {"name": "v7u_unfreeze_rw_p03", "extra": list(core) + [
            "--auto_high_sup_cutoff", "0.15",
            "--auto_high_sup_unfreeze_hgat",
            "--auto_high_sup_enc_lr_scale", "0.03",
            "--enc_update_interval", "4",
            "--tgt_group_reweight", "--tgt_group_reweight_power", "0.3",
            "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.0",
        ]},
    ]

    run_v7_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v7_refine_unfreeze",
        seeds_quick=[9294, 9295, 9296],
        seeds_extra=[9297, 9298],
        top_k_expand=3,
    )


if __name__ == "__main__":
    main()

