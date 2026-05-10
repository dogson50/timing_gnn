from pathlib import Path

from _runner_utils_v7 import run_v7_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    core = [
        "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
        "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
        "--z_noise_std", "0.005",
        "--weight_decay", "8e-5",
    ]

    candidates = [
        {"name": "v7l_best_core", "extra": list(core)},
        {
            "name": "v7l_e420_pat40",
            "extra": list(core) + [
                "--num_epoch", "420",
                "--src_loss_anneal_end", "420",
                "--early_stop_patience", "40",
                "--early_stop_min_delta", "3e-4",
                "--plateau_patience", "6",
                "--plateau_threshold", "1e-4",
            ],
        },
        {
            "name": "v7l_e500_pat50",
            "extra": list(core) + [
                "--num_epoch", "500",
                "--src_loss_anneal_end", "500",
                "--early_stop_patience", "50",
                "--early_stop_min_delta", "3e-4",
                "--plateau_patience", "7",
                "--plateau_threshold", "1e-4",
            ],
        },
        {
            "name": "v7l_e420_lr8e4",
            "extra": list(core) + [
                "--learning_rate", "8e-4",
                "--num_epoch", "420",
                "--src_loss_anneal_end", "420",
                "--early_stop_patience", "40",
                "--early_stop_min_delta", "3e-4",
                "--plateau_patience", "6",
                "--plateau_threshold", "1e-4",
            ],
        },
        {
            "name": "v7l_e420_gate075",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.75", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
                "--z_noise_std", "0.005",
                "--weight_decay", "8e-5",
                "--num_epoch", "420",
                "--src_loss_anneal_end", "420",
                "--early_stop_patience", "40",
                "--early_stop_min_delta", "3e-4",
                "--plateau_patience", "6",
                "--plateau_threshold", "1e-4",
            ],
        },
        {
            "name": "v7l_e420_rw_p03",
            "extra": list(core) + [
                "--num_epoch", "420",
                "--src_loss_anneal_end", "420",
                "--early_stop_patience", "40",
                "--early_stop_min_delta", "3e-4",
                "--plateau_patience", "6",
                "--plateau_threshold", "1e-4",
                "--tgt_group_reweight", "--tgt_group_reweight_power", "0.3",
                "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.0",
            ],
        },
    ]

    run_v7_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v7_refine_longtrain",
        seeds_quick=[9294, 9295, 9296],
        seeds_extra=[9297, 9298],
        top_k_expand=3,
    )


if __name__ == "__main__":
    main()

