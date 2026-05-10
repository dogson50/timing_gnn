from pathlib import Path

from _runner_utils_v7 import run_v7_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    best_core = [
        "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
        "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
        "--z_noise_std", "0.005",
    ]

    candidates = [
        {"name": "v7t_anchor"},
        {"name": "v7t_best_core", "extra": list(best_core)},

        # Target group reweighting around mild inverse-frequency.
        {"name": "v7t_best_core_rw_p03", "extra": list(best_core) + [
            "--tgt_group_reweight", "--tgt_group_reweight_power", "0.3",
            "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.0",
        ]},
        {"name": "v7t_best_core_rw_p05", "extra": list(best_core) + [
            "--tgt_group_reweight", "--tgt_group_reweight_power", "0.5",
            "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.5",
        ]},

        # Lighter source transfer (continuous by supervision ratio).
        {"name": "v7t_best_src_h07", "extra": list(best_core) + [
            "--auto_src_w_high", "0.70",
            "--auto_src_final_scale_high", "0.70",
            "--auto_tgt_w_high", "1.10",
        ]},
        {"name": "v7t_best_src_h06", "extra": list(best_core) + [
            "--auto_src_w_high", "0.60",
            "--auto_src_final_scale_high", "0.60",
            "--auto_tgt_w_high", "1.15",
        ]},
        {"name": "v7t_best_src_h05", "extra": list(best_core) + [
            "--auto_src_w_high", "0.50",
            "--auto_src_final_scale_high", "0.50",
            "--auto_tgt_w_high", "1.20",
        ]},

        # Lighter source + reweight.
        {"name": "v7t_src_h06_rw_p03", "extra": list(best_core) + [
            "--auto_src_w_high", "0.60",
            "--auto_src_final_scale_high", "0.60",
            "--auto_tgt_w_high", "1.15",
            "--tgt_group_reweight", "--tgt_group_reweight_power", "0.3",
            "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.0",
        ]},
        {"name": "v7t_src_h05_rw_p03", "extra": list(best_core) + [
            "--auto_src_w_high", "0.50",
            "--auto_src_final_scale_high", "0.50",
            "--auto_tgt_w_high", "1.20",
            "--tgt_group_reweight", "--tgt_group_reweight_power", "0.3",
            "--tgt_group_reweight_min", "0.4", "--tgt_group_reweight_max", "2.0",
        ]},

        # Gate band refinement near good region.
        {"name": "v7t_gate075_core", "extra": [
            "--v7_adaptive_src_gate", "--v7_gate_temp", "0.75", "--v7_gate_floor", "0.30",
            "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            "--z_noise_std", "0.005",
        ]},
        {"name": "v7t_gate065_core", "extra": [
            "--v7_adaptive_src_gate", "--v7_gate_temp", "0.65", "--v7_gate_floor", "0.25",
            "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            "--z_noise_std", "0.005",
        ]},

        # Small regularization tweak.
        {"name": "v7t_best_wd8e5", "extra": list(best_core) + ["--weight_decay", "8e-5"]},
        {"name": "v7t_best_wd1e4", "extra": list(best_core) + ["--weight_decay", "1e-4"]},
    ]

    run_v7_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v7_refine_transfer",
        seeds_quick=[9294, 9295, 9296],
        seeds_extra=[9297, 9298],
        top_k_expand=4,
    )


if __name__ == "__main__":
    main()

