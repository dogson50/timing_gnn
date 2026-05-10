from pathlib import Path

from _runner_utils_v7 import run_v7_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v7_anchor_r35"},

        # 1) Adaptive transfer gating.
        {"name": "v7_gate_t035_f010", "extra": ["--v7_adaptive_src_gate", "--v7_gate_temp", "0.35", "--v7_gate_floor", "0.10"]},
        {"name": "v7_gate_t050_f020", "extra": ["--v7_adaptive_src_gate", "--v7_gate_temp", "0.50", "--v7_gate_floor", "0.20"]},
        {"name": "v7_gate_t070_f030", "extra": ["--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30"]},

        # 2) Disentanglement.
        {"name": "v7_dis_h064_o001", "extra": ["--v7_disentangle", "--v7_disentangle_hidden", "64", "--v7_disentangle_orth_w", "0.01"]},
        {"name": "v7_dis_h128_o001", "extra": ["--v7_disentangle", "--v7_disentangle_hidden", "128", "--v7_disentangle_orth_w", "0.01"]},
        {"name": "v7_dis_h128_o002", "extra": ["--v7_disentangle", "--v7_disentangle_hidden", "128", "--v7_disentangle_orth_w", "0.02"]},

        # 3) Topology residual MoE.
        {"name": "v7_moe_k2_t10", "extra": ["--v7_topo_moe_k", "2", "--v7_topo_moe_temp", "1.0"]},
        {"name": "v7_moe_k4_t10", "extra": ["--v7_topo_moe_k", "4", "--v7_topo_moe_temp", "1.0"]},
        {"name": "v7_moe_k4_t07", "extra": ["--v7_topo_moe_k", "4", "--v7_topo_moe_temp", "0.7"]},

        # Combinations of the three ideas.
        {
            "name": "v7_gate_dis",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.50", "--v7_gate_floor", "0.20",
                "--v7_disentangle", "--v7_disentangle_hidden", "128", "--v7_disentangle_orth_w", "0.01",
            ],
        },
        {
            "name": "v7_dis_moe",
            "extra": [
                "--v7_disentangle", "--v7_disentangle_hidden", "128", "--v7_disentangle_orth_w", "0.01",
                "--v7_topo_moe_k", "4", "--v7_topo_moe_temp", "0.7",
            ],
        },
        {
            "name": "v7_gate_moe",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.50", "--v7_gate_floor", "0.20",
                "--v7_topo_moe_k", "4", "--v7_topo_moe_temp", "0.7",
            ],
        },
        {
            "name": "v7_all_three",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.50", "--v7_gate_floor", "0.20",
                "--v7_disentangle", "--v7_disentangle_hidden", "128", "--v7_disentangle_orth_w", "0.01",
                "--v7_topo_moe_k", "4", "--v7_topo_moe_temp", "0.7",
            ],
        },
    ]

    run_v7_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v7_main",
        seeds_quick=[9294, 9295, 9296],
        seeds_extra=[9297, 9298],
        top_k_expand=3,
    )


if __name__ == "__main__":
    main()
