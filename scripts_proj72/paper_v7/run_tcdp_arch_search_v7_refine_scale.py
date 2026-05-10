from pathlib import Path

from _runner_utils_v7 import run_v7_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v7r_anchor"},

        # Mild-to-strong adaptive source gating around the best v7 gate zone.
        {"name": "v7r_gate_t070_f030", "extra": ["--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30"]},
        {"name": "v7r_gate_t085_f040", "extra": ["--v7_adaptive_src_gate", "--v7_gate_temp", "0.85", "--v7_gate_floor", "0.40"]},
        {"name": "v7r_gate_t100_f050", "extra": ["--v7_adaptive_src_gate", "--v7_gate_temp", "1.00", "--v7_gate_floor", "0.50"]},

        # Branch-strength rebalancing: keep architecture but rebalance calibration/topology residual.
        {
            "name": "v7r_gate070_cal095_top105",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.95", "--v7_topo_scale", "1.05",
            ],
        },
        {
            "name": "v7r_gate070_cal090_top110",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            ],
        },
        {
            "name": "v7r_gate070_cal085_top115",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.85", "--v7_topo_scale", "1.15",
            ],
        },
        {
            "name": "v7r_gate070_top120",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_topo_scale", "1.20",
            ],
        },
        {
            "name": "v7r_gate070_cal095",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.95",
            ],
        },
        {"name": "v7r_anchor_top110", "extra": ["--v7_topo_scale", "1.10"]},
        {"name": "v7r_anchor_top120", "extra": ["--v7_topo_scale", "1.20"]},

        # Increase topology expert capacity with mild gating.
        {
            "name": "v7r_bigTopo_gate070",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--topology_expert_dim", "48", "--topology_expert_hidden", "96",
            ],
        },
        {
            "name": "v7r_bigTopo_gate085_cal095_top105",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.85", "--v7_gate_floor", "0.40",
                "--topology_expert_dim", "48", "--topology_expert_hidden", "96",
                "--v7_calib_scale", "0.95", "--v7_topo_scale", "1.05",
            ],
        },

        # Keep disentangle weakly regularized to test if it helps after branch rebalance.
        {
            "name": "v7r_dis_gate085_weakorth",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.85", "--v7_gate_floor", "0.40",
                "--v7_disentangle", "--v7_disentangle_hidden", "128", "--v7_disentangle_orth_w", "0.002",
                "--v7_calib_scale", "0.95", "--v7_topo_scale", "1.05",
            ],
        },
    ]

    run_v7_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v7_refine_scale",
        seeds_quick=[9294, 9295, 9296],
        seeds_extra=[9297, 9298],
        top_k_expand=4,
    )


if __name__ == "__main__":
    main()

