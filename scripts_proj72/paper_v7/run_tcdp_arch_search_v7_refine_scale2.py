from pathlib import Path

from _runner_utils_v7 import run_v7_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v7r2_anchor"},

        # Best point from previous round.
        {
            "name": "v7r2_best_prev",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            ],
        },

        # Around branch-balance point.
        {
            "name": "v7r2_cal088_top110",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.88", "--v7_topo_scale", "1.10",
            ],
        },
        {
            "name": "v7r2_cal092_top110",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.92", "--v7_topo_scale", "1.10",
            ],
        },
        {
            "name": "v7r2_cal090_top108",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.08",
            ],
        },
        {
            "name": "v7r2_cal090_top112",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.12",
            ],
        },

        # Around gate settings.
        {
            "name": "v7r2_gate065_f025",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.65", "--v7_gate_floor", "0.25",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            ],
        },
        {
            "name": "v7r2_gate075_f030",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.75", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            ],
        },
        {
            "name": "v7r2_gate080_f035",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.80", "--v7_gate_floor", "0.35",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
            ],
        },

        # Co-tune source-loss anneal endpoint.
        {
            "name": "v7r2_best_src075",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
                "--src_loss_final_scale", "0.75",
            ],
        },
        {
            "name": "v7r2_best_src085",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
                "--src_loss_final_scale", "0.85",
            ],
        },

        # Slightly larger topology residual capacity, still regularized by branch scales.
        {
            "name": "v7r2_best_topo80",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
                "--topology_expert_hidden", "80",
            ],
        },

        # Noise sensitivity check near best point.
        {
            "name": "v7r2_best_noise005",
            "extra": [
                "--v7_adaptive_src_gate", "--v7_gate_temp", "0.70", "--v7_gate_floor", "0.30",
                "--v7_calib_scale", "0.90", "--v7_topo_scale", "1.10",
                "--z_noise_std", "0.005",
            ],
        },
    ]

    run_v7_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v7_refine_scale2",
        seeds_quick=[9294, 9295, 9296],
        seeds_extra=[9297, 9298],
        top_k_expand=4,
    )


if __name__ == "__main__":
    main()

