from pathlib import Path

from _runner_utils import run_full_only_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v6_ct_00_anchor_r35"},
        {"name": "v6_ct_01_calib_mlp_128", "extra": ["--use_calib_mlp", "--calib_mlp_hid", "128", "--calib_mlp_dropout", "0.10"]},
        {"name": "v6_ct_02_calib_mlp_256", "extra": ["--use_calib_mlp", "--calib_mlp_hid", "256", "--calib_mlp_dropout", "0.10"]},
        {"name": "v6_ct_03_topo_48x128", "extra": ["--topology_expert_dim", "48", "--topology_expert_hidden", "128"]},
        {"name": "v6_ct_04_topo_64x128", "extra": ["--topology_expert_dim", "64", "--topology_expert_hidden", "128"]},
        {
            "name": "v6_ct_05_calib256_topo64x128",
            "extra": [
                "--use_calib_mlp", "--calib_mlp_hid", "256", "--calib_mlp_dropout", "0.10",
                "--topology_expert_dim", "64", "--topology_expert_hidden", "128",
            ],
        },
    ]
    run_full_only_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v6_calib_topo",
    )


if __name__ == "__main__":
    main()
