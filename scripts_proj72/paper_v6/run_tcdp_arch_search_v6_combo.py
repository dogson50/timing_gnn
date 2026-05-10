from pathlib import Path

from _runner_utils import run_full_only_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v6_cb_00_anchor_r35"},
        {
            "name": "v6_cb_01_calib_topo_arcfilm",
            "extra": [
                "--use_calib_mlp", "--calib_mlp_hid", "256", "--calib_mlp_dropout", "0.10",
                "--topology_expert_dim", "64", "--topology_expert_hidden", "128",
                "--use_arc_cond", "--arc_cond_mode", "film", "--arc_vocab_scope", "src_tgt",
            ],
        },
        {
            "name": "v6_cb_02_dual_concat_topo",
            "extra": [
                "--hgat_dual_readout", "--hgat_dual_merge", "concat",
                "--topology_expert_dim", "64", "--topology_expert_hidden", "128",
            ],
        },
        {
            "name": "v6_cb_03_dual_concat_calib",
            "extra": [
                "--hgat_dual_readout", "--hgat_dual_merge", "concat",
                "--use_calib_mlp", "--calib_mlp_hid", "256", "--calib_mlp_dropout", "0.10",
            ],
        },
        {
            "name": "v6_cb_04_split_arcfilm_calib",
            "extra": [
                "--hgat_net_feat_mode", "parasitic_append_split", "--hgat_par_cap_weight", "0.7",
                "--use_arc_cond", "--arc_cond_mode", "film", "--arc_vocab_scope", "src_tgt",
                "--use_calib_mlp", "--calib_mlp_hid", "256", "--calib_mlp_dropout", "0.10",
            ],
        },
        {
            "name": "v6_cb_05_h96l3_combo",
            "extra": [
                "--hgat_hid", "96", "--hgat_layers", "3", "--hgat_heads", "2",
                "--use_calib_mlp", "--calib_mlp_hid", "256", "--calib_mlp_dropout", "0.10",
                "--topology_expert_dim", "64", "--topology_expert_hidden", "128",
                "--use_arc_cond", "--arc_cond_mode", "film", "--arc_vocab_scope", "src_tgt",
            ],
        },
    ]
    run_full_only_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v6_combo",
    )


if __name__ == "__main__":
    main()
