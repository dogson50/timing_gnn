from pathlib import Path

from _runner_utils import run_full_only_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v6_gr_00_anchor_r35"},
        {"name": "v6_gr_01_dual_mean", "extra": ["--hgat_dual_readout", "--hgat_dual_merge", "mean"]},
        {"name": "v6_gr_02_dual_concat", "extra": ["--hgat_dual_readout", "--hgat_dual_merge", "concat"]},
        {"name": "v6_gr_03_arc_film", "extra": ["--use_arc_cond", "--arc_cond_mode", "film", "--arc_vocab_scope", "src_tgt"]},
        {"name": "v6_gr_04_append_split", "extra": ["--hgat_net_feat_mode", "parasitic_append_split", "--hgat_par_cap_weight", "0.7"]},
        {
            "name": "v6_gr_05_concat_arcfilm_split",
            "extra": [
                "--hgat_dual_readout", "--hgat_dual_merge", "concat",
                "--use_arc_cond", "--arc_cond_mode", "film", "--arc_vocab_scope", "src_tgt",
                "--hgat_net_feat_mode", "parasitic_append_split", "--hgat_par_cap_weight", "0.7",
            ],
        },
    ]
    run_full_only_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v6_graph_readout",
    )


if __name__ == "__main__":
    main()
