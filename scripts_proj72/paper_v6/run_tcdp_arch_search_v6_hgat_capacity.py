from pathlib import Path

from _runner_utils import run_full_only_search


def main():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        {"name": "v6_hc_00_anchor_r35"},
        {"name": "v6_hc_01_h96_l3_h2", "extra": ["--hgat_hid", "96", "--hgat_layers", "3", "--hgat_heads", "2"]},
        {"name": "v6_hc_02_h128_l3_h2", "extra": ["--hgat_hid", "128", "--hgat_layers", "3", "--hgat_heads", "2"]},
        {"name": "v6_hc_03_h96_l4_h2", "extra": ["--hgat_hid", "96", "--hgat_layers", "4", "--hgat_heads", "2"]},
        {"name": "v6_hc_04_h64_l3_h2", "extra": ["--hgat_hid", "64", "--hgat_layers", "3", "--hgat_heads", "2"]},
        {
            "name": "v6_hc_05_h96l3_arcfilm",
            "extra": ["--hgat_hid", "96", "--hgat_layers", "3", "--hgat_heads", "2", "--use_arc_cond", "--arc_cond_mode", "film", "--arc_vocab_scope", "src_tgt"],
        },
    ]
    run_full_only_search(
        project_root=project_root,
        candidate_defs=candidates,
        suite_name="v6_hgat_capacity",
    )


if __name__ == "__main__":
    main()
