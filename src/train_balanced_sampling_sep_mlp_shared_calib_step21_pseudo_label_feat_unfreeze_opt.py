r"""Step21: full-unfreeze pseudo-label curriculum (aggressive adaptation)."""

import os
import random
import numpy as np
import torch as th

from options import get_options
import tee
from test_r2_report import run_train_and_report_test
from train_balanced_sampling_sep_mlp_shared_calib_step18_pseudo_label_feat_schedule_opt import train_step18


def _apply_defaults(options):
    options.freeze_hgat = False
    options.pseudo_num_rounds = int(getattr(options, "pseudo_num_rounds", 3))
    options.pseudo_min_keep = max(12, int(getattr(options, "pseudo_min_keep", 16)))

    # Full unfreeze all rounds + moderate confidence filter.
    options.pseudo_keep_ratio_schedule = "0.60,0.75,0.90"
    options.pseudo_loss_weight_schedule = "0.40,0.80,1.10"
    options.round_freeze_schedule = "0,0,0"
    print(
        "[Step21 Defaults] "
        f"keep_ratio_schedule={options.pseudo_keep_ratio_schedule}, "
        f"pseudo_loss_weight_schedule={options.pseudo_loss_weight_schedule}, "
        f"round_freeze_schedule={options.round_freeze_schedule}"
    )


if __name__ == "__main__":
    options = get_options()
    _apply_defaults(options)

    seed = options.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    os.makedirs(options.model_saving_dir, exist_ok=True)
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    os.makedirs(copilot_log_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")
    stdout_f = os.path.join(options.model_saving_dir, "stdout.log")
    stderr_f = os.path.join(options.model_saving_dir, "stderr.log")
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        run_train_and_report_test(train_step18, options, seed, script_name=__file__)