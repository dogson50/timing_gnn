r"""Step20: balanced pseudo-label curriculum (quality-coverage tradeoff)."""

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

    # Balanced confidence + weight ramp.
    options.pseudo_keep_ratio_schedule = "0.50,0.65,0.80"
    options.pseudo_loss_weight_schedule = "0.35,0.70,1.00"
    options.round_freeze_schedule = "1,0,0"
    print(
        "[Step20 Defaults] "
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