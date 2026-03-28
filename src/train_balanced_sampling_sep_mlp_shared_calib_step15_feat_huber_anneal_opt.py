r"""Step15 (revised): step5a_feat backbone + conservative source-loss anneal.

This script is intentionally close to step5a_feat baseline and only enables a
gentle source-loss annealing schedule. It keeps MSE/no-scheduler by default.
"""

import os
import random
import numpy as np
import torch as th

from options import get_options
import tee
from test_r2_report import run_train_and_report_test
from train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt import (
    train_balanced_sep_mlp_shared_calib,
)


def _set_if_default(options, name, expected_default, new_value):
    if getattr(options, name) == expected_default:
        setattr(options, name, new_value)


def _set_if_missing(options, name, value):
    if not hasattr(options, name):
        setattr(options, name, value)


def _apply_step15_defaults(options):
    # Keep very close to baseline; only anneal source weight gently.
    _set_if_default(options, "src_loss_anneal_start", -1, 60)
    _set_if_default(options, "src_loss_anneal_end", -1, 280)
    _set_if_default(options, "src_loss_final_scale", 1.0, 0.50)

    _set_if_missing(options, "train_loss_type", "mse")
    _set_if_missing(options, "lr_scheduler", "none")
    _set_if_missing(options, "huber_delta", 1.0)
    _set_if_missing(options, "cosine_t0", max(1, int(options.num_epoch) // 4))
    _set_if_missing(options, "cosine_t_mult", 2)
    _set_if_missing(options, "cosine_eta_min", 1e-6)

    print(
        "[Step15 Revised Defaults] "
        f"src_anneal=({options.src_loss_anneal_start},{options.src_loss_anneal_end},{options.src_loss_final_scale}), "
        f"loss={options.train_loss_type}, scheduler={options.lr_scheduler}"
    )


if __name__ == "__main__":
    options = get_options()
    _apply_step15_defaults(options)

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
        run_train_and_report_test(train_balanced_sep_mlp_shared_calib, options, seed, script_name=__file__)