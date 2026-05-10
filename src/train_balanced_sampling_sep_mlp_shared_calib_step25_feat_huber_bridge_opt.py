r"""Step25: huber-loss wrapper over step5a_feat for bridge experiments.

This wrapper does not change model architecture. It only enforces Huber loss
on top of the existing step5a training pipeline so experiments are fully
revertible.
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


def _apply_step25_defaults(options):
    huber_delta = float(os.getenv("STEP25_HUBER_DELTA", "1.0"))
    setattr(options, "train_loss_type", "huber")
    setattr(options, "huber_delta", huber_delta)
    print(
        "[Step25 Defaults] "
        f"train_loss_type={options.train_loss_type}, "
        f"huber_delta={options.huber_delta}"
    )


if __name__ == "__main__":
    options = get_options()
    _apply_step25_defaults(options)

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

    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(
        copilot_log_f
    ), tee.StderrTee(copilot_log_f):
        run_train_and_report_test(
            train_balanced_sep_mlp_shared_calib,
            options,
            seed,
            script_name=__file__,
        )

