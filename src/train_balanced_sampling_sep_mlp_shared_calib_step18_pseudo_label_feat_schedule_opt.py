r"""Step18: Step5a_feat + pseudo-label with round-wise schedules.

This script extends Step14 by adding round-wise schedules for:
1) pseudo_keep_ratio
2) pseudo_loss_weight
3) freeze/unfreeze HGAT per round

It is designed for targeted experiments after observing gains from unfreezing.
"""

import os
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import R2Score
from torch.utils.data import DataLoader, ConcatDataset

from options import get_options
import tee
from test_r2_report import run_train_and_report_test

from hgat import HGATDesignEncoder
from train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt import (
    NUMERIC_COLS,
    GRAPH_NET_DIM,
    GRAPH_MOS_DIM,
    ENCODER_LR_SCALE_DEFAULT,
    CellDelayDataset,
    load_dataset_pkl,
    build_src_graph_cache,
    build_tgt_graph_cache,
    build_z_batch,
    precompute_z_from_graph_cache,
    report_graph_cache_coverage,
    compute_src_loss_weight,
    SharedCalibRegressor,
    validate_cell,
)
from train_balanced_sampling_sep_mlp_shared_calib_step14_pseudo_label_feat_opt import (
    CKPT_R2_TIE_EPS_DEFAULT,
    PseudoLabelDataset,
    WeightedDataset,
    select_high_confidence_pseudo,
    predict_unlabeled,
    _load_hgat_encoder_state_dict,
    _maybe_load_external_hgat_checkpoint,
)


def _expand_to_rounds(vals, num_rounds, cast_fn):
    out = [cast_fn(v) for v in vals]
    if len(out) == 0:
        raise ValueError("Schedule cannot be empty.")
    if len(out) < num_rounds:
        out.extend([out[-1]] * (num_rounds - len(out)))
    return out[:num_rounds]


def _parse_float_schedule(raw, num_rounds, default_val):
    if raw is None or str(raw).strip() == "":
        return [float(default_val)] * num_rounds
    txt = str(raw).strip()
    vals = [x.strip() for x in txt.split(",") if x.strip() != ""]
    return _expand_to_rounds(vals, num_rounds, float)


def _parse_bool_schedule(raw, num_rounds, default_val):
    if raw is None or str(raw).strip() == "":
        return [bool(default_val)] * num_rounds
    txt = str(raw).strip().lower()
    vals = []
    for token in [x.strip() for x in txt.split(",") if x.strip() != ""]:
        if token in ("1", "true", "t", "yes", "y"):
            vals.append(True)
        elif token in ("0", "false", "f", "no", "n"):
            vals.append(False)
        else:
            raise ValueError(f"Invalid bool token in schedule: {token}")
    return _expand_to_rounds(vals, num_rounds, bool)


def _normalize_keep_ratio(v):
    return float(np.clip(float(v), 0.0, 1.0))


def train_step18(options, seed):
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
    use_amp = bool(getattr(options, "use_amp", True))
    amp_on = bool(use_amp and device.type == "cuda")

    data_dir = options.data_save_path
    obj = load_dataset_pkl(data_dir, getattr(options, "dataset_pkl_name", "dataset.pkl"))
    meta = obj.get("meta", {})
    scaler_stats = obj.get("scaler_stats", {})
    y_scaler = obj.get("y_scaler", {})
    df_src = obj.get("src_df")
    df_tgt_train = obj.get("tgt_train_df")
    df_tgt_val = obj.get("tgt_val_df")

    # unlabeled target data
    unlabeled_csv = os.path.join(data_dir, "tgt_unlabeled.csv")
    df_tgt_unlabeled = None
    if os.path.exists(unlabeled_csv):
        import pandas as pd
        df_tgt_unlabeled = pd.read_csv(unlabeled_csv)

    x_mean = np.array([scaler_stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([scaler_stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean = float(y_scaler.get("mean", 0.0))
    y_std = float(y_scaler.get("std", 1.0))
    if abs(y_std) < 1e-12:
        y_std = 1.0

    has_unlabeled = df_tgt_unlabeled is not None and len(df_tgt_unlabeled) > 0
    print(f"[Step18] Unlabeled target samples: {len(df_tgt_unlabeled) if has_unlabeled else 0}")

    df_src_use = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    def my_collate_tgt(batch):
        xs, ys, cts, ws = zip(*batch)
        return th.stack(xs), th.stack(ys), cts, th.tensor(ws, dtype=th.float32)

    num_workers = int(getattr(options, "num_workers", 4))
    dl_kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "collate_fn": my_collate}
    dl_tgt_kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "collate_fn": my_collate_tgt}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2
        dl_tgt_kwargs["persistent_workers"] = True
        dl_tgt_kwargs["prefetch_factor"] = 2

    design_dim = getattr(options, "design_dim", options.out_dim)
    hgat_hid = getattr(options, "hgat_hid", options.hidden_dim)
    hgat_heads = getattr(options, "hgat_heads", options.num_heads)
    dropout = getattr(options, "mlp_dropout", 0.0)
    in_map = {"NET": GRAPH_NET_DIM, "PMOS": GRAPH_MOS_DIM, "NMOS": GRAPH_MOS_DIM}

    # schedules
    num_rounds = max(1, int(getattr(options, "pseudo_num_rounds", 3)))
    epochs_per_round = max(1, options.num_epoch // num_rounds)
    pseudo_min_keep = max(1, int(getattr(options, "pseudo_min_keep", 16)))
    keep_ratio_default = float(getattr(options, "pseudo_keep_ratio", 1.0))
    pl_weight_default = max(0.0, float(getattr(options, "pseudo_loss_weight", 1.0)))
    freeze_default = bool(getattr(options, "freeze_hgat", False))

    keep_ratio_schedule = _parse_float_schedule(
        getattr(options, "pseudo_keep_ratio_schedule", ""),
        num_rounds,
        keep_ratio_default,
    )
    keep_ratio_schedule = [_normalize_keep_ratio(x) for x in keep_ratio_schedule]
    pl_weight_schedule = _parse_float_schedule(
        getattr(options, "pseudo_loss_weight_schedule", ""),
        num_rounds,
        pl_weight_default,
    )
    pl_weight_schedule = [max(0.0, float(x)) for x in pl_weight_schedule]
    freeze_schedule = _parse_bool_schedule(
        getattr(options, "round_freeze_schedule", ""),
        num_rounds,
        freeze_default,
    )

    print(
        f"[Step18] Settings: rounds={num_rounds}, epochs_per_round={epochs_per_round}, "
        f"min_keep={pseudo_min_keep}"
    )
    print(f"[Step18] keep_ratio_schedule={keep_ratio_schedule}")
    print(f"[Step18] pseudo_loss_weight_schedule={pl_weight_schedule}")
    print(f"[Step18] freeze_schedule={freeze_schedule}")

    # build graph cache once
    graph_cache_src = build_src_graph_cache(data_dir, meta, device)
    graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)

    # coverage logs
    ds_tgt_cov = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std)
    ds_val_cov = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std)
    ds_src_cov = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std)
    report_graph_cache_coverage("src", graph_cache_src, list(ds_src_cov.cts))
    report_graph_cache_coverage("tgt", graph_cache_tgt, list(ds_tgt_cov.cts) + list(ds_val_cov.cts))

    pseudo_label_data = None
    global_best_val = float("-inf")
    global_best_epoch = -1
    global_best_loss = float("inf")
    round_best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_round_best.pt")
    best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_best.pt")

    for rnd in range(num_rounds):
        print(f"\n{'=' * 60}")
        print(f"[Step18] Self-training round {rnd + 1}/{num_rounds}")
        print(f"{'=' * 60}")

        current_freeze = bool(freeze_schedule[rnd])
        current_pl_weight = float(pl_weight_schedule[rnd])
        print(f"[Step18] Round mode: freeze_hgat={current_freeze}, pseudo_loss_weight={current_pl_weight:.4f}")

        ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std)
        ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std)
        val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std)

        ds_tgt_w = WeightedDataset(ds_tgt, sample_weight=1.0)
        if pseudo_label_data is not None:
            pl_ds = WeightedDataset(PseudoLabelDataset(*pseudo_label_data), sample_weight=current_pl_weight)
            combined_tgt = ConcatDataset([ds_tgt_w, pl_ds])
            print(
                f"[Step18] Training with {len(ds_tgt)} labeled + {len(pl_ds)} pseudo-labeled = "
                f"{len(combined_tgt)} target samples"
            )
        else:
            combined_tgt = ds_tgt_w
            print(f"[Step18] Training with {len(ds_tgt)} labeled target samples")

        batch_size_tgt = max(1, options.batch_size // 2)
        batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

        dl_tgt = DataLoader(combined_tgt, batch_size=batch_size_tgt, shuffle=True, **dl_tgt_kwargs)
        dl_src = DataLoader(ds_src, batch_size=batch_size_src, shuffle=True, **dl_kwargs)
        val_dl = DataLoader(val_ds, batch_size=options.batch_size, shuffle=False, **dl_kwargs)

        enc = HGATDesignEncoder(
            in_dim_map=in_map,
            hid=hgat_hid,
            out=design_dim,
            num_heads=hgat_heads,
            num_layers=getattr(options, "hgat_layers", 2),
            dropout=getattr(options, "hgat_dropout", 0.1),
            use_net_readout=getattr(options, "hgat_use_net_readout", False),
            type_attn_readout=getattr(options, "hgat_type_attn_readout", False),
        ).to(device)
        model = SharedCalibRegressor(in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout).to(device)

        if rnd == 0:
            _maybe_load_external_hgat_checkpoint(enc, options, device)

        if rnd > 0 and os.path.exists(round_best_ckpt_path):
            ckpt = th.load(round_best_ckpt_path, map_location=device)
            _load_hgat_encoder_state_dict(enc, ckpt["enc"])
            model.load_state_dict(ckpt["model"])
            print(f"[Step18] Warm-started from round {rnd} best checkpoint")

        z_dict_src = None
        z_dict_tgt = None
        if current_freeze:
            z_dict_src = precompute_z_from_graph_cache(graph_cache_src, enc)
            z_dict_tgt = precompute_z_from_graph_cache(graph_cache_tgt, enc)
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()
            optimizer = th.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
        else:
            enc_lr = float(options.learning_rate) * ENCODER_LR_SCALE_DEFAULT
            optimizer = th.optim.Adam(
                [
                    {"params": enc.parameters(), "lr": enc_lr},
                    {"params": model.parameters(), "lr": options.learning_rate},
                ],
                weight_decay=options.weight_decay,
            )

        loss_fn = nn.MSELoss()
        r2_score = R2Score().to(device)
        scaler = th.cuda.amp.GradScaler(enabled=amp_on)
        best_val = float("-inf")
        best_epoch = -1
        best_val_loss = float("inf")

        for epoch in range(epochs_per_round):
            if current_freeze:
                enc.eval()
            else:
                enc.train()
            model.train()
            r2_score.reset()
            total_loss = 0.0
            total_n = 0.0
            src_w = compute_src_loss_weight(rnd * epochs_per_round + epoch, options)

            dl_src_iter = iter(dl_src)
            for xb_tgt, yb_tgt, cts_tgt, wb_tgt in dl_tgt:
                xb_tgt = xb_tgt.to(device, non_blocking=True)
                yb_tgt = yb_tgt.to(device, non_blocking=True)
                wb_tgt = wb_tgt.to(device, non_blocking=True)
                tgt_w_sum = th.clamp(wb_tgt.sum(), min=1e-12)
                src_loss_sum = 0.0
                src_n_sum = 0
                z_step_cache = {} if not current_freeze else None

                with th.cuda.amp.autocast(enabled=amp_on):
                    zb_tgt = build_z_batch(
                        cts_tgt,
                        device,
                        design_dim,
                        z_dict=z_dict_tgt,
                        graph_cache=graph_cache_tgt,
                        enc=enc,
                        dedup=options.dedup_z,
                        z_step_cache=z_step_cache,
                        use_batched_graph_encode=True,
                    )
                    pred_tgt = model(xb_tgt, zb_tgt, node="tgt")
                    loss_tgt_vec = F.mse_loss(pred_tgt, yb_tgt, reduction="none").view(-1)
                    loss_tgt_sum = th.sum(loss_tgt_vec * wb_tgt.view(-1))

                for _ in range(max(1, options.sample_45_num)):
                    try:
                        xb_src, yb_src, cts_src = next(dl_src_iter)
                    except StopIteration:
                        dl_src_iter = iter(dl_src)
                        xb_src, yb_src, cts_src = next(dl_src_iter)
                    xb_src = xb_src.to(device, non_blocking=True)
                    yb_src = yb_src.to(device, non_blocking=True)
                    with th.cuda.amp.autocast(enabled=amp_on):
                        zb_src = build_z_batch(
                            cts_src,
                            device,
                            design_dim,
                            z_dict=z_dict_src,
                            graph_cache=graph_cache_src,
                            enc=enc,
                            dedup=options.dedup_z,
                            z_step_cache=z_step_cache,
                            use_batched_graph_encode=True,
                        )
                        pred_src = model(xb_src, zb_src, node="src")
                        loss_src = loss_fn(pred_src, yb_src)
                    src_loss_sum += loss_src * len(yb_src)
                    src_n_sum += len(yb_src)

                denom = float(tgt_w_sum.item()) + float(src_n_sum)
                denom = max(1e-12, denom)
                with th.cuda.amp.autocast(enabled=amp_on):
                    total_batch = (loss_tgt_sum + src_w * src_loss_sum) / denom

                optimizer.zero_grad()
                scaler.scale(total_batch).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += total_batch.item() * float(tgt_w_sum.item())
                total_n += float(tgt_w_sum.item())
                r2_score.update(pred_tgt.float(), yb_tgt.float())

            train_loss = total_loss / max(1.0, total_n)
            train_r2 = r2_score.compute().item() if total_n > 0 else 0.0
            val_loss, val_r2 = validate_cell(
                val_dl,
                enc,
                model,
                device,
                design_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                dedup=options.dedup_z,
                use_amp=use_amp,
            )

            global_ep = rnd * epochs_per_round + epoch
            print(
                f"r{rnd + 1} e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
                f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, src_w:{src_w:.4f}"
            )

            better_r2 = val_r2 > (best_val + CKPT_R2_TIE_EPS_DEFAULT)
            tie_better = abs(val_r2 - best_val) <= CKPT_R2_TIE_EPS_DEFAULT and val_loss < best_val_loss
            if better_r2 or tie_better:
                best_val = val_r2
                best_epoch = global_ep + 1
                best_val_loss = val_loss
                os.makedirs(options.model_saving_dir, exist_ok=True)
                th.save(
                    {
                        "enc": enc.state_dict(),
                        "model": model.state_dict(),
                        "design_dim": design_dim,
                        "hgat_hid": hgat_hid,
                        "hgat_heads": hgat_heads,
                        "scaler_stats": scaler_stats,
                        "y_scaler": y_scaler,
                        "epoch": global_ep + 1,
                        "best_val_r2": best_val,
                        "best_val_loss": best_val_loss,
                        "round": rnd + 1,
                        "script": "step18_pseudo_label_feat_schedule",
                    },
                    round_best_ckpt_path,
                )

                if val_r2 > global_best_val + CKPT_R2_TIE_EPS_DEFAULT or (
                    abs(val_r2 - global_best_val) <= CKPT_R2_TIE_EPS_DEFAULT and val_loss < global_best_loss
                ):
                    global_best_val = val_r2
                    global_best_epoch = global_ep + 1
                    global_best_loss = val_loss
                    th.save(
                        {
                            "enc": enc.state_dict(),
                            "model": model.state_dict(),
                            "design_dim": design_dim,
                            "hgat_hid": hgat_hid,
                            "hgat_heads": hgat_heads,
                            "scaler_stats": scaler_stats,
                            "y_scaler": y_scaler,
                            "epoch": global_ep + 1,
                            "best_val_r2": global_best_val,
                            "best_val_loss": global_best_loss,
                            "round": rnd + 1,
                            "script": "step18_pseudo_label_feat_schedule",
                        },
                        best_ckpt_path,
                    )
                print("Model successfully saved")

        print(f"[Round {rnd + 1} Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, val_loss:{best_val_loss:.6f}")

        # pseudo label generation for next round
        if has_unlabeled and rnd < num_rounds - 1:
            ckpt = th.load(round_best_ckpt_path, map_location=device)
            _load_hgat_encoder_state_dict(enc, ckpt["enc"])
            model.load_state_dict(ckpt["model"])

            preds_norm, x_norm, cts = predict_unlabeled(
                enc,
                model,
                device,
                design_dim,
                df_tgt_unlabeled,
                x_mean,
                x_std,
                y_mean,
                y_std,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                dedup=options.dedup_z,
                use_amp=use_amp,
            )

            keep_ratio_next = keep_ratio_schedule[min(rnd + 1, num_rounds - 1)]
            x_pl, y_pl, cts_pl, keep_idx, dmin = select_high_confidence_pseudo(
                x_norm,
                preds_norm,
                cts,
                ds_tgt.x,
                keep_ratio=keep_ratio_next,
                min_keep=pseudo_min_keep,
            )
            pseudo_label_data = (x_pl, y_pl, cts_pl)

            if len(keep_idx) > 0 and len(dmin) > 0:
                d_sorted = np.sort(dmin[keep_idx])
                d_msg = f"dist[min/med/max]={d_sorted[0]:.4f}/{np.median(d_sorted):.4f}/{d_sorted[-1]:.4f}"
            else:
                d_msg = "dist[min/med/max]=n/a"
            print(
                f"[Step18] Generated {len(preds_norm)} pseudo-labels, selected {len(x_pl)} "
                f"(keep_ratio_next={keep_ratio_next:.2f}), {d_msg}"
            )

    if global_best_epoch > 0:
        print(
            f"\n[Global Best] epoch:{global_best_epoch}, val_r2:{global_best_val:.4f}, "
            f"val_loss:{global_best_loss:.6f}, ckpt:{best_ckpt_path}"
        )


if __name__ == "__main__":
    options = get_options()
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