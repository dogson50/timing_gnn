r"""Step9: Source loss annealing — gradually reduce source domain weight during training.

Key idea: Early epochs leverage source data for general pattern learning;
later epochs focus on target domain by annealing source loss weight to a small value.
"""

import os, re, pickle, random
import numpy as np
import torch as th
import torch.nn as nn
from torchmetrics import R2Score
from torch.utils.data import Dataset as TorchDataset, DataLoader
import dgl

from options import get_options
import tee
from test_r2_report import run_train_and_report_test
from hgat import HGATDesignEncoder
from spi2graph import parse_top_subckt_pins

# ---- reuse step5 utilities ----
from train_balanced_sampling_sep_mlp_shared_calib_step5a_msselect_opt import (
    NUMERIC_COLS, TARGET_COL, GRAPH_NET_DIM, GRAPH_MOS_DIM,
    ENCODER_LR_SCALE_DEFAULT, CKPT_R2_TIE_EPS_DEFAULT,
    _ensure_pol_bit, _ensure_numeric_cols, _norm_xy,
    CellDelayDataset, load_dataset_pkl,
    extract_subckt_text, parse_transistors_spice_rich,
    build_dgl_graph_from_devs_step5,
    build_src_graph_cache, build_tgt_graph_cache,
    encode_graph_batch_hgat, build_z_batch,
    precompute_z_from_graph_cache, report_graph_cache_coverage,
    SharedCalibRegressor, validate_cell,
)


def train_step9(options, seed):
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

    x_mean = np.array([scaler_stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([scaler_stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean, y_std = float(y_scaler.get("mean", 0.0)), float(y_scaler.get("std", 1.0))
    if abs(y_std) < 1e-12: y_std = 1.0

    df_src_use = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train
    ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std)
    ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std)
    val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std)

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    batch_size_tgt = max(1, options.batch_size // 2)
    batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))
    num_workers = int(getattr(options, "num_workers", 4))
    pin_memory = bool(getattr(options, "pin_memory", device.type == "cuda"))
    persistent_workers = bool(getattr(options, "persistent_workers", num_workers > 0))
    dl_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory, "collate_fn": my_collate}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = 2

    dl_tgt = DataLoader(ds_tgt, batch_size=batch_size_tgt, shuffle=True, **dl_kwargs)
    dl_src = DataLoader(ds_src, batch_size=batch_size_src, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=options.batch_size, shuffle=False, **dl_kwargs)

    design_dim = getattr(options, "design_dim", options.out_dim)
    hgat_hid = getattr(options, "hgat_hid", options.hidden_dim)
    hgat_heads = getattr(options, "hgat_heads", options.num_heads)
    hgat_layers = getattr(options, "hgat_layers", 2)
    hgat_dropout = getattr(options, "hgat_dropout", 0.1)
    dropout = getattr(options, "mlp_dropout", 0.0)

    in_map = {"NET": GRAPH_NET_DIM, "PMOS": GRAPH_MOS_DIM, "NMOS": GRAPH_MOS_DIM}
    enc = HGATDesignEncoder(
        in_dim_map=in_map, hid=hgat_hid, out=design_dim,
        num_heads=hgat_heads, num_layers=hgat_layers, dropout=hgat_dropout,
        use_net_readout=getattr(options, "hgat_use_net_readout", False),
        type_attn_readout=getattr(options, "hgat_type_attn_readout", False),
    ).to(device)
    model = SharedCalibRegressor(in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout).to(device)

    print("----------------Loading HGAT graphs----------------")
    graph_cache_src = build_src_graph_cache(data_dir, meta, device)
    graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)
    tgt_cts_all = list(ds_tgt.cts) + list(val_ds.cts)
    src_cts_all = list(ds_src.cts)
    report_graph_cache_coverage("src", graph_cache_src, src_cts_all)
    report_graph_cache_coverage("tgt", graph_cache_tgt, tgt_cts_all)

    z_dict_src, z_dict_tgt = None, None
    if options.freeze_hgat:
        z_dict_src = precompute_z_from_graph_cache(graph_cache_src, enc)
        z_dict_tgt = precompute_z_from_graph_cache(graph_cache_tgt, enc)
        for p in enc.parameters(): p.requires_grad = False
        enc.eval()
        optimizer = th.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    else:
        enc_lr = float(options.learning_rate) * ENCODER_LR_SCALE_DEFAULT
        optimizer = th.optim.Adam([
            {"params": enc.parameters(), "lr": enc_lr},
            {"params": model.parameters(), "lr": options.learning_rate},
        ], weight_decay=options.weight_decay)

    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)
    scaler = th.cuda.amp.GradScaler(enabled=amp_on)

    # --- Annealing config ---
    anneal_start = int(getattr(options, "src_loss_anneal_start", -1))
    anneal_end = int(getattr(options, "src_loss_anneal_end", -1))
    final_scale = float(getattr(options, "src_loss_final_scale", 0.1))
    base_w = float(options.loss_weight_45)
    # If user didn't set anneal params, use sensible defaults
    if anneal_start < 1:
        anneal_start = max(1, options.num_epoch // 5)
    if anneal_end < 1:
        anneal_end = max(anneal_start + 1, options.num_epoch * 3 // 5)
    print(f"[Step9] Anneal: start={anneal_start}, end={anneal_end}, final_scale={final_scale}")

    def get_src_weight(ep):
        if ep + 1 <= anneal_start:
            return base_w
        if ep + 1 >= anneal_end:
            return base_w * final_scale
        t = float(ep + 1 - anneal_start) / float(anneal_end - anneal_start)
        return base_w * (1.0 + (final_scale - 1.0) * t)

    print("----------------Start training---------------")
    best_val = float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    ckpt_r2_tie_eps = CKPT_R2_TIE_EPS_DEFAULT
    best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_best.pt")

    for epoch in range(options.num_epoch):
        if options.freeze_hgat: enc.eval()
        else: enc.train()
        model.train()
        r2_score.reset()
        total_loss = total_n = 0
        src_w = get_src_weight(epoch)

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt in dl_tgt:
            xb_tgt = xb_tgt.to(device, non_blocking=True)
            yb_tgt = yb_tgt.to(device, non_blocking=True)
            tgt_n = len(yb_tgt)
            src_loss_sum = 0.0
            src_n_sum = 0
            z_step_cache = {} if not options.freeze_hgat else None

            with th.cuda.amp.autocast(enabled=amp_on):
                zb_tgt = build_z_batch(cts_tgt, device, design_dim, z_dict=z_dict_tgt,
                    graph_cache=graph_cache_tgt, enc=enc, dedup=options.dedup_z,
                    z_step_cache=z_step_cache, use_batched_graph_encode=True)
                pred_tgt = model(xb_tgt, zb_tgt, node="tgt")
                loss_tgt = loss_fn(pred_tgt, yb_tgt)

            for _ in range(max(1, options.sample_45_num)):
                try: xb_src, yb_src, cts_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src = next(dl_src_iter)
                xb_src = xb_src.to(device, non_blocking=True)
                yb_src = yb_src.to(device, non_blocking=True)
                with th.cuda.amp.autocast(enabled=amp_on):
                    zb_src = build_z_batch(cts_src, device, design_dim, z_dict=z_dict_src,
                        graph_cache=graph_cache_src, enc=enc, dedup=options.dedup_z,
                        z_step_cache=z_step_cache, use_batched_graph_encode=True)
                    pred_src = model(xb_src, zb_src, node="src")
                    loss_src = loss_fn(pred_src, yb_src)
                src_loss_sum += loss_src * len(yb_src)
                src_n_sum += len(yb_src)

            denom = max(1, tgt_n + src_n_sum)
            with th.cuda.amp.autocast(enabled=amp_on):
                total_batch = (loss_tgt * tgt_n + src_w * src_loss_sum) / denom

            optimizer.zero_grad()
            scaler.scale(total_batch).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_batch.item() * tgt_n
            total_n += tgt_n
            r2_score.update(pred_tgt.float(), yb_tgt.float())

        train_loss = total_loss / max(1, total_n)
        train_r2 = r2_score.compute().item() if total_n > 0 else 0.0
        val_loss, val_r2 = validate_cell(val_dl, enc, model, device, design_dim,
            z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, dedup=options.dedup_z, use_amp=use_amp)

        print(f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
              f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, src_w:{src_w:.4f}")

        better_r2 = val_r2 > (best_val + ckpt_r2_tie_eps)
        tie_better = abs(val_r2 - best_val) <= ckpt_r2_tie_eps and val_loss < best_val_loss
        if better_r2 or tie_better:
            best_val, best_epoch, best_val_loss = val_r2, epoch + 1, val_loss
            os.makedirs(options.model_saving_dir, exist_ok=True)
            th.save({"enc": enc.state_dict(), "model": model.state_dict(),
                      "design_dim": design_dim, "hgat_hid": hgat_hid, "hgat_heads": hgat_heads,
                      "scaler_stats": scaler_stats, "y_scaler": y_scaler,
                      "epoch": epoch + 1, "best_val_r2": best_val, "best_val_loss": best_val_loss,
                      "script": "step9_anneal"}, best_ckpt_path)
            print("Model successfully saved")

    if best_epoch > 0:
        print(f"[Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, val_loss:{best_val_loss:.6f}, ckpt:{best_ckpt_path}")


if __name__ == "__main__":
    options = get_options()
    seed = options.seed
    th.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if th.cuda.is_available(): th.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    os.makedirs(copilot_log_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")
    stdout_f = os.path.join(options.model_saving_dir, "stdout.log")
    stderr_f = os.path.join(options.model_saving_dir, "stderr.log")
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        run_train_and_report_test(train_step9, options, seed, script_name=__file__)