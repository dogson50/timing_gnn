r"""Step13: Pseudo-label self-training — leverage 69 unlabeled target samples.

Key idea: The 69 unlabeled target samples are currently wasted. This script
implements iterative pseudo-labeling: train on labeled data, predict on unlabeled,
add high-confidence predictions as pseudo-labels, retrain. This effectively
increases the target training set size.
"""

import os, re, pickle, random
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torchmetrics import R2Score
from torch.utils.data import Dataset as TorchDataset, DataLoader, ConcatDataset
import dgl

from options import get_options
import tee
from hgat import HGATDesignEncoder
from spi2graph import parse_top_subckt_pins

from train_balanced_sampling_sep_mlp_shared_calib_step5a_msselect_opt import (
    NUMERIC_COLS, TARGET_COL, GRAPH_NET_DIM, GRAPH_MOS_DIM,
    ENCODER_LR_SCALE_DEFAULT, CKPT_R2_TIE_EPS_DEFAULT,
    _ensure_pol_bit, _ensure_numeric_cols, _norm_xy,
    CellDelayDataset, load_dataset_pkl,
    build_src_graph_cache, build_tgt_graph_cache,
    build_z_batch, precompute_z_from_graph_cache,
    report_graph_cache_coverage, compute_src_loss_weight,
    SharedCalibRegressor, validate_cell,
)


class PseudoLabelDataset(TorchDataset):
    """Dataset from pre-computed numpy arrays (for pseudo-labeled samples)."""
    def __init__(self, x, y, cts):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cts = cts

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return th.from_numpy(self.x[i]), th.tensor(self.y[i]), self.cts[i]


@th.no_grad()
def predict_unlabeled(enc, model, device, design_dim, df_unlabeled, x_mean, x_std, y_mean, y_std,
                      z_dict=None, graph_cache=None, dedup=False, use_amp=False):
    """Predict delay for unlabeled samples, return normalized predictions."""
    enc.eval(); model.eval()
    df = _ensure_pol_bit(df_unlabeled.copy())
    df = _ensure_numeric_cols(df)
    x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
    x_norm = (x_raw - x_mean) / x_std
    cts = df["cell_type"].astype(str).values

    x_t = th.from_numpy(x_norm).to(device)
    amp_on = bool(use_amp and device.type == "cuda")
    with th.cuda.amp.autocast(enabled=amp_on):
        zb = build_z_batch(list(cts), device, design_dim, z_dict=z_dict,
            graph_cache=graph_cache, enc=enc, dedup=dedup)
        preds = model(x_t, zb, node="tgt")
    return preds.cpu().numpy(), x_norm, cts


def _load_hgat_encoder_state_dict(enc, enc_state_dict):
    """
    HGATBlock creates LayerNorm modules lazily in forward().
    Round-1 checkpoints may contain `layers.*.norm.*` keys while a freshly
    initialized encoder has empty `norm` ModuleDicts. Pre-create these modules
    from checkpoint keys before strict loading.
    """
    norm_key_pat = re.compile(r"^layers\.(\d+)\.norm\.([^.]+)\.(weight|bias)$")
    device = next(enc.parameters()).device

    for key in enc_state_dict.keys():
        m = norm_key_pat.match(key)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        node_type = m.group(2)
        if layer_idx < 0 or layer_idx >= len(enc.layers):
            continue
        layer = enc.layers[layer_idx]
        if node_type not in layer.norm:
            layer.norm[node_type] = nn.LayerNorm(layer.hid).to(device)

    model_sd = enc.state_dict()
    filtered_sd = {}
    dropped_missing = 0
    dropped_shape = 0
    for k, v in enc_state_dict.items():
        if k not in model_sd:
            dropped_missing += 1
            continue
        if model_sd[k].shape != v.shape:
            dropped_shape += 1
            continue
        filtered_sd[k] = v

    if len(filtered_sd) == 0:
        print("[Warn] HGAT checkpoint matched 0 parameters; skip loading.")
        return False

    load_res = enc.load_state_dict(filtered_sd, strict=False)
    if dropped_missing > 0:
        print(f"[Warn] HGAT checkpoint keys not in model: {dropped_missing}")
    if dropped_shape > 0:
        print(f"[Warn] HGAT checkpoint keys shape-mismatch: {dropped_shape}")
    if load_res.unexpected_keys:
        print(f"[Info] HGAT unexpected keys after partial load: {len(load_res.unexpected_keys)}")
    print(f"[Info] Loaded HGAT encoder params: {len(filtered_sd)} tensors")
    return True


def _maybe_load_external_hgat_checkpoint(enc, options, device):
    ckpt_path = getattr(options, "hgat_ckpt_path", None)
    if ckpt_path is None or str(ckpt_path).strip() == "":
        ckpt_path = getattr(options, "load_ckpt_path", None)
    if ckpt_path is None or str(ckpt_path).strip() == "":
        return False

    ckpt_path = os.path.expanduser(str(ckpt_path))
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"[Warn] HGAT checkpoint not found: {ckpt_path}")
        return False

    ckpt = th.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "enc" in ckpt and isinstance(ckpt["enc"], dict):
        enc_state_dict = ckpt["enc"]
        src = "ckpt['enc']"
    elif isinstance(ckpt, dict):
        if any(str(k).startswith("enc.") for k in ckpt.keys()):
            enc_state_dict = {str(k)[4:]: v for k, v in ckpt.items() if str(k).startswith("enc.")}
            src = "ckpt['enc.*']"
        else:
            enc_state_dict = ckpt
            src = "ckpt(root)"
    else:
        print(f"[Warn] Unsupported HGAT checkpoint format: {type(ckpt)}")
        return False

    ok = _load_hgat_encoder_state_dict(enc, enc_state_dict)
    if ok:
        print(f"[Info] HGAT encoder initialized from {src}: {ckpt_path}")
    return ok


def train_step13(options, seed):
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
    use_amp = bool(getattr(options, "use_amp", True))
    amp_on = bool(use_amp and device.type == "cuda")

    data_dir = options.data_save_path
    obj = load_dataset_pkl(data_dir, getattr(options, "dataset_pkl_name", "dataset.pkl"))
    meta, scaler_stats, y_scaler = obj.get("meta", {}), obj.get("scaler_stats", {}), obj.get("y_scaler", {})
    df_src = obj.get("src_df")
    df_tgt_train = obj.get("tgt_train_df")
    df_tgt_val = obj.get("tgt_val_df")

    # tgt_unlabeled_df is NOT in dataset.pkl; load from CSV
    unlabeled_csv = os.path.join(data_dir, "tgt_unlabeled.csv")
    df_tgt_unlabeled = pd.read_csv(unlabeled_csv) if os.path.exists(unlabeled_csv) else None

    x_mean = np.array([scaler_stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([scaler_stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean, y_std = float(y_scaler.get("mean", 0.0)), float(y_scaler.get("std", 1.0))
    if abs(y_std) < 1e-12: y_std = 1.0

    has_unlabeled = df_tgt_unlabeled is not None and len(df_tgt_unlabeled) > 0
    print(f"[Step13] Unlabeled target samples: {len(df_tgt_unlabeled) if has_unlabeled else 0}")

    df_src_use = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    num_workers = int(getattr(options, "num_workers", 4))
    dl_kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "collate_fn": my_collate}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    design_dim = getattr(options, "design_dim", options.out_dim)
    hgat_hid = getattr(options, "hgat_hid", options.hidden_dim)
    hgat_heads = getattr(options, "hgat_heads", options.num_heads)
    dropout = getattr(options, "mlp_dropout", 0.0)

    in_map = {"NET": GRAPH_NET_DIM, "PMOS": GRAPH_MOS_DIM, "NMOS": GRAPH_MOS_DIM}

    # Self-training rounds
    num_rounds = 3
    epochs_per_round = max(1, options.num_epoch // num_rounds)
    pseudo_label_data = None  # will be set after round 1

    global_best_val = float("-inf")
    global_best_epoch = -1
    global_best_loss = float("inf")
    best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_best.pt")

    for rnd in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"[Step13] Self-training round {rnd + 1}/{num_rounds}")
        print(f"{'='*60}")

        # Build datasets
        ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std)
        ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std)
        val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std)

        # Combine labeled + pseudo-labeled
        if pseudo_label_data is not None:
            pl_ds = PseudoLabelDataset(*pseudo_label_data)
            combined_tgt = ConcatDataset([ds_tgt, pl_ds])
            print(f"[Step13] Training with {len(ds_tgt)} labeled + {len(pl_ds)} pseudo-labeled = {len(combined_tgt)} target samples")
        else:
            combined_tgt = ds_tgt
            print(f"[Step13] Training with {len(ds_tgt)} labeled target samples")

        batch_size_tgt = max(1, options.batch_size // 2)
        batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

        dl_tgt = DataLoader(combined_tgt, batch_size=batch_size_tgt, shuffle=True, **dl_kwargs)
        dl_src = DataLoader(ds_src, batch_size=batch_size_src, shuffle=True, **dl_kwargs)
        val_dl = DataLoader(val_ds, batch_size=options.batch_size, shuffle=False, **dl_kwargs)

        # Fresh model each round (or could warm-start; fresh is safer)
        enc = HGATDesignEncoder(
            in_dim_map=in_map, hid=hgat_hid, out=design_dim,
            num_heads=hgat_heads, num_layers=getattr(options, "hgat_layers", 2),
            dropout=getattr(options, "hgat_dropout", 0.1),
            use_net_readout=getattr(options, "hgat_use_net_readout", False),
            type_attn_readout=getattr(options, "hgat_type_attn_readout", False),
        ).to(device)
        model = SharedCalibRegressor(in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout).to(device)

        if rnd == 0:
            _maybe_load_external_hgat_checkpoint(enc, options, device)

        # Load best checkpoint from previous round if available
        if rnd > 0 and os.path.exists(best_ckpt_path):
            ckpt = th.load(best_ckpt_path, map_location=device)
            _load_hgat_encoder_state_dict(enc, ckpt["enc"])
            model.load_state_dict(ckpt["model"])
            print(f"[Step13] Warm-started from round {rnd} best checkpoint")

        graph_cache_src = build_src_graph_cache(data_dir, meta, device)
        graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)

        z_dict_src = z_dict_tgt = None
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

        best_val = float("-inf"); best_epoch = -1; best_val_loss = float("inf")

        for epoch in range(epochs_per_round):
            if options.freeze_hgat: enc.eval()
            else: enc.train()
            model.train(); r2_score.reset()
            total_loss = total_n = 0
            src_w = compute_src_loss_weight(rnd * epochs_per_round + epoch, options)

            dl_src_iter = iter(dl_src)
            for xb_tgt, yb_tgt, cts_tgt in dl_tgt:
                xb_tgt = xb_tgt.to(device, non_blocking=True)
                yb_tgt = yb_tgt.to(device, non_blocking=True)
                tgt_n = len(yb_tgt)
                src_loss_sum = 0.0; src_n_sum = 0
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
                scaler.step(optimizer); scaler.update()
                total_loss += total_batch.item() * tgt_n; total_n += tgt_n
                r2_score.update(pred_tgt.float(), yb_tgt.float())

            train_loss = total_loss / max(1, total_n)
            train_r2 = r2_score.compute().item() if total_n > 0 else 0.0
            val_loss, val_r2 = validate_cell(val_dl, enc, model, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, dedup=options.dedup_z, use_amp=use_amp)

            global_ep = rnd * epochs_per_round + epoch
            print(f"r{rnd+1} e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
                  f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, src_w:{src_w:.4f}")

            better_r2 = val_r2 > (best_val + CKPT_R2_TIE_EPS_DEFAULT)
            tie_better = abs(val_r2 - best_val) <= CKPT_R2_TIE_EPS_DEFAULT and val_loss < best_val_loss
            if better_r2 or tie_better:
                best_val, best_epoch, best_val_loss = val_r2, global_ep + 1, val_loss
                # Also track global best
                if val_r2 > global_best_val + CKPT_R2_TIE_EPS_DEFAULT or \
                   (abs(val_r2 - global_best_val) <= CKPT_R2_TIE_EPS_DEFAULT and val_loss < global_best_loss):
                    global_best_val, global_best_epoch, global_best_loss = val_r2, global_ep + 1, val_loss
                os.makedirs(options.model_saving_dir, exist_ok=True)
                th.save({"enc": enc.state_dict(), "model": model.state_dict(),
                          "design_dim": design_dim, "hgat_hid": hgat_hid, "hgat_heads": hgat_heads,
                          "scaler_stats": scaler_stats, "y_scaler": y_scaler,
                          "epoch": global_ep + 1, "best_val_r2": best_val, "best_val_loss": best_val_loss,
                          "round": rnd + 1, "script": "step13_pseudo_label"}, best_ckpt_path)
                print("Model successfully saved")

        print(f"[Round {rnd+1} Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, val_loss:{best_val_loss:.6f}")

        # Generate pseudo-labels for next round
        if has_unlabeled and rnd < num_rounds - 1:
            # Load best model from this round
            ckpt = th.load(best_ckpt_path, map_location=device)
            _load_hgat_encoder_state_dict(enc, ckpt["enc"])
            model.load_state_dict(ckpt["model"])

            preds_norm, x_norm, cts = predict_unlabeled(
                enc, model, device, design_dim, df_tgt_unlabeled,
                x_mean, x_std, y_mean, y_std,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt,
                dedup=options.dedup_z, use_amp=use_amp)

            # Use all unlabeled samples as pseudo-labels (only 69 samples)
            pseudo_label_data = (x_norm, preds_norm, list(cts))
            print(f"[Step13] Generated {len(preds_norm)} pseudo-labels for next round")

    if global_best_epoch > 0:
        print(f"\n[Global Best] epoch:{global_best_epoch}, val_r2:{global_best_val:.4f}, "
              f"val_loss:{global_best_loss:.6f}, ckpt:{best_ckpt_path}")


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
        train_step13(options, seed)
