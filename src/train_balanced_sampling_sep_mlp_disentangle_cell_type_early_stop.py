import os
import pickle
import random
import numpy as np
import torch as th
from torchmetrics import R2Score
from torch.utils.data import DataLoader

import tee
from test_r2_report import run_train_and_report_test
from options import get_options
from train_balanced_sampling_sep_mlp_disentangle_cell_type import (
    NUMERIC_COLS,
    CellDelayDataset,
    DisentangleCellDelayRegressor,
    HGATDesignEncoder,
    _build_arc_vocabs,
    _build_semantic_pin_role_maps,
    build_alignment_labels,
    build_src_graph_cache,
    build_tgt_graph_cache,
    build_z_batch,
    conditional_cmd_loss,
    get_hgat_in_dim_map,
    load_dataset_pkl,
    precompute_z_from_graph_cache,
    supervised_contrastive_loss,
    validate_cell,
)

PROCESS_NUMERIC_COLS = [
    "voltage",
    "temp",
    "inv_v",
    "inv_temp",
    "log_slew",
    "log_cap",
    "pol_bit",
]
PROCESS_FEATURE_IDX = [NUMERIC_COLS.index(c) for c in PROCESS_NUMERIC_COLS]


def _select_process_x(xb):
    return xb[:, PROCESS_FEATURE_IDX]


@th.no_grad()
def validate_cell_proc(val_dl, enc, model, device, design_dim, *, z_dict=None, graph_cache=None, dedup=False):
    enc.eval()
    model.eval()
    loss_fn = th.nn.MSELoss()
    r2_score = R2Score().to(device)
    total_loss = 0.0
    total_n = 0
    for xb, yb, cts, _, _, _, fp, tp, pol, sense, cond in val_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        fp = fp.to(device)
        tp = tp.to(device)
        pol = pol.to(device)
        sense = sense.to(device)
        cond = cond.to(device)
        xb_proc = _select_process_x(xb)
        zb = build_z_batch(
            cts, device, design_dim,
            z_dict=z_dict, graph_cache=graph_cache, enc=enc,
            dedup=dedup,
        )
        pred, _, _ = model(
            xb_proc, zb, node="tgt",
            from_pin_id=fp, to_pin_id=tp, pol_id=pol, sense_id=sense, cond_id=cond,
        )
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        r2_score.update(pred, yb)
    avg_loss = total_loss / max(1, total_n)
    val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
    return avg_loss, val_r2


def train_balanced_sep_mlp_disentangle_cell_type_early_stop(options, seed):
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

    if options.task != "reg":
        raise ValueError("Only regression task is supported in this training script.")

    data_dir = options.data_save_path
    obj = load_dataset_pkl(data_dir, getattr(options, "dataset_pkl_name", "dataset.pkl"))
    meta = obj.get("meta", {})
    scaler_stats = obj.get("scaler_stats", {})
    y_scaler = obj.get("y_scaler", {})
    df_src = obj.get("src_df")
    df_tgt_train = obj.get("tgt_train_df")
    df_tgt_val = obj.get("tgt_val_df")
    df_tgt_test = obj.get("tgt_test_df")
    if df_tgt_train is None or df_tgt_val is None:
        raise RuntimeError("dataset.pkl must include tgt_train_df and tgt_val_df")

    x_mean = np.array([scaler_stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([scaler_stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean = float(y_scaler.get("mean", 0.0))
    y_std = float(y_scaler.get("std", 1.0))
    if abs(y_std) < 1e-12:
        y_std = 1.0

    df_src_use = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train

    arc_vocab_scope = str(getattr(options, "arc_vocab_scope", "tgt")).lower()
    if arc_vocab_scope == "src_tgt":
        pin2id, cond2id = _build_arc_vocabs(df_src_use, df_tgt_train)
    elif arc_vocab_scope == "tgt":
        pin2id, cond2id = _build_arc_vocabs(df_tgt_train)
    else:
        raise ValueError(f"Unknown arc_vocab_scope: {arc_vocab_scope}")

    role_maps = _build_semantic_pin_role_maps(df_src_use, df_tgt_train, df_tgt_val, df_tgt_test)

    ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std, pin2id, cond2id, role_maps)
    ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std, pin2id, cond2id, role_maps)
    val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std, pin2id, cond2id, role_maps)
    test_ds = CellDelayDataset(df_tgt_test, x_mean, x_std, y_mean, y_std, pin2id, cond2id, role_maps) if df_tgt_test is not None else None

    def my_collate(batch):
        xs, ys, cts, sem_keys, cmd_keys, cmd_ctxs, fps, tps, pols, senses, conds = zip(*batch)
        return (
            th.stack(xs), th.stack(ys), cts, sem_keys, cmd_keys, th.stack(cmd_ctxs),
            th.stack(fps), th.stack(tps), th.stack(pols), th.stack(senses), th.stack(conds),
        )

    batch_size_tgt = max(1, options.batch_size // 2)
    batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

    dl_tgt = DataLoader(ds_tgt, batch_size=batch_size_tgt, shuffle=True, collate_fn=my_collate)
    dl_src = DataLoader(ds_src, batch_size=batch_size_src, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=options.batch_size, shuffle=False, collate_fn=my_collate)
    test_dl = (
        DataLoader(test_ds, batch_size=options.batch_size, shuffle=False, collate_fn=my_collate)
        if test_ds is not None else None
    )

    proc_in_dim = len(PROCESS_NUMERIC_COLS)
    design_dim = getattr(options, "design_dim", options.out_dim)
    hgat_hid = getattr(options, "hgat_hid", options.hidden_dim)
    hgat_heads = getattr(options, "hgat_heads", options.num_heads)
    hgat_layers = getattr(options, "hgat_layers", 2)
    hgat_dropout = getattr(options, "hgat_dropout", 0.1)
    hgat_use_net_readout = getattr(options, "hgat_use_net_readout", False)
    hgat_type_attn_readout = getattr(options, "hgat_type_attn_readout", False)
    hgat_l2_norm = getattr(options, "hgat_l2_norm", False)
    dropout = getattr(options, "mlp_dropout", 0.0)
    node_feat_dim = getattr(options, "node_feat_dim", 128)

    in_map = get_hgat_in_dim_map()
    enc = HGATDesignEncoder(
        in_dim_map=in_map,
        hid=hgat_hid,
        out=design_dim,
        num_heads=hgat_heads,
        num_layers=hgat_layers,
        dropout=hgat_dropout,
        use_net_readout=hgat_use_net_readout,
        type_attn_readout=hgat_type_attn_readout,
        l2_norm=hgat_l2_norm,
    ).to(device)
    model = DisentangleCellDelayRegressor(
        in_dim=proc_in_dim,
        design_dim=design_dim,
        node_feat_dim=node_feat_dim,
        hid=hgat_hid,
        dropout=dropout,
        clr_proj_dim=getattr(options, "clr_proj_dim", 64),
        cmd_proj_dim=getattr(options, "cmd_proj_dim", 64),
        num_pins=len(pin2id),
        num_conds=len(cond2id),
        pin_emb_dim=getattr(options, "pin_emb_dim", 8),
        pol_emb_dim=getattr(options, "pol_emb_dim", 2),
        sense_emb_dim=getattr(options, "sense_emb_dim", 4),
        cond_emb_dim=getattr(options, "cond_emb_dim", 8),
        use_arc_cond=getattr(options, "use_arc_cond", False),
        arc_sep_domain_emb=getattr(options, "arc_sep_domain_emb", False),
        arc_cond_mode=getattr(options, "arc_cond_mode", "concat"),
        src_use_arc_cond=not getattr(options, "disable_src_arc_cond", False),
    ).to(device)

    if options.load_ckpt_path is not None:
        ckpt_path = options.load_ckpt_path
        if os.path.isdir(ckpt_path):
            cand = os.path.join(ckpt_path, "ckpt_best.pt")
            if os.path.exists(cand):
                ckpt_path = cand
            else:
                cand = os.path.join(ckpt_path, "model.pkl")
                if os.path.exists(cand):
                    ckpt_path = cand
        if os.path.exists(ckpt_path):
            if ckpt_path.endswith(".pt"):
                ckpt = th.load(ckpt_path, map_location=device)
                if "enc" in ckpt:
                    enc.load_state_dict(ckpt["enc"])
                if "model" in ckpt:
                    msg = model.load_state_dict(ckpt["model"], strict=False)
                    if getattr(msg, "missing_keys", None) or getattr(msg, "unexpected_keys", None):
                        print(
                            f"[Warn] Loaded model with missing={list(msg.missing_keys)} "
                            f"unexpected={list(msg.unexpected_keys)}"
                        )
                print(f"[Info] Loaded ckpt from {ckpt_path}")
            else:
                with open(ckpt_path, "rb") as f:
                    _, model, enc = pickle.load(f)
                model = model.to(device)
                enc = enc.to(device)
                print(f"[Info] Loaded model.pkl from {ckpt_path}")
        else:
            print(f"[Warn] load_ckpt_path not found: {ckpt_path}")

    print("----------------Loading HGAT graphs----------------")
    z_dict_src = None
    z_dict_tgt = None
    graph_cache_src = None
    graph_cache_tgt = None
    if options.freeze_hgat:
        print("[Info] freeze_hgat=True, precomputing z")
        graph_cache_src = build_src_graph_cache(data_dir, meta, device)
        graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)
        z_dict_src = precompute_z_from_graph_cache(graph_cache_src, enc, device, design_dim)
        z_dict_tgt = precompute_z_from_graph_cache(graph_cache_tgt, enc, device, design_dim)
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
        optimizer = th.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    else:
        print("[Info] freeze_hgat=False, building graph cache")
        graph_cache_src = build_src_graph_cache(data_dir, meta, device)
        graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)
        optimizer = th.optim.Adam(
            list(enc.parameters()) + list(model.parameters()),
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )

    r2_score = R2Score().to(device)

    print("----------------Start training---------------")
    split_mode = meta.get("split_info", {}).get("split_mode", "unknown")
    print(f"[Info] Target split mode: {split_mode}")
    print(f"[Info] Process branch features: {PROCESS_NUMERIC_COLS}")
    print(
        f"[Info] CLR label mode: {getattr(options, 'clr_label_mode', 'semantic_arc')}, "
        f"CMD label mode: {getattr(options, 'cmd_label_mode', 'semantic_arc')}"
    )
    if test_dl is None:
        print("[Warn] dataset.pkl has no tgt_test_df, skipping test evaluation.")

    best_val = float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    best_test_r2 = 0.0
    best_test_loss = 0.0
    best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_best.pt")
    patience = max(0, int(getattr(options, "early_stop_patience", 0)))
    min_delta = float(getattr(options, "early_stop_min_delta", 0.0))
    stale_epochs = 0
    if patience > 0:
        print(f"[Info] Early stop enabled: patience={patience}, min_delta={min_delta}")

    for epoch in range(options.num_epoch):
        if options.freeze_hgat:
            enc.eval()
        else:
            enc.train()
        model.train()
        r2_score.reset()
        total_loss = 0.0
        total_n = 0
        reg_loss_sum = 0.0
        clr_loss_sum = 0.0
        cmd_loss_sum = 0.0
        batch_cnt = 0

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt, sem_tgt, cmd_tgt_keys, cmd_ctx_tgt, fp_tgt, tp_tgt, pol_tgt, sense_tgt, cond_tgt in dl_tgt:
            xb_tgt = xb_tgt.to(device)
            yb_tgt = yb_tgt.to(device)
            cmd_ctx_tgt = cmd_ctx_tgt.to(device)
            fp_tgt = fp_tgt.to(device)
            tp_tgt = tp_tgt.to(device)
            pol_tgt = pol_tgt.to(device)
            sense_tgt = sense_tgt.to(device)
            cond_tgt = cond_tgt.to(device)
            xb_proc_tgt = _select_process_x(xb_tgt)
            zb_tgt = build_z_batch(
                cts_tgt, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, enc=enc,
                dedup=options.dedup_z,
            )
            pred_tgt, h_design_tgt, h_process_tgt = model(
                xb_proc_tgt, zb_tgt, node="tgt",
                from_pin_id=fp_tgt, to_pin_id=tp_tgt, pol_id=pol_tgt, sense_id=sense_tgt, cond_id=cond_tgt,
            )
            loss_tgt = th.nn.functional.mse_loss(pred_tgt, yb_tgt)

            loss_src_list = []
            clr_list = []
            cmd_list = []
            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src, sem_src, cmd_src_keys, cmd_ctx_src, fp_src, tp_src, pol_src, sense_src, cond_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src, sem_src, cmd_src_keys, cmd_ctx_src, fp_src, tp_src, pol_src, sense_src, cond_src = next(dl_src_iter)
                xb_src = xb_src.to(device)
                yb_src = yb_src.to(device)
                cmd_ctx_src = cmd_ctx_src.to(device)
                fp_src = fp_src.to(device)
                tp_src = tp_src.to(device)
                pol_src = pol_src.to(device)
                sense_src = sense_src.to(device)
                cond_src = cond_src.to(device)
                xb_proc_src = _select_process_x(xb_src)
                zb_src = build_z_batch(
                    cts_src, device, design_dim,
                    z_dict=z_dict_src, graph_cache=graph_cache_src, enc=enc,
                    dedup=options.dedup_z,
                )
                pred_src, h_design_src, h_process_src = model(
                    xb_proc_src, zb_src, node="src",
                    from_pin_id=fp_src, to_pin_id=tp_src, pol_id=pol_src, sense_id=sense_src, cond_id=cond_src,
                )
                loss_src = th.nn.functional.mse_loss(pred_src, yb_src)
                loss_src_list.append(loss_src)

                if options.weight_clr > 0.0:
                    clr_feats = model.project_clr(th.cat([h_design_tgt, h_design_src], dim=0))
                    clr_labels = build_alignment_labels(
                        list(cts_tgt) + list(cts_src),
                        semantic_keys=list(sem_tgt) + list(sem_src),
                        mode=getattr(options, "clr_label_mode", "semantic_arc"),
                    )
                    clr_list.append(
                        supervised_contrastive_loss(
                            clr_feats, clr_labels, temp=options.con_temp,
                            device=device, normalization=options.norm_clr,
                        )
                    )
                if options.weight_cmd > 0.0:
                    cmd_tgt = model.project_cmd(h_process_tgt, cmd_ctx_tgt)
                    cmd_src = model.project_cmd(h_process_src, cmd_ctx_src)
                    cmd_labels_tgt = build_alignment_labels(
                        list(cts_tgt),
                        semantic_keys=list(cmd_tgt_keys),
                        mode=getattr(options, "cmd_label_mode", "semantic_arc"),
                    )
                    cmd_labels_src = build_alignment_labels(
                        list(cts_src),
                        semantic_keys=list(cmd_src_keys),
                        mode=getattr(options, "cmd_label_mode", "semantic_arc"),
                    )
                    cmd_list.append(conditional_cmd_loss(cmd_src, cmd_tgt, cmd_labels_src, cmd_labels_tgt, k=options.cmd_k))

            reg_loss = (
                loss_tgt * batch_size_tgt
                + options.loss_weight_45 * sum(loss_src_list) * batch_size_src
            ) / (batch_size_tgt + len(loss_src_list) * batch_size_src)

            clr_loss = th.stack(clr_list).mean() if clr_list else th.tensor(0.0, device=device)
            cmd_loss_v = th.stack(cmd_list).mean() if cmd_list else th.tensor(0.0, device=device)
            total_loss_batch = reg_loss + options.weight_clr * clr_loss + options.weight_cmd * cmd_loss_v

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            reg_loss_sum += reg_loss.item()
            clr_loss_sum += clr_loss.item()
            cmd_loss_sum += cmd_loss_v.item()
            batch_cnt += 1
            total_loss += total_loss_batch.item() * len(yb_tgt)
            total_n += len(yb_tgt)
            r2_score.update(pred_tgt, yb_tgt)

        train_loss = total_loss / max(1, total_n)
        train_r2 = r2_score.compute().item() if total_n > 0 else 0.0

        val_loss, val_r2 = validate_cell_proc(
            val_dl, enc, model, device, design_dim,
            z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, dedup=options.dedup_z,
        )
        if test_dl is not None:
            test_loss, test_r2 = validate_cell_proc(
                test_dl, enc, model, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, dedup=options.dedup_z,
            )
        else:
            test_loss, test_r2 = float("nan"), float("nan")

        avg_reg = reg_loss_sum / max(1, batch_cnt)
        avg_clr = clr_loss_sum / max(1, batch_cnt)
        avg_cmd = cmd_loss_sum / max(1, batch_cnt)
        print(
            f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, "
            f"test_loss:{test_loss:.4f}, test_r2:{test_r2:.3f}, "
            f"reg:{avg_reg:.4f}, clr:{avg_clr:.4f}, cmd:{avg_cmd:.4f}"
        )

        improved = val_r2 > (best_val + min_delta)
        if improved:
            best_val = val_r2
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_test_r2 = test_r2
            best_test_loss = test_loss
            stale_epochs = 0
            os.makedirs(options.model_saving_dir, exist_ok=True)
            th.save(
                {
                    "enc": enc.state_dict(),
                    "model": model.state_dict(),
                    "design_dim": design_dim,
                    "hgat_hid": hgat_hid,
                    "hgat_heads": hgat_heads,
                    "pin2id": pin2id,
                    "cond2id": cond2id,
                    "use_arc_cond": getattr(options, "use_arc_cond", False),
                    "pin_emb_dim": getattr(options, "pin_emb_dim", 8),
                    "pol_emb_dim": getattr(options, "pol_emb_dim", 2),
                    "sense_emb_dim": getattr(options, "sense_emb_dim", 4),
                    "cond_emb_dim": getattr(options, "cond_emb_dim", 8),
                    "arc_vocab_scope": arc_vocab_scope,
                    "arc_sep_domain_emb": getattr(options, "arc_sep_domain_emb", False),
                    "arc_cond_mode": getattr(options, "arc_cond_mode", "concat"),
                    "disable_src_arc_cond": getattr(options, "disable_src_arc_cond", False),
                    "clr_proj_dim": getattr(options, "clr_proj_dim", 64),
                    "cmd_proj_dim": getattr(options, "cmd_proj_dim", 64),
                    "clr_label_mode": getattr(options, "clr_label_mode", "semantic_arc"),
                    "cmd_label_mode": getattr(options, "cmd_label_mode", "semantic_arc"),
                    "process_numeric_cols": PROCESS_NUMERIC_COLS,
                    "scaler_stats": scaler_stats,
                    "y_scaler": y_scaler,
                    "epoch": epoch + 1,
                    "best_val_r2": best_val,
                    "script": "disentangle_cell_type_early_stop",
                },
                best_ckpt_path,
            )
            print("Model successfully saved")
        else:
            stale_epochs += 1
            if patience > 0:
                print(f"[Info] Early-stop counter: {stale_epochs}/{patience}")
                if stale_epochs >= patience:
                    print(f"[Early Stop] epoch:{epoch + 1}, best_epoch:{best_epoch}, best_val_r2:{best_val:.4f}")
                    break

    if best_epoch > 0:
        print(
            f"[Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, "
            f"val_loss:{best_val_loss:.6f}, "
            f"test_r2:{best_test_r2:.4f}, test_loss:{best_test_loss:.6f}, "
            f"ckpt:{best_ckpt_path}"
        )


if __name__ == "__main__":
    options = get_options()
    seed = options.seed
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    stdout_f = "{}/stdout.log".format(options.model_saving_dir)
    stderr_f = "{}/stderr.log".format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        run_train_and_report_test(train_balanced_sep_mlp_disentangle_cell_type_early_stop, options, seed, script_name=__file__)