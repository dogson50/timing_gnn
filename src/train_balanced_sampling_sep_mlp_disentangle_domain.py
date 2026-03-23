r"""Balanced sampling + disentangle (domain-label CLR, CMD on design) with HGAT embeddings."""

import os
import re
import pickle
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import R2Score
from torch.utils.data import Dataset as TorchDataset, DataLoader

from options import get_options
import tee

from hgat import HGATDesignEncoder, build_dgl_graph_from_devs, get_hgat_in_dim_map
from spi2graph import parse_transistors_spice, parse_top_subckt_pins

NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit",
]
TARGET_COL = "delay"


def _ensure_pol_bit(df):
    if "pol_bit" not in df.columns:
        if "pol" in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df["pol_bit"] = 0.0
    return df


def _ensure_numeric_cols(df):
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _norm_xy(df, x_mean, x_std, y_mean, y_std):
    df = _ensure_pol_bit(df.copy())
    df = _ensure_numeric_cols(df)
    x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
    x = (x_raw - x_mean) / x_std
    y_raw = df[TARGET_COL].astype(np.float32).values
    y = (y_raw - y_mean) / y_std
    cts = df["cell_type"].astype(str).values
    return x, y, cts


class CellDelayDataset(TorchDataset):
    def __init__(self, df, x_mean, x_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return th.from_numpy(self.x[i]), th.tensor(self.y[i]), self.cts[i]


def load_dataset_pkl(data_dir: str, pkl_name: str = "dataset.pkl"):
    pkl_path = os.path.join(data_dir, pkl_name)
    if not os.path.exists(pkl_path) and pkl_name != "dataset.pkl":
        pkl_path = os.path.join(data_dir, "dataset.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing dataset.pkl in: {data_dir}")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    lines = sp_text.splitlines(keepends=True)
    collecting = False
    buf = []
    patt_begin = re.compile(r"\s*\.subckt\s+%s\b" % re.escape(subckt_name), re.IGNORECASE)
    patt_end = re.compile(r"\s*\.ends\b", re.IGNORECASE)
    for line in lines:
        if not collecting:
            if patt_begin.search(line):
                collecting = True
                buf.append(line)
        else:
            buf.append(line)
            if patt_end.match(line):
                break
    return "".join(buf) if buf else ""


def build_src_graph_cache(data_dir, meta, device):
    mapping = meta.get("src_spi_by_cell", {})
    if not mapping:
        print("[Warn] meta has no src_spi_by_cell")
        return {}

    graph_cache = {}
    for ctype, sp_path in mapping.items():
        if not sp_path:
            continue
        if not os.path.exists(sp_path):
            cand = os.path.join(data_dir, sp_path)
            if os.path.exists(cand):
                sp_path = cand
            else:
                continue
        sp_text = open(sp_path, "r", encoding="utf-8", errors="ignore").read()
        devs = parse_transistors_spice(sp_text)
        _, pins = parse_top_subckt_pins(sp_text)
        if not devs:
            continue
        g, feats, _ = build_dgl_graph_from_devs(devs, pins)
        graph_cache[ctype] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    print(f"[Info] Cached source graphs: {len(graph_cache)}")
    return graph_cache


def build_tgt_graph_cache(data_dir, meta, device):
    mapping = meta.get("tgt_subckt_by_cell", {})
    if not mapping:
        print("[Warn] meta has no tgt_subckt_by_cell")
        return {}

    tgt_spice = meta.get("tgt_sp_file", "")
    if not tgt_spice:
        print("[Warn] meta has no tgt_sp_file")
        return {}

    if not os.path.exists(tgt_spice):
        cand = os.path.join(data_dir, tgt_spice)
        if os.path.exists(cand):
            tgt_spice = cand
        else:
            raise FileNotFoundError(f"Target SPICE not found: {tgt_spice}")

    sp_text = open(tgt_spice, "r", encoding="utf-8", errors="ignore").read()

    graph_cache = {}
    for ctype, sub_name in mapping.items():
        sub_txt = extract_subckt_text(sp_text, sub_name)
        if not sub_txt:
            continue
        devs = parse_transistors_spice(sub_txt)
        _, pins = parse_top_subckt_pins(sub_txt)
        if not devs:
            continue
        g, feats, _ = build_dgl_graph_from_devs(devs, pins)
        graph_cache[ctype] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    print(f"[Info] Cached target graphs: {len(graph_cache)}")
    return graph_cache


def build_z_batch(cts, device, design_dim, *, z_dict=None, graph_cache=None, enc=None, dedup=False):
    if z_dict is None and (graph_cache is None or enc is None):
        raise ValueError("build_z_batch requires z_dict or (graph_cache + enc).")

    def _get_z(ct):
        key = str(ct)
        if z_dict is not None:
            z = z_dict.get(key)
            if z is None:
                return th.zeros(1, design_dim, device=device)
            return z
        entry = graph_cache.get(key) if graph_cache is not None else None
        if entry is None:
            return th.zeros(1, design_dim, device=device)
        g, feats = entry
        z = enc(g, feats)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return z

    if dedup:
        seen = {}
        uniq = []
        idx_map = []
        for ct in cts:
            key = str(ct)
            idx = seen.get(key)
            if idx is None:
                idx = len(uniq)
                seen[key] = idx
                uniq.append(key)
            idx_map.append(idx)
        if not uniq:
            return th.zeros((0, design_dim), device=device)
        z_unique = th.cat([_get_z(ct) for ct in uniq], dim=0)
        index = th.tensor(idx_map, device=device, dtype=th.long)
        return z_unique.index_select(0, index)

    if len(cts) == 0:
        return th.zeros((0, design_dim), device=device)
    return th.cat([_get_z(ct) for ct in cts], dim=0)


def precompute_z_from_graph_cache(graph_cache, enc, device, design_dim):
    z_dict = {}
    enc.eval()
    with th.no_grad():
        for ct, (g, feats) in graph_cache.items():
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[str(ct)] = z
    return z_dict


def _labels_to_tensor(labels, device):
    if isinstance(labels, th.Tensor):
        return labels.to(device=device, dtype=th.long)
    if len(labels) == 0:
        return th.zeros((0,), device=device, dtype=th.long)
    if isinstance(labels[0], (int, np.integer)):
        return th.tensor(labels, device=device, dtype=th.long)
    uniq = {v: i for i, v in enumerate(sorted(set(str(x) for x in labels)))}
    return th.tensor([uniq[str(x)] for x in labels], device=device, dtype=th.long)


def supervised_contrastive_loss(feats, labels, temp=1.0, device=None, normalization=False):
    if feats is None or feats.numel() == 0:
        return th.tensor(0.0, device=device)
    if normalization:
        feats = F.normalize(feats, dim=1)
    labels_t = _labels_to_tensor(labels, device=device)
    n = feats.shape[0]
    if n <= 1:
        return th.tensor(0.0, device=device)
    sim = feats @ feats.T
    sim = sim / max(1e-8, float(temp))
    sim = sim - sim.max(dim=1, keepdim=True)[0]
    exp_sim = th.exp(sim) + 1e-5
    eye = th.eye(n, device=feats.device, dtype=th.bool)
    exp_sim = exp_sim.masked_fill(eye, 0.0)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12)
    # Clamp to avoid log(0) on masked diagonal entries, which can create NaNs via 0*inf.
    prob = (exp_sim / denom).clamp(min=1e-12)
    log_prob = -th.log(prob)
    mask = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1)) & (~eye)
    mask_f = mask.float()
    pos_per = mask_f.sum(dim=1)
    loss_per = (log_prob * mask_f).sum(dim=1) / pos_per.clamp(min=1.0)
    valid = (pos_per > 0).float()
    return (loss_per * valid).sum() / valid.sum().clamp(min=1.0)


def l2diff(x1, x2):
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return l2diff(ss1, ss2)


def cmd_loss(x1, x2, k=5):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        return th.tensor(0.0, device=x1.device if x1 is not None else None)
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    scms = [l2diff(mx1, mx2)]
    for i in range(k - 1):
        scms.append(moment_diff(sx1, sx2, i + 2))
    return sum(scms)


class DisentangleCellDelayRegressor(nn.Module):
    def __init__(self, in_dim, design_dim, node_feat_dim, hid=256, dropout=0.0):
        super().__init__()
        self.mlp_design = self._make_mlp(design_dim, node_feat_dim, hid, dropout)
        self.mlp_process = self._make_mlp(in_dim, node_feat_dim, hid, dropout)
        self.head_tgt = self._make_head(node_feat_dim * 2, hid, dropout)
        self.head_src = self._make_head(node_feat_dim * 2, hid, dropout)

    @staticmethod
    def _make_mlp(in_dim, out_dim, hid, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _make_head(in_dim, hid, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def encode(self, x, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h_design = self.mlp_design(z)
        h_process = self.mlp_process(x)
        h = th.cat([h_design, h_process], dim=1)
        return h, h_design, h_process

    def forward(self, x, z, node="tgt"):
        h, h_design, h_process = self.encode(x, z)
        if node == "tgt":
            pred = self.head_tgt(h).squeeze(-1)
        elif node == "src":
            pred = self.head_src(h).squeeze(-1)
        else:
            raise ValueError(f"Unknown node type: {node}")
        return pred, h_design, h_process


@th.no_grad()
def validate_cell(val_dl, enc, model, device, design_dim, *, z_dict=None, graph_cache=None, dedup=False):
    enc.eval()
    model.eval()
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)
    total_loss = 0.0
    total_n = 0
    for xb, yb, cts in val_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        zb = build_z_batch(
            cts, device, design_dim,
            z_dict=z_dict, graph_cache=graph_cache, enc=enc,
            dedup=dedup
        )
        pred, _, _ = model(xb, zb, node="tgt")
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        r2_score.update(pred, yb)
    avg_loss = total_loss / max(1, total_n)
    val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
    return avg_loss, val_r2


def train_balanced_sep_mlp_disentangle_domain(options, seed):
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

    ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std)
    ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std)
    val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std)

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    batch_size_tgt = max(1, options.batch_size // 2)
    batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

    dl_tgt = DataLoader(ds_tgt, batch_size=batch_size_tgt, shuffle=True, collate_fn=my_collate)
    dl_src = DataLoader(ds_src, batch_size=batch_size_src, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=options.batch_size, shuffle=False, collate_fn=my_collate)

    if getattr(options, "in_dim", len(NUMERIC_COLS)) != len(NUMERIC_COLS):
        raise ValueError(
            f"options.in_dim ({getattr(options, 'in_dim', None)}) must equal number of numeric features "
            f"({len(NUMERIC_COLS)})."
        )

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
        in_dim=options.in_dim,
        design_dim=design_dim,
        node_feat_dim=node_feat_dim,
        hid=hgat_hid,
        dropout=dropout,
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
                    model.load_state_dict(ckpt["model"])
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
        optimizer = th.optim.Adam(model.parameters(),
                                  lr=options.learning_rate, weight_decay=options.weight_decay)
    else:
        print("[Info] freeze_hgat=False, building graph cache")
        graph_cache_src = build_src_graph_cache(data_dir, meta, device)
        graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)
        optimizer = th.optim.Adam(list(enc.parameters()) + list(model.parameters()),
                                  lr=options.learning_rate, weight_decay=options.weight_decay)
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)

    print("----------------Start training---------------")
    best_val = float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_best.pt")

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
        for xb_tgt, yb_tgt, cts_tgt in dl_tgt:
            xb_tgt = xb_tgt.to(device)
            yb_tgt = yb_tgt.to(device)
            zb_tgt = build_z_batch(
                cts_tgt, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, enc=enc,
                dedup=options.dedup_z
            )
            pred_tgt, h_design_tgt, h_process_tgt = model(xb_tgt, zb_tgt, node="tgt")
            loss_tgt = loss_fn(pred_tgt, yb_tgt)

            loss_src_list = []
            clr_list = []
            cmd_list = []
            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src = next(dl_src_iter)
                xb_src = xb_src.to(device)
                yb_src = yb_src.to(device)
                zb_src = build_z_batch(
                    cts_src, device, design_dim,
                    z_dict=z_dict_src, graph_cache=graph_cache_src, enc=enc,
                    dedup=options.dedup_z
                )
                pred_src, h_design_src, h_process_src = model(xb_src, zb_src, node="src")
                loss_src = loss_fn(pred_src, yb_src)
                loss_src_list.append(loss_src)

                if options.weight_clr > 0.0:
                    clr_feats = th.cat([h_process_tgt, h_process_src], dim=0)
                    clr_labels = [0] * h_process_tgt.shape[0] + [1] * h_process_src.shape[0]
                    clr_list.append(
                        supervised_contrastive_loss(
                            clr_feats, clr_labels, temp=options.con_temp,
                            device=device, normalization=options.norm_clr
                        )
                    )
                if options.weight_cmd > 0.0:
                    cmd_list.append(cmd_loss(h_design_tgt, h_design_src, k=options.cmd_k))

            reg_loss = (
                loss_tgt * batch_size_tgt +
                options.loss_weight_45 * sum(loss_src_list) * batch_size_src
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

        val_loss, val_r2 = validate_cell(
            val_dl, enc, model, device, design_dim,
            z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, dedup=options.dedup_z
        )

        avg_reg = reg_loss_sum / max(1, batch_cnt)
        avg_clr = clr_loss_sum / max(1, batch_cnt)
        avg_cmd = cmd_loss_sum / max(1, batch_cnt)
        print(
            f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, "
            f"reg:{avg_reg:.4f}, clr:{avg_clr:.4f}, cmd:{avg_cmd:.4f}"
        )

        if val_r2 > best_val:
            best_val = val_r2
            best_epoch = epoch + 1
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
                    "epoch": epoch + 1,
                    "best_val_r2": best_val,
                },
                best_ckpt_path,
            )
            print("Model successfully saved")

    if best_epoch > 0:
        print(
            f"[Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, "
            f"val_loss:{best_val_loss:.6f}, ckpt:{best_ckpt_path}"
        )


if __name__ == "__main__":
    options = get_options()
    seed = options.seed
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        train_balanced_sep_mlp_disentangle_domain(options, seed)
