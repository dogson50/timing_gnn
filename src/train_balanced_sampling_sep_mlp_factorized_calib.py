r"""Balanced sampling with factorized features + domain calibration.

Design goal:
1) Keep regression as the primary objective.
2) Use shared predictor + lightweight domain-specific calibration heads.
3) Use conditional (cell_type-wise) alignment after warmup only.
"""

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


def precompute_z_from_graph_cache(graph_cache, enc):
    z_dict = {}
    enc.eval()
    with th.no_grad():
        for ct, (g, feats) in graph_cache.items():
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[str(ct)] = z
    return z_dict


def conditional_mean_l2(feat_a, labels_a, feat_b, labels_b, device):
    if feat_a is None or feat_b is None or feat_a.numel() == 0 or feat_b.numel() == 0:
        return th.tensor(0.0, device=device)

    idx_a = {}
    for i, l in enumerate(labels_a):
        idx_a.setdefault(str(l), []).append(i)
    idx_b = {}
    for i, l in enumerate(labels_b):
        idx_b.setdefault(str(l), []).append(i)

    common = sorted(set(idx_a.keys()).intersection(set(idx_b.keys())))
    if not common:
        return th.tensor(0.0, device=device)

    losses = []
    for ct in common:
        ia = th.tensor(idx_a[ct], device=device, dtype=th.long)
        ib = th.tensor(idx_b[ct], device=device, dtype=th.long)
        ma = feat_a.index_select(0, ia).mean(dim=0)
        mb = feat_b.index_select(0, ib).mean(dim=0)
        losses.append((ma - mb).pow(2).mean())

    return th.stack(losses).mean() if losses else th.tensor(0.0, device=device)


def orthogonality_loss(h_design, h_process, eps=1e-8):
    if h_design is None or h_process is None or h_design.numel() == 0 or h_process.numel() == 0:
        return h_design.new_tensor(0.0) if h_design is not None else th.tensor(0.0)
    hd = F.normalize(h_design, dim=1, eps=eps)
    hp = F.normalize(h_process, dim=1, eps=eps)
    cos = (hd * hp).sum(dim=1)
    return (cos.pow(2)).mean()


class FactorizedCalibRegressor(nn.Module):
    def __init__(self, in_dim, design_dim, feat_dim=128, hid=256, dropout=0.0):
        super().__init__()
        self.mlp_design = self._make_mlp(design_dim, feat_dim, hid, dropout)
        self.mlp_process = self._make_mlp(in_dim, feat_dim, hid, dropout)
        self.shared_head = self._make_head(2 * feat_dim, hid, dropout)
        self.calib_tgt = self._make_calib(2 * feat_dim, feat_dim, dropout)
        self.calib_src = self._make_calib(2 * feat_dim, feat_dim, dropout)

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

    @staticmethod
    def _make_calib(in_dim, hid, dropout):
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
        pred_shared = self.shared_head(h).squeeze(-1)
        if node == "tgt":
            delta = self.calib_tgt(h).squeeze(-1)
        elif node == "src":
            delta = self.calib_src(h).squeeze(-1)
        else:
            raise ValueError(f"Unknown node type: {node}")
        pred = pred_shared + delta
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
            dedup=dedup,
        )
        pred, _, _ = model(xb, zb, node="tgt")
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        r2_score.update(pred, yb)
    avg_loss = total_loss / max(1, total_n)
    val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
    return avg_loss, val_r2


def train_balanced_sep_mlp_factorized_calib(options, seed):
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
    dropout = getattr(options, "mlp_dropout", 0.0)
    feat_dim = getattr(options, "node_feat_dim", 128)

    align_weight = float(getattr(options, "weight_cmd", 0.01))
    orth_weight = float(getattr(options, "weight_clr", 0.001))
    warmup_epochs = int(getattr(options, "pretrain_epochs", 0))
    if warmup_epochs <= 0:
        warmup_epochs = max(1, int(0.3 * options.num_epoch))

    print(f"[Info] align_weight={align_weight}, orth_weight={orth_weight}, warmup_epochs={warmup_epochs}")

    hgat_l2_norm = getattr(options, "hgat_l2_norm", False)
    in_map = get_hgat_in_dim_map()
    enc = HGATDesignEncoder(
        in_dim_map=in_map,
        hid=hgat_hid,
        out=design_dim,
        num_heads=hgat_heads,
        l2_norm=hgat_l2_norm,
    ).to(device)
    model = FactorizedCalibRegressor(
        in_dim=options.in_dim,
        design_dim=design_dim,
        feat_dim=feat_dim,
        hid=hgat_hid,
        dropout=dropout,
    ).to(device)

    if options.load_ckpt_path is not None:
        ckpt_path = options.load_ckpt_path
        if os.path.isdir(ckpt_path):
            cand = os.path.join(ckpt_path, "ckpt_best.pt")
            if os.path.exists(cand):
                ckpt_path = cand
        if os.path.exists(ckpt_path) and ckpt_path.endswith(".pt"):
            ckpt = th.load(ckpt_path, map_location=device)
            if "enc" in ckpt:
                enc.load_state_dict(ckpt["enc"], strict=False)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            print(f"[Info] Loaded ckpt from {ckpt_path}")
        elif options.load_ckpt_path is not None:
            print(f"[Warn] load_ckpt_path not found or unsupported: {ckpt_path}")

    print("----------------Loading HGAT graphs----------------")
    z_dict_src = None
    z_dict_tgt = None
    graph_cache_src = None
    graph_cache_tgt = None
    if options.freeze_hgat:
        print("[Info] freeze_hgat=True, precomputing z")
        graph_cache_src = build_src_graph_cache(data_dir, meta, device)
        graph_cache_tgt = build_tgt_graph_cache(data_dir, meta, device)
        z_dict_src = precompute_z_from_graph_cache(graph_cache_src, enc)
        z_dict_tgt = precompute_z_from_graph_cache(graph_cache_tgt, enc)
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
        align_loss_sum = 0.0
        orth_loss_sum = 0.0
        batch_cnt = 0

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt in dl_tgt:
            xb_tgt = xb_tgt.to(device)
            yb_tgt = yb_tgt.to(device)
            zb_tgt = build_z_batch(
                cts_tgt,
                device,
                design_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                enc=enc,
                dedup=options.dedup_z,
            )
            pred_tgt, h_design_tgt, h_process_tgt = model(xb_tgt, zb_tgt, node="tgt")
            loss_tgt = loss_fn(pred_tgt, yb_tgt)

            loss_src_list = []
            align_list = []
            orth_list = []

            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src = next(dl_src_iter)

                xb_src = xb_src.to(device)
                yb_src = yb_src.to(device)
                zb_src = build_z_batch(
                    cts_src,
                    device,
                    design_dim,
                    z_dict=z_dict_src,
                    graph_cache=graph_cache_src,
                    enc=enc,
                    dedup=options.dedup_z,
                )
                pred_src, h_design_src, h_process_src = model(xb_src, zb_src, node="src")
                loss_src_list.append(loss_fn(pred_src, yb_src))

                if epoch >= warmup_epochs and align_weight > 0.0:
                    align_list.append(
                        conditional_mean_l2(h_design_tgt, cts_tgt, h_design_src, cts_src, device)
                    )
                if epoch >= warmup_epochs and orth_weight > 0.0:
                    orth_tgt = orthogonality_loss(h_design_tgt, h_process_tgt)
                    orth_src = orthogonality_loss(h_design_src, h_process_src)
                    orth_list.append(0.5 * (orth_tgt + orth_src))

            reg_loss = (
                loss_tgt * batch_size_tgt
                + options.loss_weight_45 * sum(loss_src_list) * batch_size_src
            ) / (batch_size_tgt + len(loss_src_list) * batch_size_src)

            align_loss_v = th.stack(align_list).mean() if align_list else th.tensor(0.0, device=device)
            orth_loss_v = th.stack(orth_list).mean() if orth_list else th.tensor(0.0, device=device)

            total_loss_batch = reg_loss + align_weight * align_loss_v + orth_weight * orth_loss_v

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            reg_loss_sum += reg_loss.item()
            align_loss_sum += align_loss_v.item()
            orth_loss_sum += orth_loss_v.item()
            batch_cnt += 1
            total_loss += total_loss_batch.item() * len(yb_tgt)
            total_n += len(yb_tgt)
            r2_score.update(pred_tgt, yb_tgt)

        train_loss = total_loss / max(1, total_n)
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
        )

        avg_reg = reg_loss_sum / max(1, batch_cnt)
        avg_align = align_loss_sum / max(1, batch_cnt)
        avg_orth = orth_loss_sum / max(1, batch_cnt)
        print(
            f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, "
            f"reg:{avg_reg:.4f}, cond_align:{avg_align:.4f}, orth:{avg_orth:.4f}"
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
                    "script": "factorized_calib",
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

    stdout_f = "{}/stdout.log".format(options.model_saving_dir)
    stderr_f = "{}/stderr.log".format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)

    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        train_balanced_sep_mlp_factorized_calib(options, seed)
