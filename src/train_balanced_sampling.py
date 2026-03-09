r"""
Balanced sampling training for cell delay regression using HGAT embeddings.
"""

import os
import re
import pickle
import random
import numpy as np
import torch as th
import torch.nn as nn
from torchmetrics import R2Score
from torch.utils.data import Dataset, DataLoader

from options import get_options
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
import tee

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


class CellDelayDataset(Dataset):
    def __init__(self, df, x_mean, x_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return th.from_numpy(self.x[i]), th.tensor(self.y[i]), self.cts[i]


class CellDelayRegressor(nn.Module):
    def __init__(self, in_dim, design_dim, hid=256, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + design_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, x, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = th.cat([x, z], dim=1)
        return self.mlp(h).squeeze(-1)


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

    print(f"[Info] Parsing Target SPICE: {tgt_spice}")
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
        graph_cache[str(ctype)] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    print(f"[Info] Cached target graphs: {len(graph_cache)}")
    return graph_cache


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
        graph_cache[str(ctype)] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    print(f"[Info] Cached source graphs: {len(graph_cache)}")
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


def precompute_z_from_tgt_spice(data_dir, meta, enc, device, design_dim):
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

    print(f"[Info] Parsing Target SPICE: {tgt_spice}")
    sp_text = open(tgt_spice, "r", encoding="utf-8", errors="ignore").read()

    z_dict = {}
    enc.eval()
    with th.no_grad():
        for ctype, sub_name in mapping.items():
            sub_txt = extract_subckt_text(sp_text, sub_name)
            if not sub_txt:
                continue
            devs = parse_transistors_spice(sub_txt)
            _, pins = parse_top_subckt_pins(sub_txt)
            if not devs:
                continue
            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[ctype] = z
    print(f"[Info] Pre-computed target Z for {len(z_dict)} cells.")
    return z_dict


def precompute_z_from_src_spice(data_dir, meta, enc, device, design_dim):
    mapping = meta.get("src_spi_by_cell", {})
    if not mapping:
        print("[Warn] meta has no src_spi_by_cell")
        return {}

    z_dict = {}
    enc.eval()
    with th.no_grad():
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
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[ctype] = z
    print(f"[Info] Pre-computed source Z for {len(z_dict)} cells.")
    return z_dict


def train_balanced_cell(options, seed):
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

    df_7 = df_tgt_train
    df_45 = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train

    ds_7 = CellDelayDataset(df_7, x_mean, x_std, y_mean, y_std)
    ds_45 = CellDelayDataset(df_45, x_mean, x_std, y_mean, y_std)
    val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std)

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    batch_size_7 = max(1, options.batch_size // 2)
    batch_size_45 = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

    dl_7 = DataLoader(ds_7, batch_size=batch_size_7, shuffle=True, collate_fn=my_collate)
    dl_45 = DataLoader(ds_45, batch_size=batch_size_45, shuffle=True, collate_fn=my_collate)
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
    dropout = getattr(options, "mlp_dropout", 0.0)

    in_map = {"NET": 4, "PMOS": 2, "NMOS": 2}
    enc = HGATDesignEncoder(
        in_dim_map=in_map,
        hid=hgat_hid,
        out=design_dim,
        num_heads=hgat_heads,
        num_layers=hgat_layers,
        dropout=hgat_dropout,
        use_net_readout=hgat_use_net_readout,
        type_attn_readout=hgat_type_attn_readout,
    ).to(device)
    model = CellDelayRegressor(in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout).to(device)

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

    print("----------------Loading HGAT embeddings----------------")
    z_dict_src = None
    z_dict_tgt = None
    graph_cache_src = None
    graph_cache_tgt = None
    if options.freeze_hgat:
        print("[Info] freeze_hgat=True, precomputing z")
        z_dict_src = precompute_z_from_src_spice(data_dir, meta, enc, device, design_dim)
        z_dict_tgt = precompute_z_from_tgt_spice(data_dir, meta, enc, device, design_dim)
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

        dl_45_iter = iter(dl_45)
        for xb_7, yb_7, cts_7 in dl_7:
            xb_7 = xb_7.to(device)
            yb_7 = yb_7.to(device)
            zb_7 = build_z_batch(
                cts_7, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, enc=enc,
                dedup=options.dedup_z
            )
            pred_7 = model(xb_7, zb_7)
            loss_7 = loss_fn(pred_7, yb_7)

            loss_45_list = []
            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_45, yb_45, cts_45 = next(dl_45_iter)
                except StopIteration:
                    dl_45_iter = iter(dl_45)
                    xb_45, yb_45, cts_45 = next(dl_45_iter)
                xb_45 = xb_45.to(device)
                yb_45 = yb_45.to(device)
                zb_45 = build_z_batch(
                    cts_45, device, design_dim,
                    z_dict=z_dict_src, graph_cache=graph_cache_src, enc=enc,
                    dedup=options.dedup_z
                )
                pred_45 = model(xb_45, zb_45)
                loss_45 = loss_fn(pred_45, yb_45)
                loss_45_list.append(loss_45)

            total_loss_batch = (
                loss_7 * batch_size_7 +
                options.loss_weight_45 * sum(loss_45_list) * batch_size_45
            ) / (batch_size_7 + len(loss_45_list) * batch_size_45)

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item() * len(yb_7)
            total_n += len(yb_7)
            r2_score.update(pred_7, yb_7)

        train_loss = total_loss / max(1, total_n)
        train_r2 = r2_score.compute().item() if total_n > 0 else 0.0

        # validation on target
        enc.eval()
        model.eval()
        val_loss = 0.0
        val_n = 0
        r2_score.reset()
        with th.no_grad():
            for xb, yb, cts in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                zb = build_z_batch(
                    cts, device, design_dim,
                    z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, enc=enc,
                    dedup=options.dedup_z
                )
                pred = model(xb, zb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * len(yb)
                val_n += len(yb)
                r2_score.update(pred, yb)
        val_loss = val_loss / max(1, val_n)
        val_r2 = r2_score.compute().item() if val_n > 0 else 0.0

        print(f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}")

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
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        train_balanced_cell(options, seed)

import torch as th
import random
import os
import numpy as np

from torchmetrics import R2Score

if False:  # legacy layout/path training code (disabled)
    from importlib.resources import path
    from lib2to3.pytree import Node
    from tkinter import N
    from tracemalloc import start
    from dataset import *
    from options import get_options
    from model import *
    from TimeConv import *
    from Unet import UNet
    import dgl
    import pickle

    from time import time
    from random import shuffle
    import itertools
    from MyDataloader import *
    import tee
    from torch.utils.data import DataLoader

    # with open('../rawdata/ctype2id.json', 'r') as f:
    #     ctype2id = json.load(f)
    #     num_ctypes = len(ctype2id)

    idx2design = {}

    device = th.device("cuda:" + str(get_options().gpu) if th.cuda.is_available() else "cpu")
    R2_score = R2Score().to(device)
    Loss = nn.CrossEntropyLoss() if get_options().task == 'cls' else nn.MSELoss()

    DATA_SAVE_PATH = {
        '7': '../datasets/asap7-designs-repro',
        '130': '../datasets/130-designs'
    }


    def init_model(options):
        r"""

        initialize the model

        :param options:
            some additional parameters
        :return:
            param: options
            gnn : initialized gnn
            cnn: initialized gnn
            mlp: initialized mlp
        """

        # initialize the GNN model
        print('Intializing models...')
        # print(options.no_cnn,options.no_gnn)
        assert not options.no_cnn or not options.no_gnn, 'GNN and CNN model can not be both None!'
        mlp_dim = 0
        if options.no_gnn:
            gnn = None
        else:
            gnn = PathConv(
                in_feats_dim=options.out_dim,
                out_feats_dim=options.out_dim,
                cell_feat_dim=options.cell_feat_dim,
                net_feat_dim=options.net_feat_dim,
                flag_attn=options.attn,
                num_heads=options.num_heads
            )
            mlp_dim += gnn.out_feats_dim
        # initialize the cnn model
        if options.no_cnn:
            cnn = None
            fcn = None
        else:
            cnn = UNet(options.pooling) if options.unet else LayoutNet(options.pooling)
            fcn = nn.Linear(options.map_size * options.map_size, options.cnn_outdim)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(fcn.weight, gain=gain)
            mlp_dim += options.cnn_outdim
        # initialze the MLP
        mlp = MLP2(
            in_dim=mlp_dim,
            # in_dim = options.out_dim,
            out_dim=options.nlabels,
            nlayers=options.n_fcn,
            dropout=options.mlp_dropout
        )

        # fcn=None
        model = PathModel(gnn, fcn, mlp)
        print("creating model in:", options.model_saving_dir)
        print('Path Model', model)
        print('cnn:', cnn)
        # save the model and create a file to save the results
        if os.path.exists(os.path.join(options.model_saving_dir, 'model.pkl')) is False:
            with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                parameters = options
                pickle.dump((parameters, model, cnn), f)
            with open(os.path.join(options.model_saving_dir, 'res.txt'), 'w') as f:
                pass
        print('Model Initializing is accomplished!')

        return parameters, model, cnn


    def load_model(device, options):
        r"""
        Load the model

        :param device:
            the target device that the model is loaded on
        :param options:
            some additional parameters
        :return:
            param: new options
            gnn : loaded gnn
            cnn: loaded gnn
            mlp: loaded mlp
        """
        print('----------------Loading the model and hyper-parameters----------------')
        model_dir = options.model_saving_dir
        # if there is no model in the target directory, break
        if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
            param, model, cnn = init_model(options)
        else:
            # read the pkl file that saves the hype-parameters and the model.
            with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
                # param: hyper-parameters, e.g., learning rate;
                # load the model from pickle file
                param, model, cnn = pickle.load(f)
                param.model_saving_dir = options.model_saving_dir
                # make some changes to the options
                if options.change_lr:
                    param.learning_rate = options.learning_rate
                if options.change_alpha:
                    param.alpha = options.alpha
        print(f'Model successfully initialized!')
        # model = nn.DataParallel(model, device_ids=[0,1])
        # cnn = nn.DataParallel(cnn, device_ids=[0,1])
        if options.load_ckpt_path is not None:
            with open(os.path.join(options.load_ckpt_path, 'model.pkl'), 'rb') as f:
                _, model, cnn = pickle.load(f)
            print(f'Model and hyper-parameters successfully loaded from {options.load_ckpt_path}')
        model = model.to(device)
        if cnn is not None:
            cnn = cnn.to(device)
        return param, model, cnn


    def validate(loader, device, model, cnn, beta, options):
        r"""

        validate the model

        :param loader:
            the data loader to load the validation dataset
        :param device:
            device
        :param model:
            trained model
        :param mlp:
            trained mlp
        :param Loss:
            used loss function
        :param beta:
            a hyperparameter that determines the thredshold of binary classification
        :param options:
            some parameters
        :return:
            result of the validation: loss, acc,recall,precision,F1_score
        """

        overall_loss, overall_acc, overall_recall, overall_precision, overall_f1, overall_r2 = 0, 0.0, 0, 0, 0, 0
        # runtime = 0
        res = []
        with th.no_grad():
            # load validation data, one batch at a time
            # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
            # and output neighborhoods (out_block).
            # The dst_nodes of the last block of in_block/out_block is the central nodes.
            print('validate:')
            case_idx = 0
            for path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks in loader:

                total_num, total_loss, correct, fn, fp, tn, tp, total_r2 = 0, 0.0, 0, 0, 0, 0, 0, 0
                runtime = 0
                # optim.zero_grad()
                start_time = time()
                cnn_inputs = cnn_inputs.reshape((1, cnn_inputs.shape[0], cnn_inputs.shape[1], cnn_inputs.shape[2]))
                feat_map = cnn(cnn_inputs.to(device)).reshape((1, -1)) if cnn is not None else None
                # path_mask = path_mask.to_sparse()
                # transfer the data to GPU

                graph = graph.to(device)
                count_target = 0
                label_hats = None
                target_list = []
                # first_level_nodes = topo_levels[0][0]
                # init_message = graph.ndata['h'][first_level_nodes]
                path_loader = DataLoader(path_dataset, batch_size=len(path_dataset.paths), shuffle=False)
                for path_ids in path_loader:
                    path_ids = list(set(path_ids.numpy().tolist()))
                    sampled_ends, sampled_paths = {}, {}
                    for i, pathid in enumerate(path_ids):
                        level = path2level[pathid]
                        endpoint = path2endpoint[pathid]
                        # assert endpoints2path[endpoint] == pathid
                        sampled_ends[level] = sampled_ends.get(level, [])
                        sampled_ends[level].append(endpoint)
                        sampled_paths[level] = sampled_paths.get(level, [])
                        sampled_paths[level].append(pathid)

                    for level_id, level in enumerate(topo_levels):
                        nodes, eids = level[:2]
                        if len(level) == 4:
                            eids = eids.to(device)
                        targets = sampled_ends.get(level_id, [])
                        paths = sampled_paths.get(level_id, [])
                        target_list.extend(targets)
                        count_target += len(target_list)

                        if options.no_cnn or len(paths) == 0:
                            path_map = None
                        else:
                            path_mask = th.index_select(path_masks, 0, th.tensor(paths)).to(device)
                            path_map = path_mask.to_dense() * feat_map
                            # path_map = path_map.view(-1,path_map.shape[1]*path_map.shape[2])

                        cur_label_hats = model(graph, nodes, eids, targets, level_id, path_map)

                        if len(paths) == 0:
                            continue

                        if label_hats is None:
                            label_hats = cur_label_hats
                        else:
                            label_hats = th.cat((label_hats, cur_label_hats), dim=0)

                # labels: the ground-truth binary labels, decide whether path is critical or not
                # predict_labels: the predicted binary labels
                # label_hats: the output of the model, len=2 for classification, len=1 for regression
                labels = graph.ndata['label'][target_list].squeeze()
                if options.task == 'cls':
                    predict_labels = th.argmax(nn.functional.softmax(label_hats, 1), dim=1)
                    test_loss = Loss(label_hats, labels)
                    test_r2 = 0
                elif options.task == 'reg':
                    required_time = graph.ndata['required_time'][target_list].squeeze()
                    arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
                    test_loss = Loss(label_hats, arrival_time)
                    predict_labels = judge_critical(label_hats, required_time).to(device)
                    test_r2 = R2_score(label_hats, arrival_time).to(device)
                    total_r2 += test_r2.item()
                    # print('R2 score: {}'.format(test_r2))
                # calculate loss

                # print(cnn.down3.maxpool_conv.weights)
                total_num += len(labels)
                total_loss += test_loss.item()

                correct += (
                        predict_labels == labels
                ).sum().item()
                # calculate fake negative, true positive, fake negative, and true negative rate
                fn += ((predict_labels == 0) & (labels != 0)).sum().item()
                tp += ((predict_labels != 0) & (labels != 0)).sum().item()
                tn += ((predict_labels == 0) & (labels == 0)).sum().item()
                fp += ((predict_labels != 0) & (labels == 0)).sum().item()

                acc = correct / total_num
                recall = 0
                precision = 0
                if tp != 0:
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp)
                F1_score = 0
                if precision != 0 or recall != 0:
                    F1_score = 2 * recall * precision / (recall + precision)

                overall_loss += total_loss
                overall_r2 += total_r2
                overall_recall += recall
                overall_f1 += F1_score
                overall_acc += acc
                overall_precision += precision

                # overall_correct += correct
                # overall_fn  += fn
                # overall_fp += fp
                # overall_tn += tn
                # overall_tp += tp
                # overall_num += total_num
                # print('case',case_idx)
                # case_idx += 1
                # print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
                print("\tcase {} \tl:{:.3f}, r2:{:.3f}, rc:{:.3f}, F1:{:.3f}".format(case_idx, test_loss, test_r2, recall,
                                                                                     F1_score))
                case_idx += 1
                res.append([test_loss, test_r2, acc, recall, precision, F1_score])
        # calculate the overall loss / accuracy
        num_case = case_idx
        overall_loss = overall_loss / num_case
        overall_acc = overall_acc / num_case
        overall_r2 = overall_r2 / num_case
        overall_f1 = overall_f1 / num_case
        overall_recall = overall_recall / num_case
        overall_precision = overall_precision / num_case
        # calculate overall recall, precision and F1-score

        # print('overall val')
        # print("\ttp:", overall_tp, " fp:", overall_fp, " fn:", overall_fn, " tn:", overall_tn, " precision:", round(overall_precision, 3))
        print("\toverall r2:{:.3f}, rc:{:.3f}, F1:{:.3f}".format(overall_r2, overall_recall, overall_f1))

        return res, overall_f1, overall_r2


    def split_dataset(paths, critical_paths):
        non_critical_paths = list(set(paths) - set(critical_paths))
        shuffle(critical_paths)
        val_paths = critical_paths[:int(len(critical_paths) / 5)]
        test_paths = critical_paths[int(len(critical_paths) / 5):]
        shuffle(non_critical_paths)
        val_paths.extend(non_critical_paths[:int(len(non_critical_paths) / 5)])
        test_paths.extend(non_critical_paths[int(len(non_critical_paths) / 5):])

        return val_paths, test_paths


    # def transform_cellfeat(cell_feat):
    #     new_cellfeat = th.zeros((cell_feat.shape[0], cell_feat.shape[1] - num_ctypes + 4))
    #     cell_type = cell_feat[:, :num_ctypes]
    #     cell_type = th.argmax(cell_type, dim=1, keepdim=True)
    #     id2celltype = {}
    #     for cell, id in ctype2id.items():
    #         id2celltype[id] = cell
    #
    #     # new celltype feat:
    #     #   buf: 1000
    #     #   inv: 0100
    #     #   register: 0010
    #     #   others: 0001
    #     {"AND": 0, "FA": 1, "HA": 2, "MAJI": 3, "MAJ": 4, "NAND": 5, "NOR": 6, "OR": 7, "TIEHI": 8,
    #      "TIELO": 9, "XNOR": 10, "XOR": 11, "ASYNC_DFFH": 12, "DFFHQN": 13, "DFFHQ": 14, "DFFLQN": 15,
    #      "DFFLQ": 16, "DHL": 17, "DLL": 18, "ICG": 19, "SDFH": 20, "SDFL": 21, "BUF": 22, "CKINVDC": 23,
    #      "HB": 24, "INV": 25, "O2A1O1I": 26, "OA": 27, "OAI": 28, "A2O1A1I": 29, "A2O1A1O1I": 30, "AO": 31, "AOI": 32}
    #     for i in range(cell_feat.shape[0]):
    #         type = id2celltype[cell_type[i].item()]
    #         new_cellfeat[i][0] = type == 'BUFF'
    #         new_cellfeat[i][1] = type in ('INV', 'CKINVDC')
    #         new_cellfeat[i][2] = type in ('TIEHI', "TIELO", "ASYNC_DFFH", "DFFHQN", "DFFHQ",
    #                                       "DFFLQN", "DFFLQ", "DHL", "DLL", "ICG", "SDFH",
    #                                       "SDFL", "HB")
    #         new_cellfeat[i][3] = not (new_cellfeat[i][0] and new_cellfeat[i][1] and new_cellfeat[i][2])
    #         new_cellfeat[i][4:] = cell_feat[i][num_ctypes:]
    #         print(type, new_cellfeat[i])
    # new_cellfeat[:,0:1] = cell_type == ctype2id['BUF']
    # new_cellfeat[:,1:2] = cell_type == ctype2id['INV']
    # new_cellfeat[:,2:3] = ( cell_type != ctype2id['INV'] and cell_type!=ctype2id['BUF'])
    # new_cellfeat[:,3:] = cell_feat[:,num_ctypes:]


    def minMax_scalar(a):
        min_a = th.min(a)
        max_a = th.max(a)
        return (a - min_a) / (max_a - min_a)


    def norm(feature, start_idx):
        num_feat = feature.shape[1]
        for i in range(start_idx, num_feat):
            feature[:, i: i + 1] = minMax_scalar(feature[:, i]).reshape(-1, 1)
        return feature


    def load_data(data_path, usage, init_feat_dim, os_rate, feat_reduce, if_norm, design_set=None, node='7',
                  time_unit_trans=None):
        """

        load the data

        :param dataset_file: str
                a pickle file that saves the dataset
        :param init_feat_dim: int
                the dimension of the initial feature
        :return:
            updated_dataset: List[(graph, topo_levels)]
                where graph is the DAG representation of a circuit,
                and topo_levels are the calculated topological levels
        """
        assert usage in ['train', 'test'], "Wrong data usage! Should be either 'train' or 'test'."
        if design_set is None:
            design_list_file = os.path.join(data_path, '{}data_list.txt'.format(usage))
            assert os.path.exists(design_list_file), \
                "Can not find the traindata list txt '{}'".format(design_list_file)

            with open(design_list_file, 'r') as f:
                lines = f.readlines()
                design_list = [l.replace('\n', '') for l in lines]
        else:
            design_list = design_set.split(',')
        print('--- {} designs: '.format(usage), design_list)
        datasets = []
        for i, design in enumerate(design_list):
            dataset_file = os.path.join(data_path, '{}.pkl'.format(design))
            graph, topo_levels, path_masks, path2level, path2endpoint, critical_paths, cnn_inputs = th.load(dataset_file)
            # with open(dataset_file,'rb') as f:
            # graph, topo_levels, path_masks,path2level,path2endpoint,critical_paths,cnn_inputs = pickle.load(f)
            print(path_masks.shape, graph.ndata['cell_feat'].shape)

            # relocate the cell feature for merge 130nm and 7nm feature
            original_cell_feat = graph.nodes['pin'].data['cell_feat']
            print(f'original cell feat for node {node}: {original_cell_feat.shape}')
            n_nodes, _ = original_cell_feat.shape
            new_shape = (n_nodes, 137)  # 34+95+8
            new_cell_feat = th.zeros(new_shape, dtype=original_cell_feat.dtype, device=original_cell_feat.device)
            if node == '7':
                # put the 7nm cell type feat first
                new_cell_feat[:, :34] = original_cell_feat[:, :34]
                # print(f'pin info feat: {original_cell_feat[:, -8:].shape}')
                new_cell_feat[:, -8:] = original_cell_feat[:, -8:]
            elif node == '130':
                # put the 130nm cell type feat after the 7nm feat
                cell_type_feat_130 = original_cell_feat[:, :-8]
                # print(f'130 cell type feat shape: {cell_type_feat_130.shape}')
                # print(new_cell_feat[:, 34:-8].shape)
                new_cell_feat[:, 34:-8] = original_cell_feat[:, :-8]
                # print(f'pin info feat: {original_cell_feat[:, -8:].shape}')
                new_cell_feat[:, -8:] = original_cell_feat[:, -8:]
            else:
                raise ValueError
            graph.nodes['pin'].data['cell_feat'] = new_cell_feat
            # shape_ = graph.ndata['cell_feat'].shape
            # print(f'After feat transform, the new cell feat {shape_}')
            # exit()

            # change the unit of the arrival time (ps <-> ns)
            if time_unit_trans is not None:
                end = graph.nodes['pin'].data['end']
                indices = th.where(end == 1)[0]
                arrival_time = graph.nodes['pin'].data['arrival_time']
                print(f'arrival time shape: {arrival_time.shape} | elements: {arrival_time[indices]}')
                if time_unit_trans == '130_to_7':
                    "ns -> ps"
                    new_arrival_time = arrival_time * 1000.
                    print(f'new arrival time: {new_arrival_time[indices]}')
                elif time_unit_trans == '7_to_130':
                    "ps -> ns"
                    new_arrival_time = arrival_time / 1000.
                    print(f'new arrival time: {new_arrival_time[indices]}')
                elif time_unit_trans == 'div_10':
                    new_arrival_time = arrival_time / 10.
                    print(f'new arrival time: {new_arrival_time[indices]}')
                elif time_unit_trans == 'div_100':
                    new_arrival_time = arrival_time / 100.
                    print(f'new arrival time: {new_arrival_time[indices]}')
                else:
                    raise ValueError
                graph.nodes['pin'].data['arrival_time'] = new_arrival_time
                arr_time = graph.ndata['arrival_time']
                print(f'after mapping: the new arrival time is: {arr_time[indices]}')
            graph.ndata['h'] = th.zeros((graph.number_of_nodes(), init_feat_dim), dtype=th.float)
            graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float)
            # import pdb
            # pdb.set_trace()
            # net_feat = graph.ndata['net_feat']
            # print(f'Before reduce: {net_feat.shape}')
            if feat_reduce is not None:
                if feat_reduce[1] != 0:
                    graph.ndata['net_feat'] = graph.ndata['net_feat'][:, :-feat_reduce[1]]
                if feat_reduce[0] != 0:
                    graph.ndata['cell_feat'] = graph.ndata['cell_feat'][:, :-feat_reduce[0]]
                # print(graph.ndata['cell_feat'][:5])
            # net_feat = graph.ndata['net_feat']
            # print(f'After reduce: {net_feat.shape}')
            # exit()
            # normalize all the features, so that the value of different feature will not differ greatly
            # graph.ndata['cell_feat'] = transform_cellfeat(graph.ndata['cell_feat'])
            if if_norm:
                graph.ndata['cell_feat'] = norm(graph.ndata['cell_feat'], num_ctypes)
                graph.ndata['net_feat'] = norm(graph.ndata['net_feat'], num_ctypes)

            if type(cnn_inputs) == np.ndarray:
                cnn_inputs = th.from_numpy(cnn_inputs).float()
            # cnn_inputs = th.unsqueeze(cnn_inputs,dim=0)
            paths = list(range(len(graph.ndata['end'][graph.ndata['end'].squeeze() == 1])))
            # non_critical_paths = list(set(paths)-set(critical_paths))
            num_neg = len(paths) - len(critical_paths)
            num_pos = len(critical_paths)
            ratio = num_neg / num_pos - 1

            if usage == 'test':

                split_file = os.path.join(data_path, '{}_split.pkl'.format(design))
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        val_paths, test_paths = pickle.load(f)
                else:
                    val_paths, test_paths = split_dataset(paths, critical_paths)
                    with open(split_file, 'wb') as f:
                        pickle.dump((val_paths, test_paths), f)
                paths = val_paths
                print(len(paths), len(critical_paths))

            if usage == 'train':
                idx2design[i] = design
            if usage == 'train' and os_rate != 0 and ratio > 1:
                # shuffle(critical_paths)
                for _ in range(os_rate):
                    paths.extend(critical_paths)
                # while ratio>=1:
                #     paths.extend(critical_paths)
                #     ratio -= 1
                # shuffle(critical_paths)
                # paths.extend(critical_paths[:int(ratio*num_pos)])
            path_dataset = PathDataset(paths)

            datasets.append(
                (path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks)
            )

        return datasets


    def judge_critical(pred_arr_time, required_time):
        pred_slack = required_time - pred_arr_time
        is_critical = th.ones(pred_arr_time.shape)
        is_critical[pred_slack >= 0] = 0
        return is_critical


    def sample_paths(path_ids, path2level, path2endpoint):
        # sample level2path mapping
        sampled_ends, sampled_paths = {}, {}
        for _, pathid in enumerate(path_ids):
            level = path2level[pathid]
            endpoint = path2endpoint[pathid]
            # assert endpoints2path[endpoint] == pathid
            sampled_ends[level] = sampled_ends.get(level, [])
            sampled_ends[level].append(endpoint)
            sampled_paths[level] = sampled_paths.get(level, [])
            sampled_paths[level].append(pathid)
        return sampled_ends, sampled_paths


    def set_path_loader(graph, path_dataset, path2level, batch_size, device):
        # prepare the path dataloader, feat map and graph
        graph = graph.to(device)
        # print(graph)
        if len(path2level) <= batch_size:
            path_loader = DataLoader(path_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            path_loader = DataLoader(path_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        num_batch = len(path_loader)
        return graph, path_loader, num_batch


    def path_batch_forward(path_ids, path2level, path2endpoint, topo_levels, feat_map, model, graph, path_masks):
        # forward of a batch of paths in the netlist
        path_ids = list(path_ids.numpy().tolist())
        sampled_ends, sampled_paths = sample_paths(path_ids, path2level, path2endpoint)
        count_target = 0
        label_hats = None
        target_list = []
        for level_id, level in enumerate(topo_levels):
            nodes, eids = level[:2]
            if len(level) == 2:
                eids = eids.to(device)
            targets = sampled_ends.get(level_id, [])
            paths = sampled_paths.get(level_id, [])
            # print(level_id,len(targets))
            target_list.extend(targets)
            count_target += len(target_list)
            # print(level_id,len(nodes))
            if options.no_cnn or len(paths) == 0:
                path_map = None
            else:
                # path_mask = path_masks[paths].to(device)
                path_mask = th.index_select(path_masks, 0, th.tensor(paths)).to(device)
                path_map = path_mask.to_dense() * feat_map
                # path_map = path_map.view(-1,path_map.shape[1]*path_map.shape[2])
            # print(path_map.shape)
            # path_map = None
            # print(level_id,path_map.shape)
            cur_label_hats = model(graph, nodes, eids, targets, level_id, path_map)

            if len(paths) == 0:
                continue

            if label_hats is None:
                label_hats = cur_label_hats
            else:
                label_hats = th.cat((label_hats, cur_label_hats), dim=0)
        # labels = graph.ndata['label'][target_list].squeeze()
        arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
        return label_hats, arrival_time


    def get_130_design(bid, len_130_designs, sample_num):
        # sample the design indices given the batch id
        start_idx = bid * sample_num
        sampled_idx = []
        for i in range(start_idx, start_idx + sample_num):
            sampled_idx.append(i % len_130_designs)
        return sampled_idx


    def reset_graph_feature(graph, device):
        graph.ndata['h'] = th.zeros((graph.number_of_nodes(), options.out_dim), dtype=th.float).to(device)
        graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(
            device)
        graph.edges['cell'].data['e'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(
            device)


    def train(options, seed):
        th.multiprocessing.set_sharing_strategy('file_system')
        device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

        # you can define your dataset file here
        # data_save_path = options.data_save_path
        train_data_save_path = DATA_SAVE_PATH[options.train_node]
        test_data_save_path = DATA_SAVE_PATH[options.test_node]
        print(f'train data save path: {train_data_save_path}')
        print(f'test data save path: {test_data_save_path}')

        # load the model
        options.cell_feat_dim -= options.feat_reduce[0]
        options.net_feat_dim -= options.feat_reduce[1]
        options, model, cnn = load_model(device, options)
        with open(os.path.join(options.model_saving_dir, 'seed.txt'), 'a') as f:
            f.write(str(seed))

        print('Hyperparameters are listed as follows:')
        print(options)
        print('seed:', seed)
        print('The model architecture is shown as follow:')
        print(model)
        print(cnn)
        print("----------------Loading data----------------")
        # train_data_file = os.path.join(data_save_path, 'train.pkl')

        train_dataset_7 = load_data('../datasets/asap7-designs-repro', 'train', options.out_dim, options.os_rate,
                                    options.feat_reduce,
                                    options.norm, design_set='smallboom', node='7')
        train_dataset_130 = load_data('../datasets/130-designs', 'train', options.out_dim, options.os_rate,
                                      options.feat_reduce,
                                      options.norm, design_set=options.training_set, node='130')

        # val_data_file = os.path.join(data_save_path, 'test.pkl')
        val_dataset = load_data(test_data_save_path, 'test', options.out_dim, options.os_rate, options.feat_reduce,
                                options.norm, design_set=options.test_set, node=options.test_node,
                                time_unit_trans=options.time_unit_trans)

        # split the validation set and test set
        beta = options.beta

        # set the optimizer
        print(f'Setting the optimizer and weights to update')
        print(f'GNN params')
        for n, p in model.named_parameters():
            print(n)
        print(f'CNN params')
        for n, p in cnn.named_parameters():
            print(n)
        if cnn is not None:
            if options.ft_mode is None or options.ft_mode == "all":
                # optim = th.optim.Adam(
                #     itertools.chain(model.parameters(), cnn.parameters()),
                #     # model.parameters(),
                #     options.learning_rate, weight_decay=options.weight_decay
                # )
                print(f'Optimiza all the parameters!')
                pass
            elif options.ft_mode == 'mlp_gnn_cell':
                # freeze the gnn net, gnn neigh and cnn
                print(f'Optimize the mlp and gnn-cell only!')
                freeze_modules = [model.gnn[0].fc_cell_neigh, model.gnn[0].fc_net_self, model.fcn, cnn]
                for m in freeze_modules:
                    for param in m.parameters():
                        param.requires_grad = False

            elif options.ft_mode == 'mlp_gnn_cell_fcn':
                print(f'optimize the mlp, gnn cell and fcn.')
                freeze_modules = [model.gnn[0].fc_cell_neigh, model.gnn[0].fc_net_self, cnn]
                for m in freeze_modules:
                    for param in m.parameters():
                        param.requires_grad = False
            else:
                raise ValueError("Wrong ft mode!")

            optim = th.optim.Adam(
                itertools.chain(filter(lambda p: p.requires_grad, model.parameters()),
                                filter(lambda p: p.requires_grad, cnn.parameters())),
                options.learning_rate, weight_decay=options.weight_decay
            )
            cnn.train()
            model.train()
        else:
            optim = th.optim.Adam(
                model.parameters(),
                options.learning_rate, weight_decay=options.weight_decay
            )
            model.train()
        # check the gradients setting
        # print(f'gnn cell req grad: {model.gnn[0].fc_cell_self.layers[0].weight.requires_grad}')
        # print(f'gnn net req grad: {model.gnn[0].fc_net_self.layers[0].weight.requires_grad}')
        # print(f'gnn neigh req grad: {model.gnn[0].fc_cell_neigh.layers[2].weight.requires_grad}')
        # print(f'model fcn req grad: {model.fcn[0].weight.requires_grad}')
        # print(f'model mlp req grad: {model.mlp[0].layers[0].weight.requires_grad}')
        # print(f'cnn req grad {cnn.encode[0].weight.requires_grad}')
        print("----------------Start training---------------")
        pre_loss = 100
        stop_score = 0
        max_F1_score, max_r2 = 0, 0
        print(f'index2design: {idx2design}')
        th.autograd.set_detect_anomaly(True)
        len_130_dataset = len(train_dataset_130)
        dataset_130_index = 0
        batch_size_7 = options.batch_size // 2
        batch_size_130 = options.batch_size // (2 * options.sample_130_num)
        print(
            f'For each batch we sample {batch_size_7} 7nm data and {batch_size_130} 130nm data from {options.sample_130_num} designs')
        for epoch in range(options.num_epoch):
            # assume only one 7nm design is loaded
            path_dataset_7, graph_7, path2level_7, path2endpoint_7, topo_levels_7, cnn_inputs_7, path_masks_7 = \
                train_dataset_7[0]
            graph_7, path_loader_7, num_batch_7 = set_path_loader(graph_7, path_dataset_7, path2level_7, batch_size_7,
                                                                  device)
            # initialize the 130nm dataloader
            feat_map_130_all, graph_130_all, path_loader_130_all, path_loader_iter_130_all = [], [], [], []
            path2level_130_all, path2endpoint_130_all, topo_levels_130_all, cnn_inputs_130_all, path_masks_130_all = [], [], [], [], []
            for i, (path_dataset_130, graph_130, path2level_130, path2endpoint_130, topo_levels_130, cnn_inputs_130,
                    path_masks_130) in enumerate(train_dataset_130):
                graph_130, path_loader_130, num_batch_130 = set_path_loader(graph_130, path_dataset_130, path2level_130,
                                                                            batch_size_130, device)
                graph_130_all.append(graph_130)
                path_loader_130_all.append(path_loader_130)
                path_loader_iter_130_all.append(iter(path_loader_130))
                path2level_130_all.append(path2level_130)
                path2endpoint_130_all.append(path2endpoint_130)
                topo_levels_130_all.append(topo_levels_130)
                cnn_inputs_130_all.append(cnn_inputs_130)
                path_masks_130_all.append(path_masks_130)

            print(f'{len(graph_130_all)} 130nm data loaded!')

            '''
            For each batch, sample a sub-batch from the 7nm dataset and then one (or more) sub-batches 
            from the 130nm dataset.
            '''
            path_loader_7_iter = iter(path_loader_7)
            for bidx in range(num_batch_7):
                start_time = time()
                # extract the image feat
                feat_map_7 = cnn(cnn_inputs_7.to(device)).reshape((1, -1)) if cnn is not None else None
                feat_map_130_all = [cnn(cnn_input.to(device)).reshape((1, -1)) if cnn is not None else None for cnn_input in
                                    cnn_inputs_130_all]
                path_ids_7 = next(path_loader_7_iter)
                label_hats_7, arrival_time_7 = path_batch_forward(path_ids_7, path2level_7, path2endpoint_7, topo_levels_7,
                                                                  feat_map_7, model, graph_7, path_masks_7)

                # sample from 130nm data
                label_hats_130, arrival_time_130 = [], []
                sampled_130_idx = get_130_design(bidx, len(feat_map_130_all), options.sample_130_num)
                for i in sampled_130_idx:
                    feat_map_130_ = feat_map_130_all[i]
                    graph_130_ = graph_130_all[i]
                    path_loader_130_ = path_loader_130_all[i]
                    path_loader_iter_130_ = path_loader_iter_130_all[i]
                    path2level_130_ = path2level_130_all[i]
                    path2endpoint_130_ = path2endpoint_130_all[i]
                    topo_levels_130_ = topo_levels_130_all[i]
                    path_masks_130_ = path_masks_130_all[i]
                    # if running out the dataloader, reinit the dataloader
                    try:
                        path_ids_130_ = next(path_loader_iter_130_)
                    except StopIteration:
                        path_loader_iter_130_ = iter(path_loader_130_)
                        path_loader_iter_130_all[i] = path_loader_iter_130_
                        path_ids_130_ = next(path_loader_iter_130_)
                    label_hats_130_, arrival_time_130_ = path_batch_forward(path_ids_130_, path2level_130_,
                                                                            path2endpoint_130_,
                                                                            topo_levels_130_,
                                                                            feat_map_130_, model,
                                                                            graph_130_,
                                                                            path_masks_130_)
                    label_hats_130.append(label_hats_130_)
                    arrival_time_130.append(arrival_time_130_)

                # calculate the reg loss
                train_loss_7 = Loss(label_hats_7, arrival_time_7)
                train_loss_130_all = [Loss(label_hats_130_, arrival_time_130_) for (label_hats_130_, arrival_time_130_) in
                                      zip(label_hats_130, arrival_time_130)]
                total_loss = (train_loss_7 * batch_size_7 + options.loss_weight_130 * sum(
                    train_loss_130_all) * batch_size_130) / (batch_size_7 + len(train_loss_130_all) * batch_size_130)

                train_r2_7 = R2_score(label_hats_7, arrival_time_7).to(device)
                train_r2_130 = [R2_score(label_hats_130_, arrival_time_130_) for (label_hats_130_, arrival_time_130_) in
                                zip(label_hats_130, arrival_time_130)]
                # BP
                optim.zero_grad()
                total_loss.backward(retain_graph=True)
                optim.step()
                end_time = time()
                # Update the graph and cnn feature
                reset_graph_feature(graph_7, device)
                for i in sampled_130_idx:
                    reset_graph_feature(graph_130_all[i], device)
                training_sets = options.training_set.split(',')
                designs = [training_sets[i] for i in sampled_130_idx]
                designs_info = ",".join(training_sets[i] for i in sampled_130_idx)
                formatted_loss_130 = ["{:.2f}".format(loss.item()) for loss in train_loss_130_all]
                loss_130_msg = ",".join(d + ": " + formatted_loss_130[i] for i, d in enumerate(designs))
                formatted_r2_130 = ["{:.2f}".format(r2.item()) for r2 in train_r2_130]
                r2_130_msg = ",".join(d + ": " + formatted_r2_130[i] for i, d in enumerate(designs))
                print("e{}, sampled 130 designs: {}, b{}/{}, total loss:{:.2f}, 7nm loss: {:.2f}, 130nm loss: {}, "
                      "7nm r2:{:.2f}, 130nm r2: {}, batch time: {:.2f}".format(epoch, designs_info, bidx, num_batch_7,
                                                                               total_loss.item(), train_loss_7.item(),
                                                                               loss_130_msg, train_r2_7.item(), r2_130_msg,
                                                                               end_time - start_time))
                if epoch < 10:
                    flag = bidx % 50 == 0
                elif epoch < 30:
                    flag = bidx % 15 == 0
                elif epoch < 50:
                    flag = bidx % 10 == 0
                elif epoch < 100:
                    flag = bidx % 15 == 0
                else:
                    flag = bidx % 5 == 0
                if flag or bidx == num_batch_7 - 1:
                    val_res, val_F1_score, val_r2 = validate(val_dataset, device, model, cnn, beta, options)
                    if options.task == 'cls':
                        judgement = val_F1_score > max_F1_score
                    elif options.task == 'reg':
                        judgement = val_r2 > max_r2
                    else:
                        assert False
                    # judgement = True
                    if judgement:
                        stop_score = 0
                        max_F1_score = val_F1_score
                        max_r2 = val_r2
                        print("Saving model.... ", os.path.join(options.model_saving_dir))
                        if os.path.exists(options.model_saving_dir) is False:
                            os.makedirs(options.model_saving_dir)
                        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                            parameters = options
                            pickle.dump((parameters, model, cnn), f)
                        print("Model successfully saved")

