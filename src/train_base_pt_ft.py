r"""
Baseline: Pretraining on 130nm data, finetuning on 7nm data.
Adapted for cell delay regression using HGAT embeddings (SPICE only).
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
from hgat import HGATDesignEncoder, build_graph_from_spice_text, get_hgat_in_dim_map
import tee
from test_r2_report import run_train_and_report_test


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
        g, feats, _ = build_graph_from_spice_text(sp_text, root_subckt=sub_name)
        if feats["PMOS"].shape[0] + feats["NMOS"].shape[0] == 0:
            continue
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
        g, feats, _ = build_graph_from_spice_text(sp_text)
        if feats["PMOS"].shape[0] + feats["NMOS"].shape[0] == 0:
            continue
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
            g, feats, _ = build_graph_from_spice_text(sp_text, root_subckt=sub_name)
            if feats["PMOS"].shape[0] + feats["NMOS"].shape[0] == 0:
                continue
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
            g, feats, _ = build_graph_from_spice_text(sp_text)
            if feats["PMOS"].shape[0] + feats["NMOS"].shape[0] == 0:
                continue
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[ctype] = z
    print(f"[Info] Pre-computed source Z for {len(z_dict)} cells.")
    return z_dict


def train(options, seed):
    th.multiprocessing.set_sharing_strategy('file_system')
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

    pretrain_df = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train
    finetune_df = df_tgt_train
    val_df = df_tgt_val

    pretrain_ds = CellDelayDataset(pretrain_df, x_mean, x_std, y_mean, y_std)
    finetune_ds = CellDelayDataset(finetune_df, x_mean, x_std, y_mean, y_std)
    val_ds = CellDelayDataset(val_df, x_mean, x_std, y_mean, y_std)

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    pretrain_dl = DataLoader(pretrain_ds, batch_size=options.batch_size, shuffle=True, collate_fn=my_collate)
    finetune_dl = DataLoader(finetune_ds, batch_size=options.batch_size, shuffle=True, collate_fn=my_collate)
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

    with open(os.path.join(options.model_saving_dir, 'seed.txt'), 'a') as f:
        f.write(str(seed))

    print('Hyperparameters are listed as follows:')
    print(options)
    print('seed:', seed)
    print('The model architecture is shown as follow:')
    print(model)
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
    best_val = float("inf")
    best_epoch = -1
    best_val_r2 = float("-inf")
    best_stage = ""
    best_ckpt_path = os.path.join(options.model_saving_dir, "ckpt_best.pt")
    total_epochs = options.num_epoch
    pretrain_epochs = getattr(options, "pretrain_epochs", 0)
    if pretrain_epochs <= 0 or pretrain_epochs >= total_epochs:
        pretrain_epochs = total_epochs // 2
    finetune_epochs = total_epochs - pretrain_epochs

    def run_epoch(loader, z_dict, graph_cache, train_mode: bool):
        if train_mode:
            if options.freeze_hgat:
                enc.eval()
            else:
                enc.train()
            model.train()
        else:
            enc.eval()
            model.eval()

        total_loss = 0.0
        total_n = 0
        r2_score.reset()
        with th.set_grad_enabled(train_mode):
            for xb, yb, cts in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                zb = build_z_batch(
                    cts, device, design_dim,
                    z_dict=z_dict, graph_cache=graph_cache, enc=enc,
                    dedup=options.dedup_z
                )

                pred = model(xb, zb)
                loss = loss_fn(pred, yb)

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * len(yb)
                total_n += len(yb)
                r2_score.update(pred, yb)

        avg_loss = total_loss / max(1, total_n)
        avg_r2 = r2_score.compute().item() if total_n > 0 else 0.0
        return avg_loss, avg_r2

    for epoch in range(pretrain_epochs):
        train_loss, train_r2 = run_epoch(pretrain_dl, z_dict_src, graph_cache_src, True)
        val_loss, val_r2 = run_epoch(val_dl, z_dict_tgt, graph_cache_tgt, False)
        print(
            f"[Pretrain] e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_val_r2 = val_r2
            best_stage = "pretrain"
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
                    "best_val_loss": best_val,
                },
                best_ckpt_path,
            )
            print("Model successfully saved")

    for epoch in range(finetune_epochs):
        train_loss, train_r2 = run_epoch(finetune_dl, z_dict_tgt, graph_cache_tgt, True)
        val_loss, val_r2 = run_epoch(val_dl, z_dict_tgt, graph_cache_tgt, False)
        print(
            f"[Finetune] e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = pretrain_epochs + epoch + 1
            best_val_r2 = val_r2
            best_stage = "finetune"
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
                    "epoch": pretrain_epochs + epoch + 1,
                    "best_val_loss": best_val,
                },
                best_ckpt_path,
            )
            print("Model successfully saved")

    if best_epoch > 0:
        print(
            f"[Best] stage:{best_stage}, epoch:{best_epoch}, val_loss:{best_val:.6f}, "
            f"val_r2:{best_val_r2:.4f}, ckpt:{best_ckpt_path}"
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
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    os.makedirs(copilot_log_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        run_train_and_report_test(train, options, seed, script_name=__file__)