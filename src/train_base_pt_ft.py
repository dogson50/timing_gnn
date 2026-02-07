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
    dropout = getattr(options, "mlp_dropout", 0.0)

    in_map = {"NET": 4, "PMOS": 2, "NMOS": 2}
    enc = HGATDesignEncoder(in_dim_map=in_map, hid=hgat_hid, out=design_dim, num_heads=hgat_heads).to(device)
    model = CellDelayRegressor(in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout).to(device)

    with open(os.path.join(options.model_saving_dir, 'seed.txt'), 'a') as f:
        f.write(str(seed))

    print('Hyperparameters are listed as follows:')
    print(options)
    print('seed:', seed)
    print('The model architecture is shown as follow:')
    print(model)
    print("----------------Loading HGAT embeddings----------------")
    z_dict_src = precompute_z_from_src_spice(data_dir, meta, enc, device, design_dim)
    z_dict_tgt = precompute_z_from_tgt_spice(data_dir, meta, enc, device, design_dim)

    def z_provider_src(ct):
        z = z_dict_src.get(ct)
        if z is None:
            return th.zeros(1, design_dim, device=device)
        return z

    def z_provider_tgt(ct):
        z = z_dict_tgt.get(ct)
        if z is None:
            return th.zeros(1, design_dim, device=device)
        return z

    optimizer = th.optim.Adam(list(enc.parameters()) + list(model.parameters()),
                              lr=options.learning_rate, weight_decay=options.weight_decay)
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)

    print("----------------Start training---------------")
    best_val = float("inf")
    total_epochs = options.num_epoch
    pretrain_epochs = getattr(options, "pretrain_epochs", 0)
    if pretrain_epochs <= 0 or pretrain_epochs >= total_epochs:
        pretrain_epochs = total_epochs // 2
    finetune_epochs = total_epochs - pretrain_epochs

    def run_epoch(loader, z_provider, train_mode: bool):
        if train_mode:
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
                zb = th.cat([z_provider(ct) for ct in cts], dim=0)

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
        train_loss, train_r2 = run_epoch(pretrain_dl, z_provider_src, True)
        val_loss, val_r2 = run_epoch(val_dl, z_provider_tgt, False)
        print(
            f"[Pretrain] e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(options.model_saving_dir, exist_ok=True)
            with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                parameters = options
                pickle.dump((parameters, model, enc), f)
            print("Model successfully saved")

    for epoch in range(finetune_epochs):
        train_loss, train_r2 = run_epoch(finetune_dl, z_provider_tgt, True)
        val_loss, val_r2 = run_epoch(val_dl, z_provider_tgt, False)
        print(
            f"[Finetune] e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(options.model_saving_dir, exist_ok=True)
            with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                parameters = options
                pickle.dump((parameters, model, enc), f)
            print("Model successfully saved")


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
        train(options, seed)
