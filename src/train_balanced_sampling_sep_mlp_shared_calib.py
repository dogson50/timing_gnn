r"""Balanced sampling with shared trunk + domain calibration heads (no disentanglement losses)."""

import os
import re
import pickle
import random
import numpy as np
import torch as th
import torch.nn as nn
from torchmetrics import R2Score
from torch.utils.data import Dataset as TorchDataset, DataLoader

from options import get_options
import tee
from test_r2_report import run_train_and_report_test

from hgat import HGATDesignEncoder, build_graph_from_spice_text, get_hgat_in_dim_map

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


_SENSE2ID = {
    "unknown": 0,
    "positive_unate": 1,
    "negative_unate": 2,
    "non_unate": 3,
}


def _norm_pin_name(v):
    if v is None:
        return "<UNK>"
    s = str(v).strip()
    return s if s else "<UNK>"


def _norm_cond_name(v):
    if v is None:
        return "<NONE>"
    s = re.sub(r"\s+", "", str(v).strip()).upper()
    return s if s else "<NONE>"


def _build_arc_vocabs(*dfs):
    pin2id = {"<UNK>": 0}
    cond2id = {"<NONE>": 0}
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        for col in ["from_pin", "to_pin"]:
            if col not in df.columns:
                continue
            for raw in df[col].astype(str).values:
                p = _norm_pin_name(raw)
                if p not in pin2id:
                    pin2id[p] = len(pin2id)
        cond_col = "arc_cond" if "arc_cond" in df.columns else None
        if cond_col is None and "when_cond" in df.columns:
            cond_col = "when_cond"
        if cond_col is not None:
            for raw in df[cond_col].astype(str).values:
                c = _norm_cond_name(raw)
                if c not in cond2id:
                    cond2id[c] = len(cond2id)
    return pin2id, cond2id


def _to_pol_id(v):
    return 1 if str(v).lower() == "rise" else 0


def _to_sense_id(v):
    return _SENSE2ID.get(str(v).strip().lower(), 0)


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
    def __init__(self, df, x_mean, x_std, y_mean, y_std, pin2id, cond2id):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)
        self.pin2id = pin2id
        self.cond2id = cond2id
        from_vals = self.df["from_pin"].values if "from_pin" in self.df.columns else ["<UNK>"] * len(self.df)
        to_vals = self.df["to_pin"].values if "to_pin" in self.df.columns else ["<UNK>"] * len(self.df)
        pol_vals = self.df["pol"].values if "pol" in self.df.columns else ["fall"] * len(self.df)
        sense_vals = self.df["timing_sense"].values if "timing_sense" in self.df.columns else ["unknown"] * len(self.df)
        cond_vals = self.df["arc_cond"].values if "arc_cond" in self.df.columns else ["<NONE>"] * len(self.df)
        self.from_pin_id = np.array([self.pin2id.get(_norm_pin_name(v), 0) for v in from_vals], dtype=np.int64)
        self.to_pin_id = np.array([self.pin2id.get(_norm_pin_name(v), 0) for v in to_vals], dtype=np.int64)
        self.pol_id = np.array([_to_pol_id(v) for v in pol_vals], dtype=np.int64)
        self.sense_id = np.array([_to_sense_id(v) for v in sense_vals], dtype=np.int64)
        self.cond_id = np.array([self.cond2id.get(_norm_cond_name(v), 0) for v in cond_vals], dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            th.from_numpy(self.x[i]),
            th.tensor(self.y[i]),
            self.cts[i],
            th.tensor(self.from_pin_id[i], dtype=th.long),
            th.tensor(self.to_pin_id[i], dtype=th.long),
            th.tensor(self.pol_id[i], dtype=th.long),
            th.tensor(self.sense_id[i], dtype=th.long),
            th.tensor(self.cond_id[i], dtype=th.long),
        )


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
        g, feats, _ = build_graph_from_spice_text(sp_text)
        if feats["PMOS"].shape[0] + feats["NMOS"].shape[0] == 0:
            continue
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
        g, feats, _ = build_graph_from_spice_text(sp_text, root_subckt=sub_name)
        if feats["PMOS"].shape[0] + feats["NMOS"].shape[0] == 0:
            continue
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


class SharedCalibRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        design_dim,
        hid=256,
        dropout=0.0,
        *,
        num_pins=1,
        num_conds=1,
        pin_emb_dim=8,
        pol_emb_dim=2,
        sense_emb_dim=4,
        cond_emb_dim=8,
        use_arc_cond=False,
        arc_sep_domain_emb=False,
        arc_cond_mode="concat",
        src_use_arc_cond=True,
    ):
        super().__init__()
        self.use_arc_cond = bool(use_arc_cond)
        self.arc_sep_domain_emb = bool(arc_sep_domain_emb)
        self.src_use_arc_cond = bool(src_use_arc_cond)
        self.arc_cond_mode = str(arc_cond_mode).lower()
        if self.arc_cond_mode not in ("concat", "film"):
            raise ValueError(f"Unknown arc_cond_mode: {self.arc_cond_mode}")
        self.base_dim = in_dim + design_dim
        self.arc_dim = 2 * pin_emb_dim + pol_emb_dim + sense_emb_dim + cond_emb_dim

        if self.use_arc_cond:
            n_pins = max(1, int(num_pins))
            n_conds = max(1, int(num_conds))
            if self.arc_sep_domain_emb:
                self.pin_emb_tgt = nn.Embedding(n_pins, pin_emb_dim)
                self.pin_emb_src = nn.Embedding(n_pins, pin_emb_dim)
                self.pol_emb_tgt = nn.Embedding(2, pol_emb_dim)
                self.pol_emb_src = nn.Embedding(2, pol_emb_dim)
                self.sense_emb_tgt = nn.Embedding(len(_SENSE2ID), sense_emb_dim)
                self.sense_emb_src = nn.Embedding(len(_SENSE2ID), sense_emb_dim)
                self.cond_emb_tgt = nn.Embedding(n_conds, cond_emb_dim)
                self.cond_emb_src = nn.Embedding(n_conds, cond_emb_dim)
            else:
                self.pin_emb = nn.Embedding(n_pins, pin_emb_dim)
                self.pol_emb = nn.Embedding(2, pol_emb_dim)
                self.sense_emb = nn.Embedding(len(_SENSE2ID), sense_emb_dim)
                self.cond_emb = nn.Embedding(n_conds, cond_emb_dim)
            if self.arc_cond_mode == "film":
                self.film_tgt = nn.Linear(self.arc_dim, 2 * self.base_dim)
                self.film_src = nn.Linear(self.arc_dim, 2 * self.base_dim)

        extra_dim_tgt = self.arc_dim if (self.use_arc_cond and self.arc_cond_mode == "concat") else 0
        extra_dim_src = self.arc_dim if (self.use_arc_cond and self.src_use_arc_cond and self.arc_cond_mode == "concat") else 0
        self.backbone_tgt = self._make_backbone(self.base_dim + extra_dim_tgt, hid, dropout)
        self.backbone_src = self._make_backbone(self.base_dim + extra_dim_src, hid, dropout)
        self.shared_head = nn.Linear(hid, 1)
        self.calib_tgt = nn.Linear(hid, 1)
        self.calib_src = nn.Linear(hid, 1)

    @staticmethod
    def _make_backbone(input_dim, hid, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )

    def _embed_arc(self, node, from_pin_id, to_pin_id, pol_id, sense_id, cond_id):
        if self.arc_sep_domain_emb:
            if node == "tgt":
                af = self.pin_emb_tgt(from_pin_id)
                at = self.pin_emb_tgt(to_pin_id)
                ap = self.pol_emb_tgt(pol_id)
                ase = self.sense_emb_tgt(sense_id)
                ac = self.cond_emb_tgt(cond_id)
            elif node == "src":
                af = self.pin_emb_src(from_pin_id)
                at = self.pin_emb_src(to_pin_id)
                ap = self.pol_emb_src(pol_id)
                ase = self.sense_emb_src(sense_id)
                ac = self.cond_emb_src(cond_id)
            else:
                raise ValueError(f"Unknown node type: {node}")
        else:
            af = self.pin_emb(from_pin_id)
            at = self.pin_emb(to_pin_id)
            ap = self.pol_emb(pol_id)
            ase = self.sense_emb(sense_id)
            ac = self.cond_emb(cond_id)
        return th.cat([af, at, ap, ase, ac], dim=1)

    def forward(
        self,
        x,
        z,
        node="tgt",
        from_pin_id=None,
        to_pin_id=None,
        pol_id=None,
        sense_id=None,
        cond_id=None,
    ):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h_base = th.cat([x, z], dim=1)
        h = h_base
        apply_arc = self.use_arc_cond and (node == "tgt" or (node == "src" and self.src_use_arc_cond))
        if apply_arc:
            if any(v is None for v in (from_pin_id, to_pin_id, pol_id, sense_id, cond_id)):
                raise ValueError("Arc conditioning enabled but arc ids are missing.")
            arc_feat = self._embed_arc(node, from_pin_id, to_pin_id, pol_id, sense_id, cond_id)
            if self.arc_cond_mode == "concat":
                h = th.cat([h_base, arc_feat], dim=1)
            else:
                if node == "tgt":
                    gamma, beta = self.film_tgt(arc_feat).chunk(2, dim=1)
                elif node == "src":
                    gamma, beta = self.film_src(arc_feat).chunk(2, dim=1)
                else:
                    raise ValueError(f"Unknown node type: {node}")
                h = h_base * (1.0 + gamma) + beta

        if node == "tgt":
            h = self.backbone_tgt(h)
            pred = self.shared_head(h).squeeze(-1) + self.calib_tgt(h).squeeze(-1)
            return pred
        if node == "src":
            h = self.backbone_src(h)
            pred = self.shared_head(h).squeeze(-1) + self.calib_src(h).squeeze(-1)
            return pred
        raise ValueError(f"Unknown node type: {node}")


@th.no_grad()
def validate_cell(val_dl, enc, model, device, design_dim, *, z_dict=None, graph_cache=None, dedup=False):
    enc.eval()
    model.eval()
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)
    total_loss = 0.0
    total_n = 0
    for xb, yb, cts, fp, tp, pol, sense, cond in val_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        fp = fp.to(device)
        tp = tp.to(device)
        pol = pol.to(device)
        sense = sense.to(device)
        cond = cond.to(device)
        zb = build_z_batch(
            cts, device, design_dim,
            z_dict=z_dict, graph_cache=graph_cache, enc=enc,
            dedup=dedup,
        )
        pred = model(
            xb, zb, node="tgt",
            from_pin_id=fp, to_pin_id=tp, pol_id=pol, sense_id=sense, cond_id=cond,
        )
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        r2_score.update(pred, yb)
    avg_loss = total_loss / max(1, total_n)
    val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
    return avg_loss, val_r2


def train_balanced_sep_mlp_shared_calib(options, seed):
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

    ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std, pin2id, cond2id)
    ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std, pin2id, cond2id)
    val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std, pin2id, cond2id)
    test_ds = CellDelayDataset(df_tgt_test, x_mean, x_std, y_mean, y_std, pin2id, cond2id) if df_tgt_test is not None else None

    def my_collate(batch):
        xs, ys, cts, fps, tps, pols, senses, conds = zip(*batch)
        return (
            th.stack(xs), th.stack(ys), cts,
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
    model = SharedCalibRegressor(
        in_dim=options.in_dim,
        design_dim=design_dim,
        hid=hgat_hid,
        dropout=dropout,
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
    if test_dl is None:
        print("[Warn] dataset.pkl has no tgt_test_df, skipping test evaluation.")
    best_val = float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    best_test_r2 = 0.0
    best_test_loss = 0.0
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

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt, fp_tgt, tp_tgt, pol_tgt, sense_tgt, cond_tgt in dl_tgt:
            xb_tgt = xb_tgt.to(device)
            yb_tgt = yb_tgt.to(device)
            fp_tgt = fp_tgt.to(device)
            tp_tgt = tp_tgt.to(device)
            pol_tgt = pol_tgt.to(device)
            sense_tgt = sense_tgt.to(device)
            cond_tgt = cond_tgt.to(device)
            zb_tgt = build_z_batch(
                cts_tgt,
                device,
                design_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                enc=enc,
                dedup=options.dedup_z,
            )
            pred_tgt = model(
                xb_tgt, zb_tgt, node="tgt",
                from_pin_id=fp_tgt, to_pin_id=tp_tgt, pol_id=pol_tgt, sense_id=sense_tgt, cond_id=cond_tgt,
            )
            loss_tgt = loss_fn(pred_tgt, yb_tgt)

            loss_src_list = []
            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src, fp_src, tp_src, pol_src, sense_src, cond_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src, fp_src, tp_src, pol_src, sense_src, cond_src = next(dl_src_iter)
                xb_src = xb_src.to(device)
                yb_src = yb_src.to(device)
                fp_src = fp_src.to(device)
                tp_src = tp_src.to(device)
                pol_src = pol_src.to(device)
                sense_src = sense_src.to(device)
                cond_src = cond_src.to(device)
                zb_src = build_z_batch(
                    cts_src,
                    device,
                    design_dim,
                    z_dict=z_dict_src,
                    graph_cache=graph_cache_src,
                    enc=enc,
                    dedup=options.dedup_z,
                )
                pred_src = model(
                    xb_src, zb_src, node="src",
                    from_pin_id=fp_src, to_pin_id=tp_src, pol_id=pol_src, sense_id=sense_src, cond_id=cond_src,
                )
                loss_src_list.append(loss_fn(pred_src, yb_src))

            total_loss_batch = (
                loss_tgt * batch_size_tgt
                + options.loss_weight_45 * sum(loss_src_list) * batch_size_src
            ) / (batch_size_tgt + len(loss_src_list) * batch_size_src)

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

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
        if test_dl is not None:
            test_loss, test_r2 = validate_cell(
                test_dl,
                enc,
                model,
                device,
                design_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                dedup=options.dedup_z,
            )
        else:
            test_loss, test_r2 = float("nan"), float("nan")

        print(
            f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, "
            f"test_loss:{test_loss:.4f}, test_r2:{test_r2:.3f}"
        )

        if val_r2 > best_val:
            best_val = val_r2
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_test_r2 = test_r2
            best_test_loss = test_loss
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
                    "scaler_stats": scaler_stats,
                    "y_scaler": y_scaler,
                    "epoch": epoch + 1,
                    "best_val_r2": best_val,
                    "script": "shared_calib",
                },
                best_ckpt_path,
            )
            print("Model successfully saved")

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
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    os.makedirs(copilot_log_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")
    stdout_f = "{}/stdout.log".format(options.model_saving_dir)
    stderr_f = "{}/stderr.log".format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        run_train_and_report_test(train_balanced_sep_mlp_shared_calib, options, seed, script_name=__file__)