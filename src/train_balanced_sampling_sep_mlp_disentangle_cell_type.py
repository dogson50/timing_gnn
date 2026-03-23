r"""Balanced sampling + semantic disentangle/alignment with HGAT embeddings."""

import os
import re
import pickle
import random
from collections import defaultdict
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import R2Score
from torch.utils.data import Dataset as TorchDataset, DataLoader

from options import get_options
import tee

from hgat import HGATDesignEncoder, build_graph_from_spice_text, get_hgat_in_dim_map
from parse_lib import cell_type_to_topology_group

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
CMD_CONTEXT_COLS = ["log_slew", "log_cap", "pol_bit"]
CMD_CONTEXT_IDX = [NUMERIC_COLS.index(c) for c in CMD_CONTEXT_COLS]


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


def _pin_sort_key(v):
    s = _norm_pin_name(v).upper()
    m = re.fullmatch(r"([A-Z]+)(\d*)", s)
    if m:
        prefix = m.group(1)
        suffix = int(m.group(2)) if m.group(2) else 0
        return prefix, suffix, s
    return s, 0, s


def _build_semantic_pin_role_maps(*dfs):
    cell_inputs = defaultdict(set)
    cell_outputs = defaultdict(set)
    for df in dfs:
        if df is None or len(df) == 0 or "cell_name" not in df.columns:
            continue
        for _, row in df.iterrows():
            cell_name = str(row.get("cell_name"))
            cell_inputs[cell_name].add(_norm_pin_name(row.get("from_pin")))
            cell_outputs[cell_name].add(_norm_pin_name(row.get("to_pin")))

    role_maps = {}
    all_cells = set(cell_inputs.keys()) | set(cell_outputs.keys())
    for ctype in all_cells:
        in_map = {}
        for idx, raw_pin in enumerate(sorted(cell_inputs.get(ctype, []), key=_pin_sort_key), start=1):
            in_map[raw_pin.upper()] = f"IN{idx}"
        out_map = {}
        for raw_pin in cell_outputs.get(ctype, []):
            out_map[raw_pin.upper()] = "OUT"
        role_maps[ctype] = {"in": in_map, "out": out_map}
    return role_maps


def _canonicalize_pin_role(cell_name, pin_name, role_maps, io_kind="in"):
    cell_name = str(cell_name)
    raw = _norm_pin_name(pin_name).upper()
    cell_map = role_maps.get(cell_name, {})
    role = cell_map.get(io_kind, {}).get(raw)
    if role is not None:
        return role
    if io_kind == "out":
        return "OUT"
    return raw


def _canonicalize_arc_condition(cell_name, cond_name, role_maps):
    cond = _norm_cond_name(cond_name)
    if cond == "<NONE>":
        return cond
    cell_name = str(cell_name)
    pin_map = role_maps.get(cell_name, {}).get("in", {})
    out_map = role_maps.get(cell_name, {}).get("out", {})
    merged_map = {**pin_map, **out_map}
    if not merged_map:
        return cond

    tokens = sorted(merged_map.keys(), key=len, reverse=True)
    for raw in tokens:
        role = merged_map[raw]
        patt = rf"(?<![A-Z0-9_]){re.escape(raw)}(?![A-Z0-9_])"
        cond = re.sub(patt, role, cond)
    return cond


def _build_semantic_arc_key(cell_type, cell_name, from_pin, to_pin, timing_sense, arc_cond, role_maps):
    topo = cell_type_to_topology_group(cell_type)
    from_role = _canonicalize_pin_role(cell_name, from_pin, role_maps, io_kind="in")
    to_role = _canonicalize_pin_role(cell_name, to_pin, role_maps, io_kind="out")
    cond_role = _canonicalize_arc_condition(cell_name, arc_cond, role_maps)
    sense = str(timing_sense).strip().lower() if str(timing_sense).strip() else "unknown"
    return "|".join([str(topo), str(from_role), str(to_role), str(sense), str(cond_role)])


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
    def __init__(self, df, x_mean, x_std, y_mean, y_std, pin2id, cond2id, role_maps):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)
        self.pin2id = pin2id
        self.cond2id = cond2id
        self.role_maps = role_maps
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
        self.cmd_ctx = self.x[:, CMD_CONTEXT_IDX].astype(np.float32)
        self.sem_keys = np.array([
            _build_semantic_arc_key(
                self.df.iloc[i]["cell_type"],
                self.df.iloc[i]["cell_name"],
                from_vals[i],
                to_vals[i],
                sense_vals[i],
                cond_vals[i],
                self.role_maps,
            )
            for i in range(len(self.df))
        ], dtype=object)
        self.cmd_keys = np.array([
            f"{self.sem_keys[i]}|pol={str(pol_vals[i]).lower()}"
            for i in range(len(self.df))
        ], dtype=object)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            th.from_numpy(self.x[i]),
            th.tensor(self.y[i]),
            self.cts[i],
            self.sem_keys[i],
            self.cmd_keys[i],
            th.from_numpy(self.cmd_ctx[i]),
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


def build_alignment_labels(cell_types, semantic_keys=None, mode="semantic_arc"):
    if mode == "semantic_arc":
        if semantic_keys is None:
            raise ValueError("semantic_keys are required when alignment mode is semantic_arc.")
        return [str(v) for v in semantic_keys]
    if mode == "cell_type":
        return [str(ct) for ct in cell_types]
    if mode == "topology":
        return [cell_type_to_topology_group(ct) for ct in cell_types]
    raise ValueError(f"Unknown alignment mode: {mode}")


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


def conditional_cmd_loss(src_feats, tgt_feats, src_labels, tgt_labels, k=5):
    device = None
    if src_feats is not None and isinstance(src_feats, th.Tensor):
        device = src_feats.device
    elif tgt_feats is not None and isinstance(tgt_feats, th.Tensor):
        device = tgt_feats.device

    if src_feats is None or tgt_feats is None or src_feats.numel() == 0 or tgt_feats.numel() == 0:
        return th.tensor(0.0, device=device)

    src_bucket = defaultdict(list)
    tgt_bucket = defaultdict(list)
    for idx, key in enumerate(src_labels):
        src_bucket[str(key)].append(idx)
    for idx, key in enumerate(tgt_labels):
        tgt_bucket[str(key)].append(idx)

    shared_keys = sorted(set(src_bucket.keys()) & set(tgt_bucket.keys()))
    if not shared_keys:
        return th.tensor(0.0, device=device)

    losses = []
    for key in shared_keys:
        src_idx = th.tensor(src_bucket[key], device=src_feats.device, dtype=th.long)
        tgt_idx = th.tensor(tgt_bucket[key], device=tgt_feats.device, dtype=th.long)
        losses.append(cmd_loss(src_feats.index_select(0, src_idx), tgt_feats.index_select(0, tgt_idx), k=k))
    return th.stack(losses).mean() if losses else th.tensor(0.0, device=device)


class DisentangleCellDelayRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        design_dim,
        node_feat_dim,
        hid=256,
        dropout=0.0,
        clr_proj_dim=64,
        cmd_proj_dim=64,
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
        self.mlp_design = self._make_mlp(design_dim, node_feat_dim, hid, dropout)
        self.mlp_process = self._make_mlp(in_dim, node_feat_dim, hid, dropout)
        self.base_dim = node_feat_dim * 2
        self.arc_dim = 2 * pin_emb_dim + pol_emb_dim + sense_emb_dim + cond_emb_dim
        self.cmd_ctx_dim = len(CMD_CONTEXT_COLS)
        self.clr_projector = nn.Sequential(
            nn.Linear(node_feat_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, clr_proj_dim),
        )
        self.cmd_projector = nn.Sequential(
            nn.Linear(node_feat_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, cmd_proj_dim),
        )
        self.cmd_context_proj = nn.Sequential(
            nn.Linear(self.cmd_ctx_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, cmd_proj_dim),
        )
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
        self.head_tgt = self._make_head(self.base_dim + extra_dim_tgt, hid, dropout)
        self.head_src = self._make_head(self.base_dim + extra_dim_src, hid, dropout)

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

    def encode(self, x, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h_design = self.mlp_design(z)
        h_process = self.mlp_process(x)
        h = th.cat([h_design, h_process], dim=1)
        return h, h_design, h_process

    def project_clr(self, h_design):
        return self.clr_projector(h_design)

    def project_cmd(self, h_process, cmd_ctx):
        if cmd_ctx.dim() == 1:
            cmd_ctx = cmd_ctx.unsqueeze(0)
        return self.cmd_projector(h_process) - self.cmd_context_proj(cmd_ctx)

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
        h, h_design, h_process = self.encode(x, z)
        h_in = h
        apply_arc = self.use_arc_cond and (node == "tgt" or (node == "src" and self.src_use_arc_cond))
        if apply_arc:
            if any(v is None for v in (from_pin_id, to_pin_id, pol_id, sense_id, cond_id)):
                raise ValueError("Arc conditioning enabled but arc ids are missing.")
            arc_feat = self._embed_arc(node, from_pin_id, to_pin_id, pol_id, sense_id, cond_id)
            if self.arc_cond_mode == "concat":
                h_in = th.cat([h, arc_feat], dim=1)
            else:
                if node == "tgt":
                    gamma, beta = self.film_tgt(arc_feat).chunk(2, dim=1)
                elif node == "src":
                    gamma, beta = self.film_src(arc_feat).chunk(2, dim=1)
                else:
                    raise ValueError(f"Unknown node type: {node}")
                h_in = h * (1.0 + gamma) + beta
        if node == "tgt":
            pred = self.head_tgt(h_in).squeeze(-1)
        elif node == "src":
            pred = self.head_src(h_in).squeeze(-1)
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
    for xb, yb, cts, _, _, _, fp, tp, pol, sense, cond in val_dl:
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
            dedup=dedup
        )
        pred, _, _ = model(
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


def train_balanced_sep_mlp_disentangle_cell_type(options, seed):
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
    split_mode = meta.get("split_info", {}).get("split_mode", "unknown")
    print(f"[Info] Target split mode: {split_mode}")
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
            zb_tgt = build_z_batch(
                cts_tgt, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, enc=enc,
                dedup=options.dedup_z
            )
            pred_tgt, h_design_tgt, h_process_tgt = model(
                xb_tgt, zb_tgt, node="tgt",
                from_pin_id=fp_tgt, to_pin_id=tp_tgt, pol_id=pol_tgt, sense_id=sense_tgt, cond_id=cond_tgt,
            )
            loss_tgt = loss_fn(pred_tgt, yb_tgt)

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
                zb_src = build_z_batch(
                    cts_src, device, design_dim,
                    z_dict=z_dict_src, graph_cache=graph_cache_src, enc=enc,
                    dedup=options.dedup_z
                )
                pred_src, h_design_src, h_process_src = model(
                    xb_src, zb_src, node="src",
                    from_pin_id=fp_src, to_pin_id=tp_src, pol_id=pol_src, sense_id=sense_src, cond_id=cond_src,
                )
                loss_src = loss_fn(pred_src, yb_src)
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
                            device=device, normalization=options.norm_clr
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
        if test_dl is not None:
            test_loss, test_r2 = validate_cell(
                test_dl, enc, model, device, design_dim,
                z_dict=z_dict_tgt, graph_cache=graph_cache_tgt, dedup=options.dedup_z
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
                    "clr_proj_dim": getattr(options, "clr_proj_dim", 64),
                    "cmd_proj_dim": getattr(options, "cmd_proj_dim", 64),
                    "clr_label_mode": getattr(options, "clr_label_mode", "semantic_arc"),
                    "cmd_label_mode": getattr(options, "cmd_label_mode", "semantic_arc"),
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
            f"val_loss:{best_val_loss:.6f}, "
            f"test_r2:{best_test_r2:.4f}, test_loss:{best_test_loss:.6f}, "
            f"ckpt:{best_ckpt_path}"
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
        train_balanced_sep_mlp_disentangle_cell_type(options, seed)
