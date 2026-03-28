r"""Step8: Step5-A baseline + arc-conditional shared-calibration regressor."""

import os
import re
import pickle
import random
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
ENCODER_LR_SCALE_DEFAULT = 0.2
GRAPH_NET_DIM = 7
GRAPH_MOS_DIM = 8


def _norm_pin_name(v):
    if v is None:
        return "<UNK>"
    s = str(v).strip()
    return s if s else "<UNK>"


def _build_pin_vocab(*dfs):
    pin2id = {"<UNK>": 0}
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        for col in ["from_pin", "to_pin"]:
            if col not in df.columns:
                continue
            vals = df[col].astype(str).values
            for raw in vals:
                p = _norm_pin_name(raw)
                if p not in pin2id:
                    pin2id[p] = len(pin2id)
    return pin2id


def _to_pol_id(v):
    s = str(v).lower()
    return 1 if s == "rise" else 0


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
    def __init__(self, df, x_mean, x_std, y_mean, y_std, pin2id):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)
        self.pin2id = pin2id
        from_vals = self.df["from_pin"].values if "from_pin" in self.df.columns else ["<UNK>"] * len(self.df)
        to_vals = self.df["to_pin"].values if "to_pin" in self.df.columns else ["<UNK>"] * len(self.df)
        pol_vals = self.df["pol"].values if "pol" in self.df.columns else ["fall"] * len(self.df)
        self.from_pin_id = np.array([self.pin2id.get(_norm_pin_name(v), 0) for v in from_vals], dtype=np.int64)
        self.to_pin_id = np.array([self.pin2id.get(_norm_pin_name(v), 0) for v in to_vals], dtype=np.int64)
        self.pol_id = np.array([_to_pol_id(v) for v in pol_vals], dtype=np.int64)

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


def _safe_log10(v, eps=1e-12):
    return float(np.log10(max(float(v), eps)))


def _to_m(val, unit):
    if unit is None or unit == "":
        return float(val)
    u = unit.lower()
    if u == "u":
        return float(val) * 1e-6
    if u == "n":
        return float(val) * 1e-9
    if u == "p":
        return float(val) * 1e-12
    if u == "m":
        return float(val) * 1e-3
    if u == "k":
        return float(val) * 1e3
    return float(val)


def _parse_value_with_optional_unit(text):
    if text is None:
        return None
    s = str(text).strip()
    m = re.match(r"([0-9.eE\-\+]+)([a-zA-Z]?)$", s)
    if not m:
        return None
    return _to_m(m.group(1), m.group(2))


def parse_transistors_spice_rich(text):
    devs = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("*", ";", "//", "*#")):
            continue
        m = re.match(r"^[Mm](\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", s, re.I)
        if not m:
            continue

        name = "M" + m.group(1)
        d, g, sr, b, model, rest = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        model_l = model.lower()
        if "nmos" in model_l or model_l.startswith("n"):
            t = "nmos"
        elif "pmos" in model_l or model_l.startswith("p"):
            t = "pmos"
        else:
            t = model_l

        rest = rest.split("$", 1)[0]
        toks = [tok for tok in re.split(r"[,\s]+", rest) if tok]
        params = {}
        for tok in toks:
            if "=" in tok:
                k, v = tok.split("=", 1)
                params[k.lower()] = v

        w = _parse_value_with_optional_unit(params.get("w"))
        l = _parse_value_with_optional_unit(params.get("l"))

        nfin = _parse_value_with_optional_unit(params.get("nfin"))
        nf = _parse_value_with_optional_unit(params.get("nf"))
        m_mult = _parse_value_with_optional_unit(params.get("m"))
        if m_mult is None:
            m_mult = _parse_value_with_optional_unit(params.get("mult"))

        devs.append(
            {
                "name": name,
                "d": d,
                "g": g,
                "s": sr,
                "b": b,
                "type": t,
                "W": w,
                "L": l,
                "nfin": nfin,
                "nf": nf,
                "m": m_mult,
            }
        )
    return devs


def build_dgl_graph_from_devs_step5(devs, top_pins):
    nets = {}

    def net_id(n):
        if n not in nets:
            nets[n] = len(nets)
        return nets[n]

    p_count = 0
    n_count = 0
    p_devs = []
    n_devs = []

    gate_src_p, gate_dst_p = [], []
    gate_src_n, gate_dst_n = [], []
    sd_src_p, sd_dst_p = [], []
    sd_src_n, sd_dst_n = [], []

    for d in devs:
        if str(d.get("type", "")).startswith("p"):
            mid = p_count
            p_count += 1
            p_devs.append(d)
            gate_src_p.append(net_id(d["g"]))
            gate_dst_p.append(mid)
            sd_src_p.extend([mid, mid])
            sd_dst_p.extend([net_id(d["s"]), net_id(d["d"])])
        elif str(d.get("type", "")).startswith("n"):
            mid = n_count
            n_count += 1
            n_devs.append(d)
            gate_src_n.append(net_id(d["g"]))
            gate_dst_n.append(mid)
            sd_src_n.extend([mid, mid])
            sd_dst_n.extend([net_id(d["s"]), net_id(d["d"])])

    data_dict = {}
    if p_count > 0:
        data_dict[("NET", "gate_of", "PMOS")] = (th.tensor(gate_src_p), th.tensor(gate_dst_p))
        data_dict[("PMOS", "sd_to", "NET")] = (th.tensor(sd_src_p), th.tensor(sd_dst_p))
        data_dict[("NET", "back_sd", "PMOS")] = (th.tensor(sd_dst_p), th.tensor(sd_src_p))
    if n_count > 0:
        data_dict[("NET", "gate_of", "NMOS")] = (th.tensor(gate_src_n), th.tensor(gate_dst_n))
        data_dict[("NMOS", "sd_to", "NET")] = (th.tensor(sd_src_n), th.tensor(sd_dst_n))
        data_dict[("NET", "back_sd", "NMOS")] = (th.tensor(sd_dst_n), th.tensor(sd_src_n))

    g = dgl.heterograph(data_dict, num_nodes_dict={"NET": len(nets), "PMOS": p_count, "NMOS": n_count})

    gate_deg = np.zeros(len(nets), dtype=np.float32)
    sd_deg = np.zeros(len(nets), dtype=np.float32)
    for nid in gate_src_p + gate_src_n:
        gate_deg[int(nid)] += 1.0
    for nid in sd_dst_p + sd_dst_n:
        sd_deg[int(nid)] += 1.0
    deg = gate_deg + sd_deg

    max_deg = float(max(1.0, deg.max() if deg.size > 0 else 1.0))
    max_gate_deg = float(max(1.0, gate_deg.max() if gate_deg.size > 0 else 1.0))
    max_sd_deg = float(max(1.0, sd_deg.max() if sd_deg.size > 0 else 1.0))

    top_pin_set = set(str(p).upper() for p in (top_pins or []))
    rail_set = {"VDD", "VPWR", "VCC", "VSS", "VGND", "GND"}
    f_net = []
    for name, nid in sorted(nets.items(), key=lambda x: x[1]):
        up = name.upper()
        is_vdd = 1.0 if up in ("VDD", "VPWR", "VCC") else 0.0
        is_vss = 1.0 if up in ("VSS", "VGND", "GND") else 0.0
        is_top_pin = 1.0 if up in top_pin_set else 0.0
        is_internal = 1.0 if (up not in top_pin_set and up not in rail_set) else 0.0
        deg_norm = float(np.log1p(deg[nid]) / np.log1p(max_deg))
        gate_norm = float(np.log1p(gate_deg[nid]) / np.log1p(max_gate_deg))
        sd_norm = float(np.log1p(sd_deg[nid]) / np.log1p(max_sd_deg))
        f_net.append([is_vdd, is_vss, is_top_pin, deg_norm, gate_norm, sd_norm, is_internal])
    f_net = th.tensor(np.array(f_net, dtype=np.float32))

    def mos_feats(list_dev):
        arr = []
        rail_set_local = {"VDD", "VPWR", "VCC", "VSS", "VGND", "GND"}
        for d in list_dev:
            raw_w = d["W"] if d.get("W") is not None else 2.0e-8
            raw_l = d["L"] if d.get("L") is not None else 2.0e-8
            nfin = d.get("nfin")
            nf = d.get("nf")
            m_mult = d.get("m")
            nfin_eff = float(nfin if nfin is not None else (nf if nf is not None else 1.0))
            nfin_eff = max(nfin_eff, 1.0)
            m_eff = float(m_mult if m_mult is not None else 1.0)
            m_eff = max(m_eff, 1.0)

            wl_ratio = float(raw_w / max(raw_l, 1e-12))
            area = float(raw_w * raw_l)

            b = str(d.get("b", "")).upper()
            s = str(d.get("s", "")).upper()
            body_tied_source = 1.0 if b == s else 0.0
            body_is_rail = 1.0 if b in rail_set_local else 0.0

            arr.append(
                [
                    _safe_log10(raw_w),
                    _safe_log10(raw_l),
                    _safe_log10(wl_ratio),
                    _safe_log10(area),
                    _safe_log10(nfin_eff),
                    _safe_log10(m_eff),
                    body_tied_source,
                    body_is_rail,
                ]
            )

        if len(arr) == 0:
            return th.zeros((0, GRAPH_MOS_DIM), dtype=th.float32)
        return th.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats(p_devs)
    f_n = mos_feats(n_devs)

    feats = {"NET": f_net, "PMOS": f_p, "NMOS": f_n}
    in_dim_map = {"NET": GRAPH_NET_DIM, "PMOS": GRAPH_MOS_DIM, "NMOS": GRAPH_MOS_DIM}
    return g, feats, in_dim_map


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
        devs = parse_transistors_spice_rich(sp_text)
        _, pins = parse_top_subckt_pins(sp_text)
        if not devs:
            continue
        g, feats, _ = build_dgl_graph_from_devs_step5(devs, pins)
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
        devs = parse_transistors_spice_rich(sub_txt)
        _, pins = parse_top_subckt_pins(sub_txt)
        if not devs:
            continue
        g, feats, _ = build_dgl_graph_from_devs_step5(devs, pins)
        graph_cache[ctype] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    print(f"[Info] Cached target graphs: {len(graph_cache)}")
    return graph_cache


def encode_graph_batch_hgat(enc, graphs, feats_list):
    if len(graphs) == 0:
        out_dim = enc.readout[-1].out_features
        return th.zeros((0, out_dim), device=next(enc.parameters()).device)

    bg = dgl.batch(graphs)
    ntypes = ["NET", "PMOS", "NMOS"]
    batched_feats = {}
    for nt in ntypes:
        chunks = [f[nt] for f in feats_list if nt in f]
        if chunks:
            batched_feats[nt] = th.cat(chunks, dim=0)

    h = {nt: enc.embed[nt](batched_feats[nt]) for nt in batched_feats}
    for layer in enc.layers:
        h = layer(bg, h)

    counts_by_type = {
        nt: bg.batch_num_nodes(nt).tolist() if nt in bg.ntypes else [0] * len(graphs)
        for nt in ntypes
    }
    offsets_by_type = {}
    for nt in ntypes:
        offs = []
        cur = 0
        for c in counts_by_type[nt]:
            offs.append(cur)
            cur += int(c)
        offsets_by_type[nt] = offs

    out_list = []
    summary_types = ["PMOS", "NMOS"] + (["NET"] if enc.use_net_readout else [])
    for i in range(len(graphs)):
        summaries = []
        for nt in summary_types:
            if nt not in h:
                continue
            cnt = int(counts_by_type[nt][i])
            if cnt <= 0:
                continue
            st = offsets_by_type[nt][i]
            x = h[nt][st: st + cnt]
            mean_v = x.mean(dim=0, keepdim=True)
            max_v = x.max(dim=0, keepdim=True).values
            std_v = x.std(dim=0, keepdim=True, unbiased=False)
            summaries.append(th.cat([mean_v, max_v, std_v], dim=1))

        if len(summaries) == 0:
            z = th.zeros(enc.summary_dim, device=next(enc.parameters()).device)
        else:
            s = th.cat(summaries, dim=0)
            if enc.type_attn_readout and s.shape[0] > 1:
                w = th.softmax(enc.type_gate(s), dim=0)
                z = (w * s).sum(dim=0)
            else:
                z = s.mean(dim=0)

        z = enc.readout(z)
        z = nn.functional.normalize(z, dim=0)
        out_list.append(z.unsqueeze(0))

    return th.cat(out_list, dim=0)


def build_z_batch(
    cts,
    device,
    design_dim,
    *,
    z_dict=None,
    graph_cache=None,
    enc=None,
    dedup=False,
    z_step_cache=None,
    use_batched_graph_encode=False,
):
    if z_dict is None and (graph_cache is None or enc is None):
        raise ValueError("build_z_batch requires z_dict or (graph_cache + enc).")

    def _get_z(ct):
        key = str(ct)
        if z_step_cache is not None and key in z_step_cache:
            return z_step_cache[key]
        if z_dict is not None:
            z = z_dict.get(key)
            if z is None:
                z = th.zeros(1, design_dim, device=device)
            if z_step_cache is not None:
                z_step_cache[key] = z
            return z
        entry = graph_cache.get(key) if graph_cache is not None else None
        if entry is None:
            z = th.zeros(1, design_dim, device=device)
            if z_step_cache is not None:
                z_step_cache[key] = z
            return z
        g, feats = entry
        z = enc(g, feats)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z_step_cache is not None:
            z_step_cache[key] = z
        return z

    if use_batched_graph_encode and z_dict is None and graph_cache is not None and enc is not None:
        uniq = []
        seen = {}
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

        z_unique = [None] * len(uniq)
        pending_keys = []
        pending_graphs = []
        pending_feats = []
        for i, key in enumerate(uniq):
            if z_step_cache is not None and key in z_step_cache:
                z_unique[i] = z_step_cache[key]
                continue
            entry = graph_cache.get(key)
            if entry is None:
                z = th.zeros(1, design_dim, device=device)
                z_unique[i] = z
                if z_step_cache is not None:
                    z_step_cache[key] = z
                continue
            g, feats = entry
            pending_keys.append(key)
            pending_graphs.append(g)
            pending_feats.append(feats)

        if pending_graphs:
            z_pending = encode_graph_batch_hgat(enc, pending_graphs, pending_feats)
            for j, key in enumerate(pending_keys):
                z = z_pending[j:j + 1]
                i = seen[key]
                z_unique[i] = z
                if z_step_cache is not None:
                    z_step_cache[key] = z

        z_unique = th.cat(z_unique, dim=0)
        index = th.tensor(idx_map, device=device, dtype=th.long)
        return z_unique.index_select(0, index)

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


def report_graph_cache_coverage(name, graph_cache, cell_types):
    keys = set(str(k) for k in (graph_cache or {}).keys())
    cts = sorted(set(str(x) for x in cell_types))
    missing = [ct for ct in cts if ct not in keys]
    print(f"[Info] {name} cache coverage: {len(cts) - len(missing)}/{len(cts)}")
    if missing:
        print(f"[Warn] {name} cache missing cell_types: {', '.join(missing)}")


def compute_src_loss_weight(epoch_idx, options):
    base_w = float(options.loss_weight_45)
    start = int(getattr(options, "src_loss_anneal_start", -1))
    end = int(getattr(options, "src_loss_anneal_end", -1))
    final_scale = float(getattr(options, "src_loss_final_scale", 1.0))

    if start < 1 or end < 1 or end <= start:
        return base_w

    ep = int(epoch_idx) + 1  # keep parser semantics: 1-based epoch indices
    if ep <= start:
        return base_w
    if ep >= end:
        return base_w * final_scale

    t = float(ep - start) / float(end - start)
    scale = 1.0 + (final_scale - 1.0) * t
    return base_w * scale


class SharedCalibArcRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        design_dim,
        hid=256,
        dropout=0.0,
        *,
        num_pins=1,
        pin_emb_dim=8,
        pol_emb_dim=2,
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
        self.arc_dim = 2 * int(pin_emb_dim) + int(pol_emb_dim)

        if self.use_arc_cond:
            n_pins = max(1, int(num_pins))
            if self.arc_sep_domain_emb:
                self.pin_emb_tgt = nn.Embedding(n_pins, pin_emb_dim)
                self.pin_emb_src = nn.Embedding(n_pins, pin_emb_dim)
                self.pol_emb_tgt = nn.Embedding(2, pol_emb_dim)
                self.pol_emb_src = nn.Embedding(2, pol_emb_dim)
            else:
                self.pin_emb = nn.Embedding(n_pins, pin_emb_dim)
                self.pol_emb = nn.Embedding(2, pol_emb_dim)
            if self.arc_cond_mode == "film":
                self.film_tgt = nn.Linear(self.arc_dim, 2 * self.base_dim)
                self.film_src = nn.Linear(self.arc_dim, 2 * self.base_dim)

        extra_dim_tgt = self.arc_dim if (self.use_arc_cond and self.arc_cond_mode == "concat") else 0
        extra_dim_src = (
            self.arc_dim
            if (self.use_arc_cond and self.src_use_arc_cond and self.arc_cond_mode == "concat")
            else 0
        )
        self.backbone = nn.Sequential(
            nn.Linear(self.base_dim + extra_dim_tgt, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.shared_head_tgt = nn.Linear(hid, 1)
        self.calib_tgt = nn.Linear(hid, 1)

        self.backbone_src = nn.Sequential(
            nn.Linear(self.base_dim + extra_dim_src, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.shared_head_src = nn.Linear(hid, 1)
        self.calib_src = nn.Linear(hid, 1)

    def _arc_feat(self, from_pin_id, to_pin_id, pol_id, node):
        if self.arc_sep_domain_emb:
            if node == "tgt":
                af = self.pin_emb_tgt(from_pin_id)
                at = self.pin_emb_tgt(to_pin_id)
                ap = self.pol_emb_tgt(pol_id)
            elif node == "src":
                af = self.pin_emb_src(from_pin_id)
                at = self.pin_emb_src(to_pin_id)
                ap = self.pol_emb_src(pol_id)
            else:
                raise ValueError(f"Unknown node type: {node}")
        else:
            af = self.pin_emb(from_pin_id)
            at = self.pin_emb(to_pin_id)
            ap = self.pol_emb(pol_id)
        return th.cat([af, at, ap], dim=1)

    def forward(self, x, z, node="tgt", from_pin_id=None, to_pin_id=None, pol_id=None):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h_base = th.cat([x, z], dim=1)

        apply_arc = self.use_arc_cond and (node == "tgt" or (node == "src" and self.src_use_arc_cond))
        h = h_base
        if apply_arc:
            if from_pin_id is None or to_pin_id is None or pol_id is None:
                raise ValueError("Arc conditioning enabled but from_pin_id/to_pin_id/pol_id is missing.")
            arc_feat = self._arc_feat(from_pin_id, to_pin_id, pol_id, node)
            if self.arc_cond_mode == "concat":
                h = th.cat([h_base, arc_feat], dim=1)
            else:
                if node == "tgt":
                    gb = self.film_tgt(arc_feat)
                elif node == "src":
                    gb = self.film_src(arc_feat)
                else:
                    raise ValueError(f"Unknown node type: {node}")
                gamma, beta = gb.chunk(2, dim=1)
                h = h_base * (1.0 + gamma) + beta

        if node == "tgt":
            h = self.backbone(h)
            pred = self.shared_head_tgt(h).squeeze(-1)
            pred = pred + self.calib_tgt(h).squeeze(-1)
        elif node == "src":
            h = self.backbone_src(h)
            pred = self.shared_head_src(h).squeeze(-1)
            pred = pred + self.calib_src(h).squeeze(-1)
        else:
            raise ValueError(f"Unknown node type: {node}")
        return pred


@th.no_grad()
def validate_cell(
    val_dl,
    enc,
    model,
    device,
    design_dim,
    *,
    z_dict=None,
    graph_cache=None,
    dedup=False,
    use_amp=False,
):
    enc.eval()
    model.eval()
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)
    total_loss = 0.0
    total_n = 0
    amp_on = bool(use_amp and device.type == "cuda")
    for xb, yb, cts, fp, tp, pol in val_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        fp = fp.to(device, non_blocking=True)
        tp = tp.to(device, non_blocking=True)
        pol = pol.to(device, non_blocking=True)
        with th.cuda.amp.autocast(enabled=amp_on):
            zb = build_z_batch(
                cts,
                device,
                design_dim,
                z_dict=z_dict,
                graph_cache=graph_cache,
                enc=enc,
                dedup=dedup,
            )
            pred = model(xb, zb, node="tgt", from_pin_id=fp, to_pin_id=tp, pol_id=pol)
            loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        r2_score.update(pred.float(), yb.float())
    avg_loss = total_loss / max(1, total_n)
    val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
    return avg_loss, val_r2


def train_balanced_sep_mlp_shared_calib(options, seed):
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
    use_amp = bool(getattr(options, "use_amp", True))
    amp_on = bool(use_amp and device.type == "cuda")

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

    arc_vocab_scope = str(getattr(options, "arc_vocab_scope", "tgt")).lower()
    if arc_vocab_scope == "src_tgt":
        pin2id = _build_pin_vocab(df_src_use, df_tgt_train)
    elif arc_vocab_scope == "tgt":
        pin2id = _build_pin_vocab(df_tgt_train)
    else:
        raise ValueError(f"Unknown arc_vocab_scope: {arc_vocab_scope}")

    ds_tgt = CellDelayDataset(df_tgt_train, x_mean, x_std, y_mean, y_std, pin2id)
    ds_src = CellDelayDataset(df_src_use, x_mean, x_std, y_mean, y_std, pin2id)
    val_ds = CellDelayDataset(df_tgt_val, x_mean, x_std, y_mean, y_std, pin2id)
    tgt_cts_all = list(ds_tgt.cts) + list(val_ds.cts)
    src_cts_all = list(ds_src.cts)

    def my_collate(batch):
        xs, ys, cts, fps, tps, pols = zip(*batch)
        return th.stack(xs), th.stack(ys), cts, th.stack(fps), th.stack(tps), th.stack(pols)

    batch_size_tgt = max(1, options.batch_size // 2)
    batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

    num_workers = int(getattr(options, "num_workers", 4))
    pin_memory = bool(getattr(options, "pin_memory", device.type == "cuda"))
    persistent_workers = bool(getattr(options, "persistent_workers", num_workers > 0))
    prefetch_factor = int(getattr(options, "prefetch_factor", 2))
    dl_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": my_collate,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    dl_tgt = DataLoader(ds_tgt, batch_size=batch_size_tgt, shuffle=True, **dl_kwargs)
    dl_src = DataLoader(ds_src, batch_size=batch_size_src, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=options.batch_size, shuffle=False, **dl_kwargs)

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

    in_map = {"NET": GRAPH_NET_DIM, "PMOS": GRAPH_MOS_DIM, "NMOS": GRAPH_MOS_DIM}
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
    model = SharedCalibArcRegressor(
        in_dim=options.in_dim,
        design_dim=design_dim,
        hid=hgat_hid,
        dropout=dropout,
        num_pins=len(pin2id),
        pin_emb_dim=getattr(options, "pin_emb_dim", 8),
        pol_emb_dim=getattr(options, "pol_emb_dim", 2),
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
        report_graph_cache_coverage("src", graph_cache_src, src_cts_all)
        report_graph_cache_coverage("tgt", graph_cache_tgt, tgt_cts_all)
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
        report_graph_cache_coverage("src", graph_cache_src, src_cts_all)
        report_graph_cache_coverage("tgt", graph_cache_tgt, tgt_cts_all)
        enc_lr_scale = float(getattr(options, "enc_lr_scale", ENCODER_LR_SCALE_DEFAULT))
        enc_lr = float(options.learning_rate) * enc_lr_scale
        print(f"[Info] Parameter-group LR: enc_lr={enc_lr:.3e}, model_lr={options.learning_rate:.3e}")
        optimizer = th.optim.Adam(
            [
                {"params": enc.parameters(), "lr": enc_lr},
                {"params": model.parameters(), "lr": options.learning_rate},
            ],
            weight_decay=options.weight_decay,
        )

    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)
    scaler = th.cuda.amp.GradScaler(enabled=amp_on)

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
        src_weight_epoch = compute_src_loss_weight(epoch, options)

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt, fp_tgt, tp_tgt, pol_tgt in dl_tgt:
            xb_tgt = xb_tgt.to(device, non_blocking=True)
            yb_tgt = yb_tgt.to(device, non_blocking=True)
            fp_tgt = fp_tgt.to(device, non_blocking=True)
            tp_tgt = tp_tgt.to(device, non_blocking=True)
            pol_tgt = pol_tgt.to(device, non_blocking=True)
            tgt_n = len(yb_tgt)
            src_weighted_loss_sum = 0.0
            src_n_sum = 0
            z_step_cache = {} if (not options.freeze_hgat) else None

            with th.cuda.amp.autocast(enabled=amp_on):
                zb_tgt = build_z_batch(
                    cts_tgt,
                    device,
                    design_dim,
                    z_dict=z_dict_tgt,
                    graph_cache=graph_cache_tgt,
                    enc=enc,
                    dedup=options.dedup_z,
                    z_step_cache=z_step_cache,
                    use_batched_graph_encode=True,
                )
                pred_tgt = model(
                    xb_tgt,
                    zb_tgt,
                    node="tgt",
                    from_pin_id=fp_tgt,
                    to_pin_id=tp_tgt,
                    pol_id=pol_tgt,
                )
                loss_tgt = loss_fn(pred_tgt, yb_tgt)

            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src, fp_src, tp_src, pol_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src, fp_src, tp_src, pol_src = next(dl_src_iter)
                xb_src = xb_src.to(device, non_blocking=True)
                yb_src = yb_src.to(device, non_blocking=True)
                fp_src = fp_src.to(device, non_blocking=True)
                tp_src = tp_src.to(device, non_blocking=True)
                pol_src = pol_src.to(device, non_blocking=True)
                src_n = len(yb_src)
                with th.cuda.amp.autocast(enabled=amp_on):
                    zb_src = build_z_batch(
                        cts_src,
                        device,
                        design_dim,
                        z_dict=z_dict_src,
                        graph_cache=graph_cache_src,
                        enc=enc,
                        dedup=options.dedup_z,
                        z_step_cache=z_step_cache,
                        use_batched_graph_encode=True,
                    )
                    pred_src = model(
                        xb_src,
                        zb_src,
                        node="src",
                        from_pin_id=fp_src,
                        to_pin_id=tp_src,
                        pol_id=pol_src,
                    )
                    loss_src = loss_fn(pred_src, yb_src)
                src_weighted_loss_sum = src_weighted_loss_sum + (loss_src * src_n)
                src_n_sum += src_n

            denom = max(1, tgt_n + src_n_sum)
            with th.cuda.amp.autocast(enabled=amp_on):
                total_loss_batch = (
                    loss_tgt * tgt_n + src_weight_epoch * src_weighted_loss_sum
                ) / denom

            optimizer.zero_grad()
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_loss_batch.item() * len(yb_tgt)
            total_n += len(yb_tgt)
            r2_score.update(pred_tgt.float(), yb_tgt.float())

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
            use_amp=use_amp,
        )

        print(
            f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
            f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, src_w:{src_weight_epoch:.4f}"
        )

        better_r2 = val_r2 > (best_val + 1e-4)
        tie_better = abs(val_r2 - best_val) <= 1e-4 and val_loss < best_val_loss
        if better_r2 or tie_better:
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
                    "pin2id": pin2id,
                    "use_arc_cond": getattr(options, "use_arc_cond", False),
                    "pin_emb_dim": getattr(options, "pin_emb_dim", 8),
                    "pol_emb_dim": getattr(options, "pol_emb_dim", 2),
                    "arc_vocab_scope": arc_vocab_scope,
                    "arc_sep_domain_emb": getattr(options, "arc_sep_domain_emb", False),
                    "arc_cond_mode": getattr(options, "arc_cond_mode", "concat"),
                    "disable_src_arc_cond": getattr(options, "disable_src_arc_cond", False),
                    "scaler_stats": scaler_stats,
                    "y_scaler": y_scaler,
                    "epoch": epoch + 1,
                    "best_val_r2": best_val,
                    "src_loss_weight_epoch": src_weight_epoch,
                    "script": "shared_calib_step8_arc_opt",
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
    stdout_f = "{}/stdout.log".format(options.model_saving_dir)
    stderr_f = "{}/stderr.log".format(options.model_saving_dir)
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")
    os.makedirs(options.model_saving_dir, exist_ok=True)
    os.makedirs(copilot_log_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        run_train_and_report_test(train_balanced_sep_mlp_shared_calib, options, seed, script_name=__file__)