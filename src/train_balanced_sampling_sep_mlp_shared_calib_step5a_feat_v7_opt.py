r"""Balanced sampling with shared trunk + domain calibration heads (no disentanglement losses)."""

import os
import re
import pickle
import random
import numbers
import shutil
import tempfile
import contextlib
import sys
import argparse
import numpy as np
import torch as th
import torch.nn as nn
from torchmetrics import R2Score
from torch.utils.data import Dataset as TorchDataset, DataLoader
import dgl

from options import get_options
import tee

from hgat import HGATDesignEncoder, build_dgl_graph_from_devs_rich
from spi2graph import parse_top_subckt_pins, parse_passive_parasitics_spice

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
CKPT_R2_TIE_EPS_DEFAULT = 1e-4


def _normalize_host_path(path):
    if not path:
        return path
    p = str(path)
    if os.name != "nt":
        return p
    # Allow Windows Python to consume WSL-style absolute paths.
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"
    return p


def _resolve_hgat_net_feat_mode(options):
    enrich_flag = bool(getattr(options, "enrich_parasitic_net_feat", False))
    raw_mode = str(getattr(options, "hgat_net_feat_mode", "auto")).strip().lower()
    if raw_mode in ("", "auto"):
        mode = "parasitic_replace" if enrich_flag else "base"
    else:
        mode = raw_mode
    valid = {"base", "parasitic_replace", "parasitic_append", "parasitic_append_split"}
    if mode not in valid:
        raise ValueError(f"Unsupported --hgat_net_feat_mode={mode}, valid={sorted(valid)}")
    return mode


def _looks_like_dim_map(obj):
    return isinstance(obj, dict) and len(obj) > 0 and all(isinstance(v, numbers.Number) for v in obj.values())


def _looks_like_feat_dict(obj):
    if not isinstance(obj, dict):
        return False
    needed = {"NET", "PMOS", "NMOS"}
    if not needed.issubset(set(obj.keys())):
        return False
    for k in needed:
        v = obj.get(k)
        if isinstance(v, numbers.Number):
            return False
        if th.is_tensor(v):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            continue
        return False
    return True


def _looks_like_graph_obj(obj):
    return hasattr(obj, "to") and hasattr(obj, "ntypes") and hasattr(obj, "num_nodes")


def _looks_like_graph_meta(obj):
    if not isinstance(obj, dict):
        return False
    return ("num_nets" in obj) or ("net_name_to_id" in obj) or ("pin_to_net_id" in obj)


def _move_hgat_feats_to_device(feats, device):
    expected = ("NET", "PMOS", "NMOS")
    if not isinstance(feats, dict):
        raise TypeError(f"HGAT feats must be dict, got: {type(feats)}")
    moved = {}
    for nt in expected:
        if nt not in feats:
            raise KeyError(f"HGAT feats missing key: {nt}, got keys: {list(feats.keys())}")
        v = feats[nt]
        if not th.is_tensor(v):
            v = th.as_tensor(v, dtype=th.float32)
        if v.dim() == 0:
            raise TypeError(f"HGAT feats['{nt}'] is scalar; likely wrong return-order from graph builder.")
        if v.dim() == 1:
            v = v.unsqueeze(0)
        moved[nt] = v.to(device)
    return moved


def _unpack_graph_builder_output(ret, ctype="", domain=""):
    if not isinstance(ret, (tuple, list)):
        ret = (ret,)
    g = None
    feats = None
    g_meta = None
    for x in ret:
        if g is None and _looks_like_graph_obj(x):
            g = x
            continue
        if feats is None and _looks_like_feat_dict(x):
            feats = x
            continue
        if g_meta is None and _looks_like_graph_meta(x):
            g_meta = x
            continue

    if g is None:
        types = [type(x).__name__ for x in ret]
        raise TypeError(f"Graph builder output has no graph object (domain={domain}, cell={ctype}, types={types})")
    if feats is None:
        # Try one last fallback: some environments may place feats where NET/PMOS/NMOS are tensors inside another mapping.
        for x in ret:
            if isinstance(x, dict):
                if "feats" in x and _looks_like_feat_dict(x["feats"]):
                    feats = x["feats"]
                    break
        if feats is None:
            types = [type(x).__name__ for x in ret]
            raise TypeError(f"Graph builder output has no valid feats dict (domain={domain}, cell={ctype}, types={types})")

    if g_meta is None:
        g_meta = {}
    return g, feats, g_meta


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
    if "from_pin" in df.columns:
        from_pins = df["from_pin"].fillna("").astype(str).values
    else:
        from_pins = np.array([""] * len(df), dtype=object)
    if "to_pin" in df.columns:
        to_pins = df["to_pin"].fillna("").astype(str).values
    else:
        to_pins = np.array([""] * len(df), dtype=object)
    if "topology_group" in df.columns:
        topo_groups = df["topology_group"].fillna("__UNK__").astype(str).values
    else:
        topo_groups = np.array(["__UNK__"] * len(df), dtype=object)
    return x, y, cts, from_pins, to_pins, topo_groups


class CellDelayDataset(TorchDataset):
    def __init__(
        self,
        df,
        x_mean,
        x_std,
        y_mean,
        y_std,
        sample_weights=None,
        group_col="group_id",
        topo_to_id=None,
    ):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts, self.from_pins, self.to_pins, self.topo_groups = _norm_xy(
            self.df, x_mean, x_std, y_mean, y_std
        )
        if group_col in self.df.columns:
            self.group_ids = self.df[group_col].fillna("__NA__").astype(str).values
        else:
            self.group_ids = np.array(["__NA__"] * len(self.x), dtype=object)
        if topo_to_id is None:
            topo_to_id = {"__UNK__": 0}
        self.topo_to_id = dict(topo_to_id)
        self.topo_ids = np.array([int(self.topo_to_id.get(str(t), 0)) for t in self.topo_groups], dtype=np.int64)
        if sample_weights is None:
            self.sample_weights = np.ones((len(self.x),), dtype=np.float32)
        else:
            sw = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
            if len(sw) != len(self.x):
                raise ValueError(f"sample_weights length mismatch: {len(sw)} vs {len(self.x)}")
            self.sample_weights = sw

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            th.from_numpy(self.x[i]),
            th.tensor(self.y[i]),
            self.cts[i],
            self.from_pins[i],
            self.to_pins[i],
            int(self.topo_ids[i]),
            self.group_ids[i],
            float(self.sample_weights[i]),
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


def build_dgl_graph_from_devs_step5(
    devs,
    top_pins,
    passives=None,
    return_meta=False,
    enrich_parasitic_net_feat=False,
    include_body_edges=False,
    net_feat_mode="auto",
    par_cap_weight=0.5,
):
    return build_dgl_graph_from_devs_rich(
        devs,
        top_pins,
        passives=passives,
        return_meta=return_meta,
        enrich_parasitic_net_feat=enrich_parasitic_net_feat,
        include_body_edges=include_body_edges,
        net_feat_mode=net_feat_mode,
        par_cap_weight=par_cap_weight,
    )


def build_src_graph_cache(
    data_dir,
    meta,
    device,
    enrich_parasitic_net_feat=False,
    include_body_edges=False,
    net_feat_mode="auto",
    par_cap_weight=0.5,
):
    mapping = meta.get("src_spi_by_cell", {})
    if not mapping:
        print("[Warn] meta has no src_spi_by_cell")
        return {}

    graph_cache = {}
    data_dir = _normalize_host_path(data_dir)
    for ctype, sp_path in mapping.items():
        if not sp_path:
            continue
        sp_path = _normalize_host_path(sp_path)
        if not os.path.exists(sp_path):
            cand = os.path.join(data_dir, sp_path)
            if os.path.exists(cand):
                sp_path = cand
            else:
                continue
        sp_text = open(sp_path, "r", encoding="utf-8", errors="ignore").read()
        devs = parse_transistors_spice_rich(sp_text)
        passives = parse_passive_parasitics_spice(sp_text)
        _, pins = parse_top_subckt_pins(sp_text)
        if not devs:
            continue
        out = build_dgl_graph_from_devs_step5(
            devs,
            pins,
            passives=passives,
            return_meta=True,
            enrich_parasitic_net_feat=enrich_parasitic_net_feat,
            include_body_edges=include_body_edges,
            net_feat_mode=net_feat_mode,
            par_cap_weight=par_cap_weight,
        )
        g, feats, g_meta = _unpack_graph_builder_output(out, ctype=ctype, domain="src")
        graph_cache[ctype] = (g.to(device), _move_hgat_feats_to_device(feats, device), g_meta)
    print(f"[Info] Cached source graphs: {len(graph_cache)}")
    return graph_cache


def build_tgt_graph_cache(
    data_dir,
    meta,
    device,
    enrich_parasitic_net_feat=False,
    include_body_edges=False,
    net_feat_mode="auto",
    par_cap_weight=0.5,
):
    mapping = meta.get("tgt_subckt_by_cell", {})
    if not mapping:
        print("[Warn] meta has no tgt_subckt_by_cell")
        return {}

    data_dir = _normalize_host_path(data_dir)
    tgt_spice = _normalize_host_path(meta.get("tgt_sp_file", ""))
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
        passives = parse_passive_parasitics_spice(sub_txt)
        _, pins = parse_top_subckt_pins(sub_txt)
        if not devs:
            continue
        out = build_dgl_graph_from_devs_step5(
            devs,
            pins,
            passives=passives,
            return_meta=True,
            enrich_parasitic_net_feat=enrich_parasitic_net_feat,
            include_body_edges=include_body_edges,
            net_feat_mode=net_feat_mode,
            par_cap_weight=par_cap_weight,
        )
        g, feats, g_meta = _unpack_graph_builder_output(out, ctype=ctype, domain="tgt")
        graph_cache[ctype] = (g.to(device), _move_hgat_feats_to_device(feats, device), g_meta)
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


def _norm_pin_name(pin):
    if pin is None:
        return ""
    s = str(pin).strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s.upper()


def _arc_cache_key(ct, from_pin=None, to_pin=None):
    ct_key = str(ct)
    fp = _norm_pin_name(from_pin)
    tp = _norm_pin_name(to_pin)
    if fp == "" and tp == "":
        return ct_key
    return f"{ct_key}||{fp}->{tp}"


def _build_net_focus_from_meta(graph_meta, from_pin, to_pin, device):
    if not graph_meta:
        return None
    num_nets = int(graph_meta.get("num_nets", 0) or 0)
    if num_nets <= 0:
        return None
    pin_to_net = graph_meta.get("pin_to_net_id", {}) or {}
    net_name_to_id = graph_meta.get("net_name_to_id", {}) or {}
    net_ids = []
    for pin in (from_pin, to_pin):
        p = _norm_pin_name(pin)
        if p == "":
            continue
        nid = pin_to_net.get(p)
        if nid is None:
            nid = net_name_to_id.get(p)
        if nid is None:
            continue
        nid = int(nid)
        if 0 <= nid < num_nets:
            net_ids.append(nid)
    if len(net_ids) == 0:
        return None
    focus = th.zeros((num_nets,), dtype=th.float32, device=device)
    focus[list(dict.fromkeys(net_ids))] = 1.0
    return focus


def _effective_z_dim(base_dim, dual_readout=False, dual_merge="concat"):
    if bool(dual_readout) and str(dual_merge).strip().lower() == "concat":
        return int(base_dim) * 2
    return int(base_dim)


def _encode_hgat_z(
    enc,
    g,
    feats,
    net_focus=None,
    *,
    dual_readout=False,
    dual_merge="concat",
):
    z_local = enc(g, feats, net_focus=net_focus)
    if z_local.dim() == 1:
        z_local = z_local.unsqueeze(0)
    if not dual_readout:
        return z_local

    z_global = enc(g, feats, net_focus=None)
    if z_global.dim() == 1:
        z_global = z_global.unsqueeze(0)

    merge = str(dual_merge).strip().lower()
    if merge == "concat":
        return th.cat([z_global, z_local], dim=1)
    if merge == "mean":
        return 0.5 * (z_global + z_local)
    raise ValueError(f"Unsupported hgat_dual_merge={dual_merge}")


def build_z_batch(
    cts,
    device,
    design_dim,
    *,
    from_pins=None,
    to_pins=None,
    z_dict=None,
    graph_cache=None,
    enc=None,
    dedup=False,
    z_step_cache=None,
    use_batched_graph_encode=False,
    dual_readout=False,
    dual_merge="concat",
    disable_graph_feature=False,
):
    if disable_graph_feature:
        return th.zeros((len(cts), design_dim), device=device)
    if z_dict is None and (graph_cache is None or enc is None):
        raise ValueError("build_z_batch requires z_dict or (graph_cache + enc).")
    use_arc = (from_pins is not None) or (to_pins is not None)
    if use_arc:
        if from_pins is None:
            from_pins = [""] * len(cts)
        if to_pins is None:
            to_pins = [""] * len(cts)
        if len(from_pins) != len(cts) or len(to_pins) != len(cts):
            raise ValueError("from_pins/to_pins length must match cts length.")

    def _get_z(ct, from_pin=None, to_pin=None):
        key = _arc_cache_key(ct, from_pin, to_pin) if use_arc else str(ct)
        if z_step_cache is not None and key in z_step_cache:
            return z_step_cache[key]
        if z_dict is not None and (not use_arc):
            z = z_dict.get(str(ct))
            if z is None:
                z = th.zeros(1, design_dim, device=device)
            if z_step_cache is not None:
                z_step_cache[key] = z
            return z
        if z_dict is not None and use_arc and key in z_dict:
            z = z_dict[key]
            if z_step_cache is not None:
                z_step_cache[key] = z
            return z
        entry = graph_cache.get(key) if graph_cache is not None else None
        if entry is None and graph_cache is not None:
            entry = graph_cache.get(str(ct))
        if entry is None:
            # Fallback to cell-level precomputed z if provided.
            if z_dict is not None and str(ct) in z_dict:
                z = z_dict[str(ct)]
                if z_step_cache is not None:
                    z_step_cache[key] = z
                return z
            z = th.zeros(1, design_dim, device=device)
            if z_step_cache is not None:
                z_step_cache[key] = z
            return z
        if len(entry) >= 3:
            g, feats, graph_meta = entry[0], entry[1], entry[2]
        else:
            g, feats = entry
            graph_meta = None
        net_focus = _build_net_focus_from_meta(graph_meta, from_pin, to_pin, device) if use_arc else None
        z = _encode_hgat_z(
            enc,
            g,
            feats,
            net_focus=net_focus,
            dual_readout=dual_readout,
            dual_merge=dual_merge,
        )
        # For arc-aware path (especially freeze HGAT), memoize by arc-key
        # so each unique (cell_type, from_pin, to_pin) is encoded once.
        if use_arc and z_dict is not None:
            z_dict[key] = z.detach()
        if z_step_cache is not None:
            z_step_cache[key] = z
        return z

    if (
        (not use_arc)
        and use_batched_graph_encode
        and (not dual_readout)
        and z_dict is None
        and graph_cache is not None
        and enc is not None
    ):
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
            g, feats = entry[0], entry[1]
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
        for i, ct in enumerate(cts):
            fp = from_pins[i] if use_arc else None
            tp = to_pins[i] if use_arc else None
            key = _arc_cache_key(ct, fp, tp) if use_arc else str(ct)
            idx = seen.get(key)
            if idx is None:
                idx = len(uniq)
                seen[key] = idx
                uniq.append((ct, fp, tp))
            idx_map.append(idx)
        if not uniq:
            return th.zeros((0, design_dim), device=device)
        z_unique = th.cat([_get_z(ct, fp, tp) for ct, fp, tp in uniq], dim=0)
        index = th.tensor(idx_map, device=device, dtype=th.long)
        return z_unique.index_select(0, index)

    if len(cts) == 0:
        return th.zeros((0, design_dim), device=device)
    if use_arc:
        return th.cat([_get_z(ct, from_pins[i], to_pins[i]) for i, ct in enumerate(cts)], dim=0)
    return th.cat([_get_z(ct) for ct in cts], dim=0)


def precompute_z_from_graph_cache(graph_cache, enc, *, dual_readout=False, dual_merge="concat"):
    z_dict = {}
    enc.eval()
    with th.no_grad():
        for ct, entry in graph_cache.items():
            g, feats = entry[0], entry[1]
            z = _encode_hgat_z(
                enc,
                g,
                feats,
                net_focus=None,
                dual_readout=dual_readout,
                dual_merge=dual_merge,
            )
            z_dict[str(ct)] = z
    return z_dict


def _filter_compatible_state_dict(model, state_dict):
    model_sd = model.state_dict()
    keep = {}
    dropped_shape = []
    dropped_missing = []
    for k, v in state_dict.items():
        if k not in model_sd:
            dropped_missing.append(k)
            continue
        if model_sd[k].shape != v.shape:
            dropped_shape.append(k)
            continue
        keep[k] = v
    return keep, dropped_missing, dropped_shape


def _clone_state_dict_to_cpu(state_dict):
    cloned = {}
    for k, v in state_dict.items():
        if th.is_tensor(v):
            cloned[k] = v.detach().cpu().clone()
        else:
            cloned[k] = v
    return cloned


def _load_hgat_encoder_state_dict(enc, enc_state_dict):
    """Load HGAT state dict with compatibility handling across script versions."""
    norm_key_pat = re.compile(r"^layers\.(\d+)\.norm\.([A-Za-z0-9_]+)\.")
    device = next(enc.parameters()).device

    # Some older checkpoints contain node-type LN params that are missing in a fresh model.
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

    filtered_sd, dropped_missing, dropped_shape = _filter_compatible_state_dict(enc, enc_state_dict)
    if len(filtered_sd) == 0:
        print("[Warn] HGAT checkpoint matched 0 parameters; skip loading encoder checkpoint.")
        return False

    load_res = enc.load_state_dict(filtered_sd, strict=False)
    if dropped_missing:
        print(f"[Warn] HGAT checkpoint keys not in model: {len(dropped_missing)}")
    if dropped_shape:
        print(f"[Warn] HGAT checkpoint keys shape-mismatch: {len(dropped_shape)}")
    if load_res.missing_keys:
        print(f"[Info] HGAT model missing keys after partial load: {len(load_res.missing_keys)}")
    if load_res.unexpected_keys:
        print(f"[Info] HGAT unexpected keys after partial load: {len(load_res.unexpected_keys)}")
    print(f"[Info] Loaded HGAT encoder params: {len(filtered_sd)} tensors")
    return True


def maybe_load_hgat_encoder_checkpoint(enc, options, device):
    """Optionally initialize HGAT encoder from external checkpoint."""
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
    enc_state_dict = None
    if isinstance(ckpt, dict) and "enc" in ckpt and isinstance(ckpt["enc"], dict):
        enc_state_dict = ckpt["enc"]
        src = "ckpt['enc']"
    elif isinstance(ckpt, dict):
        # Support plain state_dict formats.
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


def maybe_load_full_checkpoint(enc, model, options, device):
    ckpt_path = getattr(options, "init_full_ckpt_path", None)
    if ckpt_path is None or str(ckpt_path).strip() == "":
        return False
    ckpt_path = os.path.expanduser(str(ckpt_path))
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"[Warn] init_full_ckpt_path not found: {ckpt_path}")
        return False

    ckpt = th.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        print(f"[Warn] Unsupported full ckpt format: {type(ckpt)}")
        return False
    enc_sd = ckpt.get("enc")
    model_sd = ckpt.get("model")
    if not isinstance(enc_sd, dict) or not isinstance(model_sd, dict):
        print("[Warn] full ckpt must contain dict keys: enc/model")
        return False

    ok_enc = _load_hgat_encoder_state_dict(enc, enc_sd)
    model_sd_fit, mm_missing, mm_shape = _filter_compatible_state_dict(model, model_sd)
    if len(model_sd_fit) == 0:
        print("[Warn] Full-ckpt model state matched 0 tensors.")
        return False
    load_res = model.load_state_dict(model_sd_fit, strict=False)
    if mm_missing:
        print(f"[Warn] Full-ckpt model keys not in model: {len(mm_missing)}")
    if mm_shape:
        print(f"[Warn] Full-ckpt model keys shape-mismatch: {len(mm_shape)}")
    if load_res.missing_keys:
        print(f"[Info] Full-ckpt model missing keys after partial load: {len(load_res.missing_keys)}")
    if load_res.unexpected_keys:
        print(f"[Info] Full-ckpt model unexpected keys after partial load: {len(load_res.unexpected_keys)}")
    if ok_enc:
        print(f"[Info] Full checkpoint initialized (enc+model): {ckpt_path}")
    return ok_enc


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


def _infer_supervision_ratio(options, df_tgt_train, df_tgt_val, df_tgt_test):
    ds_name = str(getattr(options, "dataset_pkl_name", "") or "")
    m = re.search(r"sup(\d+)p", ds_name)
    if m:
        pct = float(m.group(1))
        if pct > 0:
            return pct / 100.0, "dataset_name"

    n_train = int(len(df_tgt_train) if df_tgt_train is not None else 0)
    n_val = int(len(df_tgt_val) if df_tgt_val is not None else 0)
    n_test = int(len(df_tgt_test) if df_tgt_test is not None else 0)
    denom = max(1, n_train + n_val + n_test)
    return float(n_train) / float(denom), "split_count"


def apply_auto_transfer_by_sup(options, df_tgt_train, df_tgt_val, df_tgt_test):
    if not bool(getattr(options, "auto_transfer_by_sup", False)):
        return

    sup_ratio, src = _infer_supervision_ratio(options, df_tgt_train, df_tgt_val, df_tgt_test)
    low = max(1e-6, float(getattr(options, "auto_sup_low", 0.01)))
    high = max(low + 1e-6, float(getattr(options, "auto_sup_high", 0.20)))
    norm = (sup_ratio - low) / (high - low)
    norm = float(np.clip(norm, 0.0, 1.0))
    curve_power = max(1e-6, float(getattr(options, "auto_sup_curve_power", 1.0)))
    norm_eff = float(np.clip(norm ** curve_power, 0.0, 1.0))

    src_w_low = float(getattr(options, "auto_src_w_low", 1.0))
    src_w_high = float(getattr(options, "auto_src_w_high", 0.35))
    src_base_w = src_w_low + (src_w_high - src_w_low) * norm_eff

    src_fs_low = float(getattr(options, "auto_src_final_scale_low", 0.70))
    src_fs_high = float(getattr(options, "auto_src_final_scale_high", 0.25))
    src_final_scale = src_fs_low + (src_fs_high - src_fs_low) * norm_eff

    tgt_w_low = float(getattr(options, "auto_tgt_w_low", 1.20))
    tgt_w_high = float(getattr(options, "auto_tgt_w_high", 1.00))
    tgt_w = tgt_w_low + (tgt_w_high - tgt_w_low) * norm_eff

    options.loss_weight_45 = float(src_base_w)
    options.src_loss_final_scale = float(src_final_scale)
    options.target_loss_weight = float(tgt_w)

    # Optional hard overrides for high-supervision regime.
    high_cutoff = float(getattr(options, "auto_high_sup_cutoff", -1.0))
    hard_applied = False
    if high_cutoff >= 0.0 and sup_ratio >= high_cutoff:
        hard_src_w = float(getattr(options, "auto_high_sup_src_w", -1.0))
        hard_src_fs = float(getattr(options, "auto_high_sup_src_final_scale", -1.0))
        hard_tgt_w = float(getattr(options, "auto_high_sup_tgt_w", -1.0))
        hard_anneal_end = int(getattr(options, "auto_high_sup_src_anneal_end", -1))
        hard_enc_lr_scale = float(getattr(options, "auto_high_sup_enc_lr_scale", -1.0))
        if hard_src_w >= 0.0:
            options.loss_weight_45 = float(hard_src_w)
            hard_applied = True
        if hard_src_fs >= 0.0:
            options.src_loss_final_scale = float(hard_src_fs)
            hard_applied = True
        if hard_tgt_w >= 0.0:
            options.target_loss_weight = float(hard_tgt_w)
            hard_applied = True
        if hard_enc_lr_scale >= 0.0:
            options.enc_lr_scale = float(hard_enc_lr_scale)
            hard_applied = True
        if hard_anneal_end >= 0:
            options.src_loss_anneal_end = int(hard_anneal_end)
            hard_applied = True
        if bool(getattr(options, "auto_high_sup_disable_domain_calibration", False)):
            options.disable_domain_calibration = True
            hard_applied = True
        if bool(getattr(options, "auto_high_sup_disable_topology_expert", False)):
            options.use_topology_expert = False
            hard_applied = True
        if bool(getattr(options, "auto_high_sup_unfreeze_hgat", False)):
            options.freeze_hgat = False
            hard_applied = True

    print(
        f"[Info] Auto transfer by supervision: enabled "
        f"(sup_ratio={sup_ratio:.4f}, source={src}, norm={norm:.3f}, "
        f"curve_power={curve_power:.3f}, norm_eff={norm_eff:.3f})"
    )
    print(
        f"[Info] Auto transfer params -> "
        f"loss_weight_45={options.loss_weight_45:.4f}, "
        f"src_loss_final_scale={options.src_loss_final_scale:.4f}, "
        f"target_loss_weight={options.target_loss_weight:.4f}"
    )
    if hard_applied:
        print(
            f"[Info] Auto transfer hard-override active at sup_ratio={sup_ratio:.4f} "
            f"(cutoff={high_cutoff:.4f})"
        )
        if bool(getattr(options, "disable_domain_calibration", False)):
            print("[Info] Auto transfer hard-override: disable_domain_calibration=True")
        if not bool(getattr(options, "use_topology_expert", True)):
            print("[Info] Auto transfer hard-override: use_topology_expert=False")
        if not bool(getattr(options, "freeze_hgat", True)):
            print("[Info] Auto transfer hard-override: freeze_hgat=False")
        if float(getattr(options, "auto_high_sup_enc_lr_scale", -1.0)) >= 0.0:
            print(f"[Info] Auto transfer hard-override: enc_lr_scale={float(options.enc_lr_scale):.4f}")


def build_target_group_sample_weights(df_tgt_train, options):
    n = len(df_tgt_train)
    if n <= 0:
        return np.ones((0,), dtype=np.float32), None
    if not bool(getattr(options, "tgt_group_reweight", False)):
        return np.ones((n,), dtype=np.float32), None

    col = str(getattr(options, "tgt_group_reweight_col", "group_id"))
    if col not in df_tgt_train.columns:
        print(f"[Warn] --tgt_group_reweight enabled but column '{col}' not found; fallback to uniform weights.")
        return np.ones((n,), dtype=np.float32), None

    power = max(0.0, float(getattr(options, "tgt_group_reweight_power", 0.5)))
    w_min = max(0.0, float(getattr(options, "tgt_group_reweight_min", 0.2)))
    w_max = max(w_min, float(getattr(options, "tgt_group_reweight_max", 3.0)))

    groups = df_tgt_train[col].fillna("__NA__").astype(str)
    group_counts = groups.value_counts(dropna=False)
    if group_counts.empty:
        return np.ones((n,), dtype=np.float32), None

    max_cnt = max(1.0, float(group_counts.max()))
    group_w_map = {}
    for gid, cnt in group_counts.items():
        cntf = max(1.0, float(cnt))
        raw = (max_cnt / cntf) ** power if power > 0 else 1.0
        group_w_map[str(gid)] = float(np.clip(raw, w_min, w_max))

    sample_w = groups.map(group_w_map).astype(np.float32).to_numpy()
    mean_w = float(sample_w.mean()) if sample_w.size > 0 else 1.0
    if mean_w > 1e-12:
        sample_w = sample_w / mean_w

    info = {
        "col": col,
        "power": power,
        "clip_min": w_min,
        "clip_max": w_max,
        "num_groups": int(len(group_counts)),
        "sample_w_min": float(sample_w.min()) if sample_w.size > 0 else 1.0,
        "sample_w_max": float(sample_w.max()) if sample_w.size > 0 else 1.0,
        "sample_w_mean": float(sample_w.mean()) if sample_w.size > 0 else 1.0,
    }
    return sample_w, info


def build_topology_vocab(*dfs):
    topo_set = set()
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        if "topology_group" in df.columns:
            vals = df["topology_group"].fillna("__UNK__").astype(str).unique().tolist()
            topo_set.update(vals)
    topo_list = sorted(topo_set)
    topo_to_id = {"__UNK__": 0}
    for t in topo_list:
        if t == "__UNK__":
            continue
        topo_to_id[t] = len(topo_to_id)
    return topo_to_id


def build_train_loss_fn(options):
    loss_type = str(getattr(options, "train_loss_type", "mse")).strip().lower()
    if loss_type in ("huber", "smooth_l1", "smoothl1"):
        huber_delta = float(getattr(options, "huber_delta", 1.0))
        print(f"[Info] Training loss: Huber(delta={huber_delta})")
        return nn.HuberLoss(delta=huber_delta)
    print("[Info] Training loss: MSE")
    return nn.MSELoss()


def compute_loss_with_optional_weights(loss_fn, pred, target, sample_w=None):
    if sample_w is None:
        return loss_fn(pred, target)

    w = sample_w.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    if w.numel() != pred.numel():
        raise ValueError(f"sample_w size mismatch: {w.numel()} vs pred {pred.numel()}")
    w_sum = w.sum().clamp_min(1e-12)

    if isinstance(loss_fn, nn.MSELoss):
        per_elem = (pred - target).pow(2)
    elif isinstance(loss_fn, nn.HuberLoss):
        delta = float(getattr(loss_fn, "delta", 1.0))
        try:
            per_elem = th.nn.functional.huber_loss(pred, target, delta=delta, reduction="none")
        except Exception:
            per_elem = th.nn.functional.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    else:
        per_elem = (pred - target).pow(2)
    return (per_elem * w).sum() / w_sum


def clip_grad_by_optimizer(optimizer, max_norm):
    params = []
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            if p is not None and p.grad is not None:
                params.append(p)
    if not params:
        return 0.0
    grad_norm = th.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    try:
        return float(grad_norm.item())
    except Exception:
        return float(grad_norm)


def build_scheduler(optimizer, options):
    sched = str(getattr(options, "lr_scheduler", "none")).strip().lower()
    if sched in ("cosine_wr", "cosine", "cosineannealingwarmrestarts"):
        t0_raw = int(getattr(options, "cosine_t0", 0))
        if t0_raw <= 0:
            t0 = max(1, int(options.num_epoch) // 4)
        else:
            t0 = t0_raw
        t_mult = int(getattr(options, "cosine_t_mult", 2))
        eta_min = float(getattr(options, "cosine_eta_min", 1e-6))
        t0 = max(1, t0)
        t_mult = max(1, t_mult)
        eta_min = max(0.0, eta_min)
        scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=t_mult,
            eta_min=eta_min,
        )
        print(f"[Info] Scheduler: CosineAnnealingWarmRestarts(T_0={t0}, T_mult={t_mult}, eta_min={eta_min})")
        return scheduler
    if sched in ("plateau", "reduce_on_plateau"):
        factor = float(getattr(options, "plateau_factor", 0.5))
        patience = int(getattr(options, "plateau_patience", 5))
        threshold = float(getattr(options, "plateau_threshold", 1e-4))
        min_lr = float(getattr(options, "plateau_min_lr", 1e-6))
        factor = min(max(factor, 1e-3), 0.999)
        patience = max(1, patience)
        threshold = max(0.0, threshold)
        min_lr = max(0.0, min_lr)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
        )
        print(
            f"[Info] Scheduler: ReduceLROnPlateau(mode=min, factor={factor}, "
            f"patience={patience}, threshold={threshold}, min_lr={min_lr})"
        )
        return scheduler
    print("[Info] Scheduler: none")
    return None


class SharedCalibRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        design_dim,
        hid=256,
        dropout=0.0,
        use_calib_mlp=False,
        calib_mlp_hid=0,
        calib_mlp_dropout=-1.0,
        use_topology_expert=False,
        num_topologies=1,
        topology_expert_dim=16,
        topology_expert_hidden=64,
        disable_domain_calibration=False,
        use_disentangle=False,
        disentangle_hidden=0,
        topo_moe_k=1,
        topo_moe_temp=1.0,
        calib_branch_scale=1.0,
        topo_branch_scale=1.0,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim + design_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.use_disentangle = bool(use_disentangle)
        self.disentangle_hidden = int(disentangle_hidden) if int(disentangle_hidden) > 0 else int(hid)
        self.disentangle_hidden = max(8, self.disentangle_hidden)
        if self.use_disentangle:
            self.inv_proj = nn.Sequential(
                nn.Linear(hid, self.disentangle_hidden),
                nn.ReLU(),
            )
            self.dom_proj = nn.Sequential(
                nn.Linear(hid, self.disentangle_hidden),
                nn.ReLU(),
            )
            pred_hid = self.disentangle_hidden
        else:
            self.inv_proj = None
            self.dom_proj = None
            pred_hid = hid

        self.shared_head = nn.Linear(pred_hid, 1)
        use_calib_mlp = bool(use_calib_mlp)
        calib_hid = int(calib_mlp_hid) if int(calib_mlp_hid) > 0 else int(pred_hid)
        calib_drop = float(calib_mlp_dropout) if float(calib_mlp_dropout) >= 0 else float(dropout)
        if use_calib_mlp:
            self.calib_tgt = nn.Sequential(
                nn.Linear(pred_hid, calib_hid),
                nn.ReLU(),
                nn.Dropout(calib_drop),
                nn.Linear(calib_hid, 1),
            )
            self.calib_src = nn.Sequential(
                nn.Linear(pred_hid, calib_hid),
                nn.ReLU(),
                nn.Dropout(calib_drop),
                nn.Linear(calib_hid, 1),
            )
        else:
            self.calib_tgt = nn.Linear(pred_hid, 1)
            self.calib_src = nn.Linear(pred_hid, 1)
        self.disable_domain_calibration = bool(disable_domain_calibration)
        self.use_topology_expert = bool(use_topology_expert)
        self.calib_branch_scale = max(0.0, float(calib_branch_scale))
        self.topo_branch_scale = max(0.0, float(topo_branch_scale))
        self.num_topologies = max(1, int(num_topologies))
        self.topo_moe_k = max(1, int(topo_moe_k))
        self.topo_moe_temp = max(1e-3, float(topo_moe_temp))
        if self.use_topology_expert:
            topo_dim = max(4, int(topology_expert_dim))
            topo_hid = max(8, int(topology_expert_hidden))
            self.topo_emb = nn.Embedding(self.num_topologies, topo_dim)
            topo_in_dim = pred_hid + topo_dim
            if self.topo_moe_k > 1:
                self.topo_gate_tgt = nn.Linear(topo_in_dim, self.topo_moe_k)
                self.topo_gate_src = nn.Linear(topo_in_dim, self.topo_moe_k)
                self.topo_experts_tgt = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(topo_in_dim, topo_hid),
                            nn.ReLU(),
                            nn.Linear(topo_hid, 1),
                        )
                        for _ in range(self.topo_moe_k)
                    ]
                )
                self.topo_experts_src = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(topo_in_dim, topo_hid),
                            nn.ReLU(),
                            nn.Linear(topo_hid, 1),
                        )
                        for _ in range(self.topo_moe_k)
                    ]
                )
                self.topo_res_tgt = None
                self.topo_res_src = None
            else:
                self.topo_res_tgt = nn.Sequential(
                    nn.Linear(topo_in_dim, topo_hid),
                    nn.ReLU(),
                    nn.Linear(topo_hid, 1),
                )
                self.topo_res_src = nn.Sequential(
                    nn.Linear(topo_in_dim, topo_hid),
                    nn.ReLU(),
                    nn.Linear(topo_hid, 1),
                )
                self.topo_gate_tgt = None
                self.topo_gate_src = None
                self.topo_experts_tgt = None
                self.topo_experts_src = None
        else:
            self.topo_emb = None
            self.topo_res_tgt = None
            self.topo_res_src = None
            self.topo_gate_tgt = None
            self.topo_gate_src = None
            self.topo_experts_tgt = None
            self.topo_experts_src = None

    def encode_hidden(self, x, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h_backbone = self.backbone(th.cat([x, z], dim=1))
        if self.use_disentangle:
            h_inv = self.inv_proj(h_backbone)
            h_dom = self.dom_proj(h_backbone)
            h_shared = h_inv
            h_calib = h_dom
            h_topo = h_dom
        else:
            h_inv = None
            h_dom = None
            h_shared = h_backbone
            h_calib = h_backbone
            h_topo = h_backbone
        return {
            "h_backbone": h_backbone,
            "h_inv": h_inv,
            "h_dom": h_dom,
            "h_shared": h_shared,
            "h_calib": h_calib,
            "h_topo": h_topo,
        }

    def _apply_topology_residual(self, h_topo, topo_ids, node):
        if (not self.use_topology_expert) or topo_ids is None:
            return th.zeros((h_topo.shape[0],), device=h_topo.device, dtype=h_topo.dtype)
        topo_ids = topo_ids.to(device=h_topo.device, dtype=th.long).clamp_(0, self.num_topologies - 1)
        topo_feat = self.topo_emb(topo_ids)
        topo_in = th.cat([h_topo, topo_feat], dim=1)
        if self.topo_moe_k > 1:
            if node == "tgt":
                gate_logits = self.topo_gate_tgt(topo_in) / self.topo_moe_temp
                gate = th.softmax(gate_logits, dim=1)
                exp_out = th.stack([m(topo_in).squeeze(-1) for m in self.topo_experts_tgt], dim=1)
            elif node == "src":
                gate_logits = self.topo_gate_src(topo_in) / self.topo_moe_temp
                gate = th.softmax(gate_logits, dim=1)
                exp_out = th.stack([m(topo_in).squeeze(-1) for m in self.topo_experts_src], dim=1)
            else:
                raise ValueError(f"Unknown node type: {node}")
            res = (gate * exp_out).sum(dim=1)
        elif node == "tgt":
            res = self.topo_res_tgt(topo_in).squeeze(-1)
        elif node == "src":
            res = self.topo_res_src(topo_in).squeeze(-1)
        else:
            raise ValueError(f"Unknown node type: {node}")
        return self.topo_branch_scale * res

    def forward_with_aux(self, x, z, node="tgt", topo_ids=None):
        aux = self.encode_hidden(x, z)
        h_shared = aux["h_shared"]
        h_calib = aux["h_calib"]
        h_topo = aux["h_topo"]

        pred = self.shared_head(h_shared).squeeze(-1)
        if not self.disable_domain_calibration:
            if node == "tgt":
                pred = pred + self.calib_branch_scale * self.calib_tgt(h_calib).squeeze(-1)
            elif node == "src":
                pred = pred + self.calib_branch_scale * self.calib_src(h_calib).squeeze(-1)
            else:
                raise ValueError(f"Unknown node type: {node}")
        elif node not in ("tgt", "src"):
            raise ValueError(f"Unknown node type: {node}")
        pred = pred + self._apply_topology_residual(h_topo, topo_ids, node)
        return pred, aux

    def forward(self, x, z, node="tgt", topo_ids=None):
        pred, _ = self.forward_with_aux(x, z, node=node, topo_ids=topo_ids)
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
    dual_readout=False,
    dual_merge="concat",
    y_mean=0.0,
    y_std=1.0,
    mape_eps=1e-6,
    compute_metrics=True,
    disable_graph_feature=False,
):
    enc.eval()
    model.eval()
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device) if compute_metrics else None
    total_loss = 0.0
    total_n = 0
    total_abs_err = 0.0
    total_abs_pct = 0.0
    amp_on = bool(use_amp and device.type == "cuda")
    z_eval_cache = {}
    for batch in val_dl:
        if len(batch) == 8:
            xb, yb, cts, from_pins, to_pins, topo_ids, _, _ = batch
        elif len(batch) == 7:
            xb, yb, cts, from_pins, to_pins, _, _ = batch
            topo_ids = None
        elif len(batch) == 5:
            xb, yb, cts, from_pins, to_pins = batch
            topo_ids = None
        else:
            raise ValueError(f"Unexpected batch tuple length in validate_cell: {len(batch)}")
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if topo_ids is not None:
            topo_ids = topo_ids.to(device, non_blocking=True)
        with th.cuda.amp.autocast(enabled=amp_on):
            zb = build_z_batch(
                cts,
                device,
                design_dim,
                from_pins=from_pins,
                to_pins=to_pins,
                z_dict=z_dict,
                graph_cache=graph_cache,
                enc=enc,
                dedup=dedup,
                z_step_cache=z_eval_cache,
                use_batched_graph_encode=True,
                dual_readout=dual_readout,
                dual_merge=dual_merge,
                disable_graph_feature=disable_graph_feature,
            )
            pred = model(xb, zb, node="tgt", topo_ids=topo_ids)
            loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        if compute_metrics:
            r2_score.update(pred.float(), yb.float())
            pred_raw = pred.float() * float(y_std) + float(y_mean)
            yb_raw = yb.float() * float(y_std) + float(y_mean)
            abs_err = (pred_raw - yb_raw).abs()
            total_abs_err += abs_err.sum().item()
            denom = yb_raw.abs().clamp_min(float(mape_eps))
            total_abs_pct += (abs_err / denom * 100.0).sum().item()
    avg_loss = total_loss / max(1, total_n)
    if compute_metrics:
        val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
        val_mae = total_abs_err / max(1, total_n)
        val_mape = total_abs_pct / max(1, total_n)
    else:
        val_r2, val_mae, val_mape = None, None, None
    return avg_loss, val_r2, val_mae, val_mape


def train_balanced_sep_mlp_shared_calib(options, seed):
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
    if device.type == "cuda":
        th.backends.cudnn.benchmark = True
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
    df_tgt_test = obj.get("tgt_test_df")
    if df_tgt_train is None or df_tgt_val is None:
        raise RuntimeError("dataset.pkl must include tgt_train_df and tgt_val_df")
    apply_auto_transfer_by_sup(options, df_tgt_train, df_tgt_val, df_tgt_test)

    x_mean = np.array([scaler_stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([scaler_stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean = float(y_scaler.get("mean", 0.0))
    y_std = float(y_scaler.get("std", 1.0))
    if abs(y_std) < 1e-12:
        y_std = 1.0

    df_src_use = df_src if df_src is not None and len(df_src) > 0 else df_tgt_train
    tgt_group_col = str(getattr(options, "tgt_group_reweight_col", "group_id"))
    tgt_sample_w, tgt_group_w_info = build_target_group_sample_weights(df_tgt_train, options)
    topo_to_id = build_topology_vocab(df_src_use, df_tgt_train, df_tgt_val, df_tgt_test)

    ds_tgt = CellDelayDataset(
        df_tgt_train,
        x_mean,
        x_std,
        y_mean,
        y_std,
        sample_weights=tgt_sample_w,
        group_col=tgt_group_col,
        topo_to_id=topo_to_id,
    )
    ds_src = CellDelayDataset(
        df_src_use, x_mean, x_std, y_mean, y_std, group_col=tgt_group_col, topo_to_id=topo_to_id
    )
    val_ds = CellDelayDataset(
        df_tgt_val, x_mean, x_std, y_mean, y_std, group_col=tgt_group_col, topo_to_id=topo_to_id
    )
    has_test_samples = df_tgt_test is not None and len(df_tgt_test) > 0
    test_ds = (
        CellDelayDataset(df_tgt_test, x_mean, x_std, y_mean, y_std, group_col=tgt_group_col, topo_to_id=topo_to_id)
        if has_test_samples
        else None
    )
    tgt_cts_all = list(ds_tgt.cts) + list(val_ds.cts)
    if test_ds is not None:
        tgt_cts_all += list(test_ds.cts)
    src_cts_all = list(ds_src.cts)

    def my_collate(batch):
        if len(batch) == 0:
            raise ValueError("Empty batch in collate.")
        item_len = len(batch[0])
        if item_len == 8:
            xs, ys, cts, from_pins, to_pins, topo_ids, group_ids, sample_ws = zip(*batch)
        elif item_len == 7:
            xs, ys, cts, from_pins, to_pins, group_ids, sample_ws = zip(*batch)
            topo_ids = tuple([0] * len(xs))
        elif item_len == 5:
            xs, ys, cts, from_pins, to_pins = zip(*batch)
            topo_ids = tuple([0] * len(xs))
            group_ids = tuple(["__NA__"] * len(xs))
            sample_ws = tuple([1.0] * len(xs))
        else:
            raise ValueError(f"Unsupported sample tuple length: {item_len}")
        return (
            th.stack(xs),
            th.stack(ys),
            cts,
            from_pins,
            to_pins,
            th.tensor(topo_ids, dtype=th.long),
            group_ids,
            th.tensor(sample_ws, dtype=th.float32),
        )

    batch_size_tgt = max(1, options.batch_size // 2)
    batch_size_src = max(1, options.batch_size // (2 * max(1, options.sample_45_num)))

    num_workers = int(getattr(options, "num_workers", 4))
    pin_memory = bool(getattr(options, "pin_memory", device.type == "cuda"))
    persistent_workers_opt = getattr(options, "persistent_workers", num_workers > 0)
    persistent_workers = bool(int(persistent_workers_opt)) if not isinstance(persistent_workers_opt, bool) else persistent_workers_opt
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
    skip_test_eval = bool(getattr(options, "skip_test_eval", False))
    test_dl = None
    if skip_test_eval:
        print("[Info] skip_test_eval=True: disable periodic/final test evaluation.")
    elif test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=options.batch_size, shuffle=False, **dl_kwargs)
    else:
        print("[Info] tgt_test_df is empty/missing in dataset.pkl; test evaluation disabled.")

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
    enrich_parasitic_net_feat = bool(getattr(options, "enrich_parasitic_net_feat", False))
    hgat_net_feat_mode = _resolve_hgat_net_feat_mode(options)
    hgat_par_cap_weight = float(getattr(options, "hgat_par_cap_weight", 0.5))
    hgat_rich_include_body = bool(getattr(options, "hgat_rich_include_body", False))
    hgat_dual_readout = bool(getattr(options, "hgat_dual_readout", False))
    hgat_dual_merge = str(getattr(options, "hgat_dual_merge", "concat")).strip().lower()
    dropout = getattr(options, "mlp_dropout", 0.0)
    use_calib_mlp = bool(getattr(options, "use_calib_mlp", False))
    calib_mlp_hid = int(getattr(options, "calib_mlp_hid", 0))
    calib_mlp_dropout = float(getattr(options, "calib_mlp_dropout", -1.0))
    disable_domain_calibration = bool(getattr(options, "disable_domain_calibration", False))
    disable_graph_feature = bool(getattr(options, "disable_graph_feature", False))
    use_topology_expert = bool(getattr(options, "use_topology_expert", False))
    topology_expert_dim = int(getattr(options, "topology_expert_dim", 16))
    topology_expert_hidden = int(getattr(options, "topology_expert_hidden", 64))
    v7_adaptive_src_gate = bool(getattr(options, "v7_adaptive_src_gate", False))
    v7_gate_temp = max(1e-3, float(getattr(options, "v7_gate_temp", 0.5)))
    v7_gate_floor = float(getattr(options, "v7_gate_floor", 0.2))
    v7_gate_floor = min(max(v7_gate_floor, 0.0), 1.0)
    v7_disentangle = bool(getattr(options, "v7_disentangle", False))
    v7_disentangle_hidden = int(getattr(options, "v7_disentangle_hidden", 0))
    v7_disentangle_orth_w = max(0.0, float(getattr(options, "v7_disentangle_orth_w", 0.0)))
    v7_topo_moe_k = max(1, int(getattr(options, "v7_topo_moe_k", 1)))
    v7_topo_moe_temp = max(1e-3, float(getattr(options, "v7_topo_moe_temp", 1.0)))
    v7_calib_scale = max(0.0, float(getattr(options, "v7_calib_scale", 1.0)))
    v7_topo_scale = max(0.0, float(getattr(options, "v7_topo_scale", 1.0)))
    num_topologies = max(1, len(topo_to_id))
    z_dim = _effective_z_dim(design_dim, hgat_dual_readout, hgat_dual_merge)
    print(f"[Info] HGAT NET parasitic feature enrich: {enrich_parasitic_net_feat}")
    print(f"[Info] HGAT NET feature mode: {hgat_net_feat_mode}")
    print(f"[Info] HGAT parasitic cap weight: {hgat_par_cap_weight:.3f}")
    print(f"[Info] HGAT rich include body edges: {hgat_rich_include_body}")
    print(f"[Info] HGAT dual readout: {hgat_dual_readout} (merge={hgat_dual_merge})")
    print(f"[Info] Effective design dim (z): {z_dim}")
    print(
        f"[Info] Calibration head: {'MLP' if use_calib_mlp else 'Linear'}"
        + (f" (hid={calib_mlp_hid if calib_mlp_hid > 0 else hgat_hid}, dropout={calib_mlp_dropout if calib_mlp_dropout >= 0 else dropout})" if use_calib_mlp else "")
    )
    print(f"[Info] Disable domain calibration: {disable_domain_calibration}")
    print(f"[Info] Disable graph feature: {disable_graph_feature}")
    print(
        f"[Info] Topology expert: {'enabled' if use_topology_expert else 'disabled'}"
        + (f" (num_topologies={num_topologies}, emb={topology_expert_dim}, hid={topology_expert_hidden})" if use_topology_expert else "")
    )
    print(
        f"[Info] V7 adaptive src gate: {'enabled' if v7_adaptive_src_gate else 'disabled'}"
        + (f" (temp={v7_gate_temp:.3f}, floor={v7_gate_floor:.3f})" if v7_adaptive_src_gate else "")
    )
    print(
        f"[Info] V7 disentangle: {'enabled' if v7_disentangle else 'disabled'}"
        + (f" (hid={v7_disentangle_hidden if v7_disentangle_hidden > 0 else hgat_hid}, orth_w={v7_disentangle_orth_w:.4f})" if v7_disentangle else "")
    )
    print(
        f"[Info] V7 topology MoE: {'enabled' if (use_topology_expert and v7_topo_moe_k > 1) else 'disabled'}"
        + (f" (k={v7_topo_moe_k}, temp={v7_topo_moe_temp:.3f})" if (use_topology_expert and v7_topo_moe_k > 1) else "")
    )
    print(f"[Info] V7 branch scales: calib={v7_calib_scale:.3f}, topo={v7_topo_scale:.3f}")
    if tgt_group_w_info is not None:
        print(
            "[Info] Target group reweight enabled: "
            f"col={tgt_group_w_info['col']}, groups={tgt_group_w_info['num_groups']}, "
            f"power={tgt_group_w_info['power']:.3f}, clip=[{tgt_group_w_info['clip_min']:.3f},{tgt_group_w_info['clip_max']:.3f}], "
            f"sample_w[min/mean/max]=[{tgt_group_w_info['sample_w_min']:.3f}/{tgt_group_w_info['sample_w_mean']:.3f}/{tgt_group_w_info['sample_w_max']:.3f}]"
        )
    else:
        print("[Info] Target group reweight: disabled")

    if hgat_net_feat_mode == "parasitic_append_split":
        graph_net_dim = 9
    elif hgat_net_feat_mode == "parasitic_append":
        graph_net_dim = 8
    else:
        graph_net_dim = GRAPH_NET_DIM
    in_map = {"NET": graph_net_dim, "PMOS": GRAPH_MOS_DIM, "NMOS": GRAPH_MOS_DIM}
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
        design_dim=z_dim,
        hid=hgat_hid,
        dropout=dropout,
        use_calib_mlp=use_calib_mlp,
        calib_mlp_hid=calib_mlp_hid,
        calib_mlp_dropout=calib_mlp_dropout,
        use_topology_expert=use_topology_expert,
        num_topologies=num_topologies,
        topology_expert_dim=topology_expert_dim,
        topology_expert_hidden=topology_expert_hidden,
        disable_domain_calibration=disable_domain_calibration,
        use_disentangle=v7_disentangle,
        disentangle_hidden=v7_disentangle_hidden,
        topo_moe_k=v7_topo_moe_k,
        topo_moe_temp=v7_topo_moe_temp,
        calib_branch_scale=v7_calib_scale,
        topo_branch_scale=v7_topo_scale,
    ).to(device)

    maybe_load_hgat_encoder_checkpoint(enc, options, device)
    maybe_load_full_checkpoint(enc, model, options, device)

    print("----------------Loading HGAT graphs----------------")
    z_dict_src = None
    z_dict_tgt = None
    graph_cache_src = None
    graph_cache_tgt = None
    if disable_graph_feature:
        print("[Info] disable_graph_feature=True, skip HGAT graph building/cache and use zero-z.")
        if options.freeze_hgat:
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()
            optimizer = th.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
        else:
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
    elif options.freeze_hgat:
        print("[Info] freeze_hgat=True, precomputing z")
        graph_cache_src = build_src_graph_cache(
            data_dir,
            meta,
            device,
            enrich_parasitic_net_feat=enrich_parasitic_net_feat,
            include_body_edges=hgat_rich_include_body,
            net_feat_mode=hgat_net_feat_mode,
            par_cap_weight=hgat_par_cap_weight,
        )
        graph_cache_tgt = build_tgt_graph_cache(
            data_dir,
            meta,
            device,
            enrich_parasitic_net_feat=enrich_parasitic_net_feat,
            include_body_edges=hgat_rich_include_body,
            net_feat_mode=hgat_net_feat_mode,
            par_cap_weight=hgat_par_cap_weight,
        )
        report_graph_cache_coverage("src", graph_cache_src, src_cts_all)
        report_graph_cache_coverage("tgt", graph_cache_tgt, tgt_cts_all)
        z_dict_src = precompute_z_from_graph_cache(
            graph_cache_src,
            enc,
            dual_readout=hgat_dual_readout,
            dual_merge=hgat_dual_merge,
        )
        z_dict_tgt = precompute_z_from_graph_cache(
            graph_cache_tgt,
            enc,
            dual_readout=hgat_dual_readout,
            dual_merge=hgat_dual_merge,
        )
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
        optimizer = th.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    else:
        print("[Info] freeze_hgat=False, building graph cache")
        graph_cache_src = build_src_graph_cache(
            data_dir,
            meta,
            device,
            enrich_parasitic_net_feat=enrich_parasitic_net_feat,
            include_body_edges=hgat_rich_include_body,
            net_feat_mode=hgat_net_feat_mode,
            par_cap_weight=hgat_par_cap_weight,
        )
        graph_cache_tgt = build_tgt_graph_cache(
            data_dir,
            meta,
            device,
            enrich_parasitic_net_feat=enrich_parasitic_net_feat,
            include_body_edges=hgat_rich_include_body,
            net_feat_mode=hgat_net_feat_mode,
            par_cap_weight=hgat_par_cap_weight,
        )
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

    loss_fn = build_train_loss_fn(options)
    z_noise_std = max(0.0, float(getattr(options, "z_noise_std", 0.0)))
    if z_noise_std > 0:
        print(f"[Info] Z noise augmentation enabled: std={z_noise_std}")
    target_loss_weight = float(getattr(options, "target_loss_weight", 1.0))
    print(f"[Info] Target loss weight: {target_loss_weight:.4f}")
    grad_clip_norm = max(0.0, float(getattr(options, "grad_clip_norm", 0.0)))
    if grad_clip_norm > 0:
        print(f"[Info] Grad clipping enabled: max_norm={grad_clip_norm}")
    skip_train_r2 = bool(getattr(options, "skip_train_r2", False))
    fast_eval_loss_only = bool(getattr(options, "fast_eval_loss_only", False))
    epoch_log_interval = max(1, int(getattr(options, "epoch_log_interval", 1)))
    if skip_train_r2:
        print("[Info] skip_train_r2=True: disable train-R2 accumulation/computation.")
    if fast_eval_loss_only:
        print("[Info] fast_eval_loss_only=True: validation during training uses loss only; full val metrics run once at end.")
    if epoch_log_interval > 1:
        print(f"[Info] Epoch log interval: every {epoch_log_interval} epochs")
    r2_score = None if skip_train_r2 else R2Score().to(device)
    scaler = th.cuda.amp.GradScaler(enabled=amp_on)
    scheduler = build_scheduler(optimizer, options)
    enc_update_interval = max(1, int(getattr(options, "enc_update_interval", 1)))
    if (not options.freeze_hgat) and enc_update_interval > 1:
        print(f"[Info] Fast unfreeze: encoder updated every {enc_update_interval} target batches")

    early_stop_patience = int(getattr(options, "early_stop_patience", 0))
    early_stop_min_delta = max(0.0, float(getattr(options, "early_stop_min_delta", 0.0)))
    if early_stop_patience > 0:
        print(f"[Info] Early stop enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    else:
        print("[Info] Early stop: disabled")

    final_model_dir = os.path.abspath(str(options.model_saving_dir))
    work_model_dir = os.path.abspath(str(getattr(options, "model_work_dir", final_model_dir)))
    if work_model_dir != final_model_dir:
        print(f"[Info] Fast checkpoint dir: {work_model_dir}")
        print(f"[Info] Final checkpoint dir: {final_model_dir}")

    print("----------------Start training---------------")
    best_val = float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_mae = None
    best_val_mape = None
    best_test_r2 = None
    best_test_loss = None
    best_test_mae = None
    best_test_mape = None
    best_test_eval_epoch = None
    final_test_r2 = None
    final_test_loss = None
    final_test_mae = None
    final_test_mape = None
    best_ckpt_obj = None
    best_ckpt_path = os.path.join(work_model_dir, "ckpt_best.pt")
    final_best_ckpt_path = os.path.join(final_model_dir, "ckpt_best.pt")
    val_eval_interval = max(1, int(getattr(options, "val_eval_interval", 1)))
    test_eval_interval = max(1, int(getattr(options, "test_eval_interval", 50)))
    print(f"[Info] Val eval interval: every {val_eval_interval} epochs")
    if test_dl is not None:
        print(f"[Info] Test eval interval: every {test_eval_interval} epochs (using current best-val checkpoint)")
    else:
        print("[Info] Test eval interval: disabled")
    enc_eval = None
    model_eval = None
    if test_dl is not None:
        enc_eval = HGATDesignEncoder(
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
        model_eval = SharedCalibRegressor(
            in_dim=options.in_dim,
            design_dim=z_dim,
            hid=hgat_hid,
            dropout=dropout,
            use_calib_mlp=use_calib_mlp,
            calib_mlp_hid=calib_mlp_hid,
            calib_mlp_dropout=calib_mlp_dropout,
            use_topology_expert=use_topology_expert,
            num_topologies=num_topologies,
            topology_expert_dim=topology_expert_dim,
            topology_expert_hidden=topology_expert_hidden,
            disable_domain_calibration=disable_domain_calibration,
            use_disentangle=v7_disentangle,
            disentangle_hidden=v7_disentangle_hidden,
            topo_moe_k=v7_topo_moe_k,
            topo_moe_temp=v7_topo_moe_temp,
            calib_branch_scale=v7_calib_scale,
            topo_branch_scale=v7_topo_scale,
        ).to(device)
        for p in enc_eval.parameters():
            p.requires_grad = False
        for p in model_eval.parameters():
            p.requires_grad = False
    patience_counter = 0

    global_step = 0
    # Reuse z across consecutive no-encoder-update steps.
    # Cache is invalidated whenever encoder parameters are updated.
    z_interval_cache = {}
    for epoch in range(options.num_epoch):
        if options.freeze_hgat:
            enc.eval()
        else:
            enc.train()
        model.train()
        if r2_score is not None:
            r2_score.reset()
        total_loss = 0.0
        total_n = 0
        src_weight_epoch = compute_src_loss_weight(epoch, options)

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt, from_pins_tgt, to_pins_tgt, topo_ids_tgt, _, sample_w_tgt in dl_tgt:
            update_encoder_this_step = bool(
                options.freeze_hgat
                or enc_update_interval <= 1
                or (global_step % enc_update_interval == 0)
            )
            xb_tgt = xb_tgt.to(device, non_blocking=True)
            yb_tgt = yb_tgt.to(device, non_blocking=True)
            topo_ids_tgt = topo_ids_tgt.to(device, non_blocking=True)
            sample_w_tgt = sample_w_tgt.to(device, non_blocking=True)
            tgt_n = len(yb_tgt)
            src_weighted_loss_sum = 0.0
            src_n_sum = 0
            if options.freeze_hgat:
                z_step_cache = None
            else:
                if update_encoder_this_step:
                    z_interval_cache = {}
                z_step_cache = z_interval_cache

            with th.cuda.amp.autocast(enabled=amp_on):
                z_ctx = contextlib.nullcontext() if update_encoder_this_step else th.no_grad()
                with z_ctx:
                    zb_tgt = build_z_batch(
                        cts_tgt,
                        device,
                        z_dim,
                        from_pins=from_pins_tgt,
                        to_pins=to_pins_tgt,
                        z_dict=z_dict_tgt,
                        graph_cache=graph_cache_tgt,
                        enc=enc,
                        dedup=options.dedup_z,
                        z_step_cache=z_step_cache,
                        use_batched_graph_encode=True,
                        dual_readout=hgat_dual_readout,
                        dual_merge=hgat_dual_merge,
                        disable_graph_feature=disable_graph_feature,
                    )
                if not update_encoder_this_step:
                    zb_tgt = zb_tgt.detach()
                if z_noise_std > 0:
                    zb_tgt = zb_tgt + th.randn_like(zb_tgt) * z_noise_std
                    if hgat_l2_norm:
                        zb_tgt = th.nn.functional.normalize(zb_tgt, p=2, dim=1)
                pred_tgt, aux_tgt = model.forward_with_aux(xb_tgt, zb_tgt, node="tgt", topo_ids=topo_ids_tgt)
                loss_tgt = compute_loss_with_optional_weights(
                    loss_fn,
                    pred_tgt,
                    yb_tgt,
                    sample_w=sample_w_tgt if tgt_group_w_info is not None else None,
                )
                if v7_disentangle and aux_tgt["h_inv"] is not None and aux_tgt["h_dom"] is not None:
                    inv_t = th.nn.functional.normalize(aux_tgt["h_inv"], p=2, dim=1)
                    dom_t = th.nn.functional.normalize(aux_tgt["h_dom"], p=2, dim=1)
                    orth_tgt = (inv_t * dom_t).sum(dim=1).pow(2).mean()
                else:
                    orth_tgt = th.zeros((), device=pred_tgt.device, dtype=pred_tgt.dtype)

            orth_weighted_loss_sum = orth_tgt * tgt_n
            orth_n_sum = tgt_n
            if v7_adaptive_src_gate:
                tgt_ref = aux_tgt["h_shared"].detach()
                tgt_center = tgt_ref.mean(dim=0, keepdim=True)
            else:
                tgt_center = None

            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src, from_pins_src, to_pins_src, topo_ids_src, _, _ = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src, from_pins_src, to_pins_src, topo_ids_src, _, _ = next(dl_src_iter)
                xb_src = xb_src.to(device, non_blocking=True)
                yb_src = yb_src.to(device, non_blocking=True)
                topo_ids_src = topo_ids_src.to(device, non_blocking=True)
                src_n = len(yb_src)
                with th.cuda.amp.autocast(enabled=amp_on):
                    z_ctx = contextlib.nullcontext() if update_encoder_this_step else th.no_grad()
                    with z_ctx:
                        zb_src = build_z_batch(
                            cts_src,
                            device,
                            z_dim,
                            from_pins=from_pins_src,
                            to_pins=to_pins_src,
                            z_dict=z_dict_src,
                            graph_cache=graph_cache_src,
                            enc=enc,
                            dedup=options.dedup_z,
                            z_step_cache=z_step_cache,
                            use_batched_graph_encode=True,
                            dual_readout=hgat_dual_readout,
                            dual_merge=hgat_dual_merge,
                            disable_graph_feature=disable_graph_feature,
                        )
                    if not update_encoder_this_step:
                        zb_src = zb_src.detach()
                    if z_noise_std > 0:
                        zb_src = zb_src + th.randn_like(zb_src) * z_noise_std
                        if hgat_l2_norm:
                            zb_src = th.nn.functional.normalize(zb_src, p=2, dim=1)
                    pred_src, aux_src = model.forward_with_aux(xb_src, zb_src, node="src", topo_ids=topo_ids_src)
                    if tgt_center is not None:
                        src_ref = th.nn.functional.normalize(aux_src["h_shared"], p=2, dim=1)
                        center_ref = th.nn.functional.normalize(tgt_center, p=2, dim=1).expand(src_ref.shape[0], -1)
                        sim = (src_ref * center_ref).sum(dim=1)
                        gate = v7_gate_floor + (1.0 - v7_gate_floor) * th.sigmoid(sim / v7_gate_temp)
                    else:
                        gate = None
                    loss_src = compute_loss_with_optional_weights(loss_fn, pred_src, yb_src, sample_w=gate)
                    if v7_disentangle and aux_src["h_inv"] is not None and aux_src["h_dom"] is not None:
                        inv_s = th.nn.functional.normalize(aux_src["h_inv"], p=2, dim=1)
                        dom_s = th.nn.functional.normalize(aux_src["h_dom"], p=2, dim=1)
                        orth_src = (inv_s * dom_s).sum(dim=1).pow(2).mean()
                        orth_weighted_loss_sum = orth_weighted_loss_sum + (orth_src * src_n)
                        orth_n_sum += src_n
                src_weighted_loss_sum = src_weighted_loss_sum + (loss_src * src_n)
                src_n_sum += src_n

            denom = max(1e-12, target_loss_weight * tgt_n + src_n_sum)
            with th.cuda.amp.autocast(enabled=amp_on):
                total_loss_batch = (
                    target_loss_weight * loss_tgt * tgt_n + src_weight_epoch * src_weighted_loss_sum
                ) / denom
                if v7_disentangle_orth_w > 0:
                    total_loss_batch = total_loss_batch + v7_disentangle_orth_w * (
                        orth_weighted_loss_sum / max(1, orth_n_sum)
                    )

            optimizer.zero_grad()
            scaler.scale(total_loss_batch).backward()
            if grad_clip_norm > 0:
                if amp_on:
                    scaler.unscale_(optimizer)
                clip_grad_by_optimizer(optimizer, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if update_encoder_this_step and (not options.freeze_hgat):
                z_interval_cache = {}

            total_loss += total_loss_batch.item() * len(yb_tgt)
            total_n += len(yb_tgt)
            if r2_score is not None:
                r2_score.update(pred_tgt.float(), yb_tgt.float())
            global_step += 1

        train_loss = total_loss / max(1, total_n)
        train_r2 = (r2_score.compute().item() if (r2_score is not None and total_n > 0) else None)
        is_plateau = isinstance(scheduler, th.optim.lr_scheduler.ReduceLROnPlateau)
        if scheduler is not None and (not is_plateau):
            scheduler.step()

        do_val_eval = ((epoch + 1) % val_eval_interval == 0) or ((epoch + 1) == int(options.num_epoch))
        val_loss, val_r2, val_mae, val_mape = (None, None, None, None)
        stop_now = False
        if do_val_eval:
            val_loss, val_r2, val_mae, val_mape = validate_cell(
                val_dl,
                enc,
                model,
                device,
                z_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                dedup=options.dedup_z,
                use_amp=use_amp,
                dual_readout=hgat_dual_readout,
                dual_merge=hgat_dual_merge,
                y_mean=y_mean,
                y_std=y_std,
                compute_metrics=(not fast_eval_loss_only),
                disable_graph_feature=disable_graph_feature,
            )
            if fast_eval_loss_only:
                delta_req_loss = max(1e-12, early_stop_min_delta)
                better_loss = val_loss < (best_val_loss - delta_req_loss)
                if better_loss:
                    best_epoch = epoch + 1
                    best_val_loss = val_loss
                    patience_counter = 0
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
                            "src_loss_weight_epoch": src_weight_epoch,
                            "script": "shared_calib",
                        },
                        best_ckpt_path,
                    )
                    best_ckpt_obj = {
                        "enc": _clone_state_dict_to_cpu(enc.state_dict()),
                        "model": _clone_state_dict_to_cpu(model.state_dict()),
                    }
                    print("Model successfully saved")
                else:
                    if early_stop_patience > 0:
                        patience_counter += 1
                        if patience_counter >= early_stop_patience:
                            print(f"[EarlyStop] no improvement for {early_stop_patience} eval rounds; stop at epoch {epoch}")
                            stop_now = True
            else:
                delta_req = max(CKPT_R2_TIE_EPS_DEFAULT, early_stop_min_delta)
                better_r2 = val_r2 > (best_val + delta_req)
                tie_better = abs(val_r2 - best_val) <= delta_req and val_loss < best_val_loss
                if better_r2 or tie_better:
                    best_val = val_r2
                    best_epoch = epoch + 1
                    best_val_loss = val_loss
                    best_val_mae = val_mae
                    best_val_mape = val_mape
                    patience_counter = 0
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
                            "src_loss_weight_epoch": src_weight_epoch,
                            "script": "shared_calib",
                        },
                        best_ckpt_path,
                    )
                    best_ckpt_obj = {
                        "enc": _clone_state_dict_to_cpu(enc.state_dict()),
                        "model": _clone_state_dict_to_cpu(model.state_dict()),
                    }
                    print("Model successfully saved")
                else:
                    if early_stop_patience > 0:
                        patience_counter += 1
                        if patience_counter >= early_stop_patience:
                            print(f"[EarlyStop] no improvement for {early_stop_patience} eval rounds; stop at epoch {epoch}")
                            stop_now = True
        if scheduler is not None and is_plateau and do_val_eval and (val_loss is not None):
            # For ReduceLROnPlateau, step on validation metric only.
            # This avoids over-decay when val_eval_interval > 1.
            scheduler.step(float(val_loss))

        test_loss, test_r2, test_mae, test_mape = (None, None, None, None)
        do_test_eval = (
            test_dl is not None
            and os.path.exists(best_ckpt_path)
            and do_val_eval
            and (((epoch + 1) % test_eval_interval == 0) or ((epoch + 1) == int(options.num_epoch)))
        )
        if do_test_eval:
            can_eval = True
            if isinstance(best_ckpt_obj, dict):
                enc_sd = best_ckpt_obj.get("enc")
                model_sd = best_ckpt_obj.get("model")
            else:
                ckpt = th.load(best_ckpt_path, map_location=device)
                enc_sd = ckpt.get("enc") if isinstance(ckpt, dict) else None
                model_sd = ckpt.get("model") if isinstance(ckpt, dict) else None
            if not isinstance(enc_sd, dict):
                print(f"[Warn] Periodic test eval skipped at e{epoch + 1}: ckpt missing valid 'enc'.")
                can_eval = False
            else:
                ok = _load_hgat_encoder_state_dict(enc_eval, enc_sd)
                if not ok:
                    print(f"[Warn] Periodic test eval skipped at e{epoch + 1}: failed to load encoder from ckpt.")
                    can_eval = False
            if can_eval:
                if not isinstance(model_sd, dict):
                    print(f"[Warn] Periodic test eval skipped at e{epoch + 1}: ckpt missing valid 'model'.")
                    can_eval = False
                else:
                    model_sd_fit, mm_missing, mm_shape = _filter_compatible_state_dict(model_eval, model_sd)
                    if len(model_sd_fit) == 0:
                        print(f"[Warn] Periodic test eval skipped at e{epoch + 1}: model state_dict matched 0 tensors.")
                        can_eval = False
                    else:
                        load_res = model_eval.load_state_dict(model_sd_fit, strict=False)
                        if mm_missing:
                            print(f"[Warn] Periodic test model keys not in model: {len(mm_missing)}")
                        if mm_shape:
                            print(f"[Warn] Periodic test model keys shape-mismatch: {len(mm_shape)}")
                        if load_res.missing_keys:
                            print(f"[Info] Periodic test model missing keys after partial load: {len(load_res.missing_keys)}")
                        if load_res.unexpected_keys:
                            print(f"[Info] Periodic test model unexpected keys after partial load: {len(load_res.unexpected_keys)}")
            if can_eval:
                test_loss, test_r2, test_mae, test_mape = validate_cell(
                    test_dl,
                    enc_eval,
                    model_eval,
                    device,
                    z_dim,
                    z_dict=z_dict_tgt,
                    graph_cache=graph_cache_tgt,
                    dedup=options.dedup_z,
                    use_amp=use_amp,
                    dual_readout=hgat_dual_readout,
                    dual_merge=hgat_dual_merge,
                    y_mean=y_mean,
                    y_std=y_std,
                    disable_graph_feature=disable_graph_feature,
                )
                best_test_r2 = test_r2
                best_test_loss = test_loss
                best_test_mae = test_mae
                best_test_mape = test_mape
                best_test_eval_epoch = epoch + 1

        cur_lr = float(optimizer.param_groups[-1]["lr"])
        train_r2_str = "skip" if train_r2 is None else f"{train_r2:.3f}"
        if do_val_eval and val_loss is not None:
            if val_r2 is None:
                log_msg = (
                    f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2_str}, "
                    f"val_loss:{val_loss:.4f}, lr:{cur_lr:.2e}, src_w:{src_weight_epoch:.4f}"
                )
            else:
                log_msg = (
                    f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2_str}, "
                    f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, val_mae:{val_mae:.4f}, val_mape:{val_mape:.2f}, "
                    f"lr:{cur_lr:.2e}, src_w:{src_weight_epoch:.4f}"
                )
        else:
            log_msg = (
                f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2_str}, "
                f"lr:{cur_lr:.2e}, src_w:{src_weight_epoch:.4f}, val_eval:skip"
            )
        if test_loss is not None:
            log_msg += (
                f", best_test_loss:{test_loss:.4f}, best_test_r2:{test_r2:.3f}, "
                f"best_test_mae:{test_mae:.4f}, best_test_mape:{test_mape:.2f}"
                f" (eval@e{epoch + 1}, ckpt_best@e{best_epoch})"
            )
        should_log = (
            do_val_eval
            or stop_now
            or ((epoch + 1) % epoch_log_interval == 0)
            or ((epoch + 1) == int(options.num_epoch))
        )
        if should_log:
            print(log_msg)
        if stop_now:
            break

    if best_epoch > 0 and os.path.exists(best_ckpt_path):
        os.makedirs(final_model_dir, exist_ok=True)
        if os.path.abspath(best_ckpt_path) != os.path.abspath(final_best_ckpt_path):
            shutil.copy2(best_ckpt_path, final_best_ckpt_path)
            print(f"[Info] Copied best checkpoint to final dir: {final_best_ckpt_path}")
        else:
            final_best_ckpt_path = best_ckpt_path

    # Always run one final test evaluation at training end using ckpt_best.
    # This makes test metrics available even when early stop happens before test_eval_interval.
    if (
        test_dl is not None
        and best_epoch > 0
        and os.path.exists(best_ckpt_path)
        and enc_eval is not None
        and model_eval is not None
    ):
        can_eval = True
        if isinstance(best_ckpt_obj, dict):
            enc_sd = best_ckpt_obj.get("enc")
            model_sd = best_ckpt_obj.get("model")
        else:
            ckpt = th.load(best_ckpt_path, map_location=device)
            enc_sd = ckpt.get("enc") if isinstance(ckpt, dict) else None
            model_sd = ckpt.get("model") if isinstance(ckpt, dict) else None

        if not isinstance(enc_sd, dict):
            print("[Warn] Final test eval skipped: ckpt missing valid 'enc'.")
            can_eval = False
        else:
            ok = _load_hgat_encoder_state_dict(enc_eval, enc_sd)
            if not ok:
                print("[Warn] Final test eval skipped: failed to load encoder from ckpt.")
                can_eval = False

        if can_eval:
            if not isinstance(model_sd, dict):
                print("[Warn] Final test eval skipped: ckpt missing valid 'model'.")
                can_eval = False
            else:
                model_sd_fit, mm_missing, mm_shape = _filter_compatible_state_dict(model_eval, model_sd)
                if len(model_sd_fit) == 0:
                    print("[Warn] Final test eval skipped: model state_dict matched 0 tensors.")
                    can_eval = False
                else:
                    load_res = model_eval.load_state_dict(model_sd_fit, strict=False)
                    if mm_missing:
                        print(f"[Warn] Final test model keys not in model: {len(mm_missing)}")
                    if mm_shape:
                        print(f"[Warn] Final test model keys shape-mismatch: {len(mm_shape)}")
                    if load_res.missing_keys:
                        print(f"[Info] Final test model missing keys after partial load: {len(load_res.missing_keys)}")
                    if load_res.unexpected_keys:
                        print(f"[Info] Final test model unexpected keys after partial load: {len(load_res.unexpected_keys)}")

        if can_eval:
            final_test_loss, final_test_r2, final_test_mae, final_test_mape = validate_cell(
                test_dl,
                enc_eval,
                model_eval,
                device,
                z_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                dedup=options.dedup_z,
                use_amp=use_amp,
                dual_readout=hgat_dual_readout,
                dual_merge=hgat_dual_merge,
                y_mean=y_mean,
                y_std=y_std,
                disable_graph_feature=disable_graph_feature,
            )
            print(
                f"[FinalTest] loss:{final_test_loss:.6f}, r2:{final_test_r2:.4f}, "
                f"mae:{final_test_mae:.6f}, mape:{final_test_mape:.3f}, "
                f"ckpt_best_epoch:{best_epoch}"
            )

    # In fast_eval_loss_only mode, validation during training may not compute R2/MAE/MAPE.
    # Recompute full validation metrics once on ckpt_best before printing summary.
    if (
        best_epoch > 0
        and os.path.exists(best_ckpt_path)
        and (best_val_mae is None or best_val_mape is None or not np.isfinite(best_val))
    ):
        can_eval = True
        if isinstance(best_ckpt_obj, dict):
            enc_sd = best_ckpt_obj.get("enc")
            model_sd = best_ckpt_obj.get("model")
        else:
            ckpt = th.load(best_ckpt_path, map_location=device)
            enc_sd = ckpt.get("enc") if isinstance(ckpt, dict) else None
            model_sd = ckpt.get("model") if isinstance(ckpt, dict) else None

        if not isinstance(enc_sd, dict):
            can_eval = False
            print("[Warn] Final val-metric recompute skipped: ckpt missing valid 'enc'.")
        else:
            ok = _load_hgat_encoder_state_dict(enc, enc_sd)
            if not ok:
                can_eval = False
                print("[Warn] Final val-metric recompute skipped: failed to load encoder from ckpt.")

        if can_eval:
            if not isinstance(model_sd, dict):
                can_eval = False
                print("[Warn] Final val-metric recompute skipped: ckpt missing valid 'model'.")
            else:
                model_sd_fit, _, _ = _filter_compatible_state_dict(model, model_sd)
                if len(model_sd_fit) == 0:
                    can_eval = False
                    print("[Warn] Final val-metric recompute skipped: model state_dict matched 0 tensors.")
                else:
                    model.load_state_dict(model_sd_fit, strict=False)

        if can_eval:
            val_loss_best, val_r2_best, val_mae_best, val_mape_best = validate_cell(
                val_dl,
                enc,
                model,
                device,
                z_dim,
                z_dict=z_dict_tgt,
                graph_cache=graph_cache_tgt,
                dedup=options.dedup_z,
                use_amp=use_amp,
                dual_readout=hgat_dual_readout,
                dual_merge=hgat_dual_merge,
                y_mean=y_mean,
                y_std=y_std,
                compute_metrics=True,
                disable_graph_feature=disable_graph_feature,
            )
            best_val_loss = float(val_loss_best)
            best_val = float(val_r2_best)
            best_val_mae = float(val_mae_best)
            best_val_mape = float(val_mape_best)

    if best_epoch > 0:
        best_msg = (
            f"[Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, "
            f"val_loss:{best_val_loss:.6f}, val_mae:{best_val_mae:.6f}, val_mape:{best_val_mape:.3f}"
        )
        if best_test_loss is not None:
            best_msg += (
                f", periodic_best_test_r2:{best_test_r2:.4f}, periodic_best_test_loss:{best_test_loss:.6f}, "
                f"periodic_best_test_mae:{best_test_mae:.6f}, periodic_best_test_mape:{best_test_mape:.3f}, "
                f"test_eval_epoch:{best_test_eval_epoch}"
            )
        if final_test_loss is not None:
            best_msg += (
                f", final_test_r2:{final_test_r2:.4f}, "
                f"final_test_loss:{final_test_loss:.6f}, final_test_mae:{final_test_mae:.6f}, final_test_mape:{final_test_mape:.3f}"
            )
        best_msg += f", ckpt:{final_best_ckpt_path}"
        print(best_msg)


def _resolve_model_work_dir(model_saving_dir, script_stem):
    final_dir = os.path.abspath(str(model_saving_dir))
    work_dir = final_dir
    if os.name != "nt":
        if final_dir.startswith("/mnt/"):
            try:
                work_dir = tempfile.mkdtemp(prefix=f"{script_stem}_", dir="/tmp")
            except Exception:
                work_dir = final_dir
    return final_dir, work_dir


def _parse_v7_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v7_adaptive_src_gate", action="store_true",
                        help="enable sample-level source gating by similarity to target hidden centroid")
    parser.add_argument("--v7_gate_temp", type=float, default=0.5,
                        help="temperature in source gate sigmoid")
    parser.add_argument("--v7_gate_floor", type=float, default=0.2,
                        help="minimum source sample weight when adaptive gate is enabled")
    parser.add_argument("--v7_disentangle", action="store_true",
                        help="split hidden into invariant/domain-specific branches")
    parser.add_argument("--v7_disentangle_hidden", type=int, default=0,
                        help="hidden dim for disentangled branches (<=0 follows hidden_dim)")
    parser.add_argument("--v7_disentangle_orth_w", type=float, default=0.0,
                        help="orthogonality regularization weight between invariant/domain features")
    parser.add_argument("--v7_topo_moe_k", type=int, default=1,
                        help="number of topology experts per domain (1=single residual head)")
    parser.add_argument("--v7_topo_moe_temp", type=float, default=1.0,
                        help="softmax temperature for topology MoE gate")
    parser.add_argument("--v7_calib_scale", type=float, default=1.0,
                        help="scale factor for domain calibration branch output")
    parser.add_argument("--v7_topo_scale", type=float, default=1.0,
                        help="scale factor for topology residual branch output")
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    v7_opts, base_argv = _parse_v7_args(sys.argv[1:])
    options = get_options(base_argv)
    for k, v in vars(v7_opts).items():
        setattr(options, k, v)
    seed = options.seed
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")

    final_model_dir, work_model_dir = _resolve_model_work_dir(options.model_saving_dir, script_stem)
    options.model_saving_dir = final_model_dir
    options.model_work_dir = work_model_dir
    stdout_f = os.path.join(work_model_dir, "stdout.log")
    stderr_f = os.path.join(work_model_dir, "stderr.log")

    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(work_model_dir, exist_ok=True)
    os.makedirs(copilot_log_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        train_balanced_sep_mlp_shared_calib(options, seed)
