r"""Balanced sampling with shared trunk + domain calibration heads (no disentanglement losses)."""

import os
import re
import pickle
import random
import shutil
import tempfile
import contextlib
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
    return x, y, cts, from_pins, to_pins


class CellDelayDataset(TorchDataset):
    def __init__(self, df, x_mean, x_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cts, self.from_pins, self.to_pins = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return th.from_numpy(self.x[i]), th.tensor(self.y[i]), self.cts[i], self.from_pins[i], self.to_pins[i]


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


def build_dgl_graph_from_devs_step5(devs, top_pins, passives=None, return_meta=False):
    return build_dgl_graph_from_devs_rich(devs, top_pins, passives=passives, return_meta=return_meta)


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
        passives = parse_passive_parasitics_spice(sp_text)
        _, pins = parse_top_subckt_pins(sp_text)
        if not devs:
            continue
        g, feats, _, g_meta = build_dgl_graph_from_devs_step5(devs, pins, passives=passives, return_meta=True)
        graph_cache[ctype] = (g.to(device), {k: v.to(device) for k, v in feats.items()}, g_meta)
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
        passives = parse_passive_parasitics_spice(sub_txt)
        _, pins = parse_top_subckt_pins(sub_txt)
        if not devs:
            continue
        g, feats, _, g_meta = build_dgl_graph_from_devs_step5(devs, pins, passives=passives, return_meta=True)
        graph_cache[ctype] = (g.to(device), {k: v.to(device) for k, v in feats.items()}, g_meta)
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
):
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
        z = enc(g, feats, net_focus=net_focus)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # For arc-aware path (especially freeze HGAT), memoize by arc-key
        # so each unique (cell_type, from_pin, to_pin) is encoded once.
        if use_arc and z_dict is not None:
            z_dict[key] = z.detach()
        if z_step_cache is not None:
            z_step_cache[key] = z
        return z

    if (not use_arc) and use_batched_graph_encode and z_dict is None and graph_cache is not None and enc is not None:
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


def precompute_z_from_graph_cache(graph_cache, enc):
    z_dict = {}
    enc.eval()
    with th.no_grad():
        for ct, entry in graph_cache.items():
            g, feats = entry[0], entry[1]
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
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


def build_train_loss_fn(options):
    loss_type = str(getattr(options, "train_loss_type", "mse")).strip().lower()
    if loss_type in ("huber", "smooth_l1", "smoothl1"):
        huber_delta = float(getattr(options, "huber_delta", 1.0))
        print(f"[Info] Training loss: Huber(delta={huber_delta})")
        return nn.HuberLoss(delta=huber_delta)
    print("[Info] Training loss: MSE")
    return nn.MSELoss()


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
    def __init__(self, in_dim, design_dim, hid=256, dropout=0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim + design_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.shared_head = nn.Linear(hid, 1)
        self.calib_tgt = nn.Linear(hid, 1)
        self.calib_src = nn.Linear(hid, 1)

    def forward(self, x, z, node="tgt"):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = th.cat([x, z], dim=1)
        h = self.backbone(h)
        pred = self.shared_head(h).squeeze(-1)
        if node == "tgt":
            pred = pred + self.calib_tgt(h).squeeze(-1)
        elif node == "src":
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
    z_eval_cache = {}
    for xb, yb, cts, from_pins, to_pins in val_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
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
            )
            pred = model(xb, zb, node="tgt")
            loss = loss_fn(pred, yb)
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        r2_score.update(pred.float(), yb.float())
    avg_loss = total_loss / max(1, total_n)
    val_r2 = r2_score.compute().item() if total_n > 0 else 0.0
    return avg_loss, val_r2


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
    test_ds = CellDelayDataset(df_tgt_test, x_mean, x_std, y_mean, y_std) if df_tgt_test is not None else None
    tgt_cts_all = list(ds_tgt.cts) + list(val_ds.cts)
    if test_ds is not None:
        tgt_cts_all += list(test_ds.cts)
    src_cts_all = list(ds_src.cts)

    def my_collate(batch):
        xs, ys, cts, from_pins, to_pins = zip(*batch)
        return th.stack(xs), th.stack(ys), cts, from_pins, to_pins

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
    test_dl = None
    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=options.batch_size, shuffle=False, **dl_kwargs)
    else:
        print("[Warn] tgt_test_df missing in dataset.pkl; skip per-epoch test evaluation.")

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
        l2_norm=hgat_l2_norm,
    ).to(device)
    model = SharedCalibRegressor(in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout).to(
        device
    )

    maybe_load_hgat_encoder_checkpoint(enc, options, device)

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

    loss_fn = build_train_loss_fn(options)
    z_noise_std = max(0.0, float(getattr(options, "z_noise_std", 0.0)))
    if z_noise_std > 0:
        print(f"[Info] Z noise augmentation enabled: std={z_noise_std}")
    r2_score = R2Score().to(device)
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
    best_test_r2 = None
    best_test_loss = None
    best_test_eval_epoch = None
    best_ckpt_obj = None
    best_ckpt_path = os.path.join(work_model_dir, "ckpt_best.pt")
    final_best_ckpt_path = os.path.join(final_model_dir, "ckpt_best.pt")
    val_eval_interval = max(1, int(getattr(options, "val_eval_interval", 1)))
    test_eval_interval = max(1, int(getattr(options, "test_eval_interval", 50)))
    print(f"[Info] Val eval interval: every {val_eval_interval} epochs")
    print(f"[Info] Test eval interval: every {test_eval_interval} epochs (using current best-val checkpoint)")
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
            in_dim=options.in_dim, design_dim=design_dim, hid=hgat_hid, dropout=dropout
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
        r2_score.reset()
        total_loss = 0.0
        total_n = 0
        src_weight_epoch = compute_src_loss_weight(epoch, options)

        dl_src_iter = iter(dl_src)
        for xb_tgt, yb_tgt, cts_tgt, from_pins_tgt, to_pins_tgt in dl_tgt:
            update_encoder_this_step = bool(
                options.freeze_hgat
                or enc_update_interval <= 1
                or (global_step % enc_update_interval == 0)
            )
            xb_tgt = xb_tgt.to(device, non_blocking=True)
            yb_tgt = yb_tgt.to(device, non_blocking=True)
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
                        design_dim,
                        from_pins=from_pins_tgt,
                        to_pins=to_pins_tgt,
                        z_dict=z_dict_tgt,
                        graph_cache=graph_cache_tgt,
                        enc=enc,
                        dedup=options.dedup_z,
                        z_step_cache=z_step_cache,
                        use_batched_graph_encode=True,
                    )
                if not update_encoder_this_step:
                    zb_tgt = zb_tgt.detach()
                if z_noise_std > 0:
                    zb_tgt = zb_tgt + th.randn_like(zb_tgt) * z_noise_std
                    if hgat_l2_norm:
                        zb_tgt = th.nn.functional.normalize(zb_tgt, p=2, dim=1)
                pred_tgt = model(xb_tgt, zb_tgt, node="tgt")
                loss_tgt = loss_fn(pred_tgt, yb_tgt)

            for _ in range(max(1, options.sample_45_num)):
                try:
                    xb_src, yb_src, cts_src, from_pins_src, to_pins_src = next(dl_src_iter)
                except StopIteration:
                    dl_src_iter = iter(dl_src)
                    xb_src, yb_src, cts_src, from_pins_src, to_pins_src = next(dl_src_iter)
                xb_src = xb_src.to(device, non_blocking=True)
                yb_src = yb_src.to(device, non_blocking=True)
                src_n = len(yb_src)
                with th.cuda.amp.autocast(enabled=amp_on):
                    z_ctx = contextlib.nullcontext() if update_encoder_this_step else th.no_grad()
                    with z_ctx:
                        zb_src = build_z_batch(
                            cts_src,
                            device,
                            design_dim,
                            from_pins=from_pins_src,
                            to_pins=to_pins_src,
                            z_dict=z_dict_src,
                            graph_cache=graph_cache_src,
                            enc=enc,
                            dedup=options.dedup_z,
                            z_step_cache=z_step_cache,
                            use_batched_graph_encode=True,
                        )
                    if not update_encoder_this_step:
                        zb_src = zb_src.detach()
                    if z_noise_std > 0:
                        zb_src = zb_src + th.randn_like(zb_src) * z_noise_std
                        if hgat_l2_norm:
                            zb_src = th.nn.functional.normalize(zb_src, p=2, dim=1)
                    pred_src = model(xb_src, zb_src, node="src")
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
            if update_encoder_this_step and (not options.freeze_hgat):
                z_interval_cache = {}

            total_loss += total_loss_batch.item() * len(yb_tgt)
            total_n += len(yb_tgt)
            r2_score.update(pred_tgt.float(), yb_tgt.float())
            global_step += 1

        train_loss = total_loss / max(1, total_n)
        train_r2 = r2_score.compute().item() if total_n > 0 else 0.0
        is_plateau = isinstance(scheduler, th.optim.lr_scheduler.ReduceLROnPlateau)
        if scheduler is not None and (not is_plateau):
            scheduler.step()

        do_val_eval = ((epoch + 1) % val_eval_interval == 0) or ((epoch + 1) == int(options.num_epoch))
        val_loss, val_r2 = (None, None)
        stop_now = False
        if do_val_eval:
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

            delta_req = max(CKPT_R2_TIE_EPS_DEFAULT, early_stop_min_delta)
            better_r2 = val_r2 > (best_val + delta_req)
            tie_better = abs(val_r2 - best_val) <= delta_req and val_loss < best_val_loss
            if better_r2 or tie_better:
                best_val = val_r2
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
        if scheduler is not None and is_plateau and do_val_eval and (val_loss is not None):
            # For ReduceLROnPlateau, step on validation metric only.
            # This avoids over-decay when val_eval_interval > 1.
            scheduler.step(float(val_loss))

        test_loss, test_r2 = (None, None)
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
                test_loss, test_r2 = validate_cell(
                    test_dl,
                    enc_eval,
                    model_eval,
                    device,
                    design_dim,
                    z_dict=z_dict_tgt,
                    graph_cache=graph_cache_tgt,
                    dedup=options.dedup_z,
                    use_amp=use_amp,
                )
                best_test_r2 = test_r2
                best_test_loss = test_loss
                best_test_eval_epoch = epoch + 1

        cur_lr = float(optimizer.param_groups[-1]["lr"])
        if do_val_eval and val_loss is not None and val_r2 is not None:
            log_msg = (
                f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
                f"val_loss:{val_loss:.4f}, val_r2:{val_r2:.3f}, lr:{cur_lr:.2e}, src_w:{src_weight_epoch:.4f}"
            )
        else:
            log_msg = (
                f"e{epoch}, train_loss:{train_loss:.4f}, r2:{train_r2:.3f}, "
                f"lr:{cur_lr:.2e}, src_w:{src_weight_epoch:.4f}, val_eval:skip"
            )
        if test_loss is not None:
            log_msg += (
                f", best_test_loss:{test_loss:.4f}, best_test_r2:{test_r2:.3f}"
                f" (eval@e{epoch + 1}, ckpt_best@e{best_epoch})"
            )
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

    if best_epoch > 0:
        best_msg = (
            f"[Best] epoch:{best_epoch}, val_r2:{best_val:.4f}, "
            f"val_loss:{best_val_loss:.6f}"
        )
        if best_test_loss is not None:
            best_msg += (
                f", periodic_best_test_r2:{best_test_r2:.4f}, periodic_best_test_loss:{best_test_loss:.6f}, "
                f"test_eval_epoch:{best_test_eval_epoch}"
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
