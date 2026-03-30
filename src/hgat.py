import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from spi2graph import extract_subckt_with_dependencies, flatten_subckt_hierarchy, parse_passive_parasitics_spice, parse_top_subckt_pins, parse_transistors_spice

try:
    import dgl
    from dgl.nn import GATConv, HeteroGraphConv
except Exception:
    raise ImportError("DGL is required for HGAT. Please install dgl (CPU/GPU).")


HGAT_RELATIONS = ["gate_of", "sd_to", "back_sd", "res_to", "cap_to"]
HGAT_IN_DIM_MAP = {"NET": 10, "PMOS": 4, "NMOS": 4}
_POWER_VDD = {"VDD", "VPWR", "VCC"}
_POWER_VSS = {"0", "GND", "VGND", "VSS"}


def get_hgat_in_dim_map():
    return dict(HGAT_IN_DIM_MAP)


def _empty_index():
    return torch.zeros((0,), dtype=torch.int64)


def _norm_degree(arr):
    if arr.size == 0:
        return arr.astype(np.float32)
    max_v = float(max(1.0, arr.max()))
    return (np.log1p(arr) / np.log1p(max_v)).astype(np.float32)


def _net_alias(name: str):
    up = str(name).upper()
    if up in _POWER_VDD:
        return "VDD"
    if up in _POWER_VSS:
        return "VSS"
    return up


def _build_graph_meta(nets, top_pins):
    """Build lightweight graph metadata for arc-aware net focus."""
    top_pin_set = {str(p).upper() for p in (top_pins or [])}
    net_name_to_id = {}
    pin_to_net_id = {}
    for name, nid in sorted(nets.items(), key=lambda x: x[1]):
        up = str(name).upper()
        if up not in net_name_to_id:
            net_name_to_id[up] = int(nid)
        if up in top_pin_set and up not in pin_to_net_id:
            pin_to_net_id[up] = int(nid)
    return {
        "net_name_to_id": net_name_to_id,
        "pin_to_net_id": pin_to_net_id,
        "num_nets": int(len(nets)),
    }


class HGATBlock(nn.Module):
    def __init__(self, hid, rels, num_heads=1, dropout=0.1):
        super().__init__()
        self.conv = HeteroGraphConv(
            {r: GATConv(hid, hid, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout) for r in rels},
            aggregate="sum",
        )
        self.norm = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        self.hid = hid

    def _get_norm(self, nt, device):
        if nt not in self.norm:
            self.norm[nt] = nn.LayerNorm(self.hid)
            self.norm[nt].to(device)
        return self.norm[nt]

    def forward(self, g, h):
        out = self.conv(g, h)
        out = {k: v.mean(1) for k, v in out.items()}
        h_new = {}
        for nt, v in out.items():
            residual = h.get(nt)
            if residual is not None and residual.shape == v.shape:
                v = v + residual
            norm = self._get_norm(nt, v.device)
            v = norm(v)
            v = F.relu(v)
            v = self.dropout(v)
            h_new[nt] = v
        return h_new


class HGATDesignEncoder(nn.Module):
    def __init__(
        self,
        in_dim_map,
        hid=64,
        out=64,
        num_heads=1,
        num_layers=3,
        dropout=0.1,
        use_net_readout=False,
        type_attn_readout=False,
        l2_norm=False,
    ):
        super().__init__()
        self.embed = nn.ModuleDict({nt: nn.Linear(in_dim_map[nt], hid) for nt in in_dim_map})
        self.use_net_readout = bool(use_net_readout)
        self.type_attn_readout = bool(type_attn_readout)
        self.l2_norm = bool(l2_norm)
        self.summary_dim = 3 * hid
        self.layers = nn.ModuleList(
            [HGATBlock(hid=hid, rels=HGAT_RELATIONS, num_heads=num_heads, dropout=dropout) for _ in range(max(1, int(num_layers)))]
        )
        if self.type_attn_readout:
            self.type_gate = nn.Linear(self.summary_dim, 1)
        self.readout = nn.Sequential(
            nn.Linear(self.summary_dim, out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out, out),
        )

    def _summarize_hidden(self, h, net_focus=None):
        summaries = []
        ntypes = ["PMOS", "NMOS"] + (["NET"] if self.use_net_readout else [])
        for nt in ntypes:
            if nt in h and h[nt].shape[0] > 0:
                x = h[nt]
                # Optional arc-aware NET focus: summarize only selected NET nodes.
                if nt == "NET" and net_focus is not None:
                    if net_focus.dim() == 1 and net_focus.numel() == x.shape[0]:
                        mask = net_focus > 0
                        if bool(mask.any()):
                            x = x[mask]
                mean_v = x.mean(dim=0, keepdim=True)
                max_v = x.max(dim=0, keepdim=True).values
                std_v = x.std(dim=0, keepdim=True, unbiased=False)
                summaries.append(torch.cat([mean_v, max_v, std_v], dim=1))
        if len(summaries) == 0:
            for v in h.values():
                mean_v = v.mean(dim=0, keepdim=True)
                max_v = v.max(dim=0, keepdim=True).values
                std_v = v.std(dim=0, keepdim=True, unbiased=False)
                summaries.append(torch.cat([mean_v, max_v, std_v], dim=1))

        s = torch.cat(summaries, dim=0)
        if self.type_attn_readout and s.shape[0] > 1:
            w = torch.softmax(self.type_gate(s), dim=0)
            z = (w * s).sum(dim=0)
        else:
            z = s.mean(dim=0)
        return z

    def forward(self, g, feats, net_focus=None):
        h = {nt: self.embed[nt](feats[nt]) for nt in feats}
        for layer in self.layers:
            h = layer(g, h)
        z = self._summarize_hidden(h, net_focus=net_focus)
        z = self.readout(z)
        if self.l2_norm:
            z = F.normalize(z, dim=0)
        return z


def build_dgl_graph_from_devs(devs, top_pins, passives=None, return_meta=False):
    passives = passives or []
    nets = {}

    def net_id(name):
        if name not in nets:
            nets[name] = len(nets)
        return nets[name]

    p_devs = [d for d in devs if str(d.get("type", "")).startswith("p")]
    n_devs = [d for d in devs if str(d.get("type", "")).startswith("n")]
    p_count = len(p_devs)
    n_count = len(n_devs)

    gate_src_p, gate_dst_p = [], []
    gate_src_n, gate_dst_n = [], []
    sd_src_p, sd_dst_p = [], []
    sd_src_n, sd_dst_n = [], []

    for idx, dev in enumerate(p_devs):
        gate_src_p.append(net_id(dev["g"]))
        gate_dst_p.append(idx)
        sd_src_p.extend([idx, idx, idx])
        sd_dst_p.extend([net_id(dev["s"]), net_id(dev["d"]), net_id(dev["b"])])

    for idx, dev in enumerate(n_devs):
        gate_src_n.append(net_id(dev["g"]))
        gate_dst_n.append(idx)
        sd_src_n.extend([idx, idx, idx])
        sd_dst_n.extend([net_id(dev["s"]), net_id(dev["d"]), net_id(dev["b"])])

    res_src, res_dst = [], []
    cap_src, cap_dst = [], []
    for elem in passives:
        n1 = net_id(elem["n1"])
        n2 = net_id(elem["n2"])
        if n1 == n2:
            continue
        if elem["kind"] == "res":
            res_src.extend([n1, n2])
            res_dst.extend([n2, n1])
        elif elem["kind"] == "cap":
            cap_src.extend([n1, n2])
            cap_dst.extend([n2, n1])

    data_dict = {
        ("NET", "gate_of", "PMOS"): (torch.tensor(gate_src_p, dtype=torch.int64), torch.tensor(gate_dst_p, dtype=torch.int64))
        if gate_src_p
        else (_empty_index(), _empty_index()),
        ("PMOS", "sd_to", "NET"): (torch.tensor(sd_src_p, dtype=torch.int64), torch.tensor(sd_dst_p, dtype=torch.int64))
        if sd_src_p
        else (_empty_index(), _empty_index()),
        ("NET", "back_sd", "PMOS"): (torch.tensor(sd_dst_p, dtype=torch.int64), torch.tensor(sd_src_p, dtype=torch.int64))
        if sd_dst_p
        else (_empty_index(), _empty_index()),
        ("NET", "gate_of", "NMOS"): (torch.tensor(gate_src_n, dtype=torch.int64), torch.tensor(gate_dst_n, dtype=torch.int64))
        if gate_src_n
        else (_empty_index(), _empty_index()),
        ("NMOS", "sd_to", "NET"): (torch.tensor(sd_src_n, dtype=torch.int64), torch.tensor(sd_dst_n, dtype=torch.int64))
        if sd_src_n
        else (_empty_index(), _empty_index()),
        ("NET", "back_sd", "NMOS"): (torch.tensor(sd_dst_n, dtype=torch.int64), torch.tensor(sd_src_n, dtype=torch.int64))
        if sd_dst_n
        else (_empty_index(), _empty_index()),
        ("NET", "res_to", "NET"): (torch.tensor(res_src, dtype=torch.int64), torch.tensor(res_dst, dtype=torch.int64))
        if res_src
        else (_empty_index(), _empty_index()),
        ("NET", "cap_to", "NET"): (torch.tensor(cap_src, dtype=torch.int64), torch.tensor(cap_dst, dtype=torch.int64))
        if cap_src
        else (_empty_index(), _empty_index()),
    }

    g = dgl.heterograph(
        data_dict,
        num_nodes_dict={"NET": len(nets), "PMOS": p_count, "NMOS": n_count},
    )

    dev_deg = np.zeros(len(nets), dtype=np.float32)
    for nid in gate_src_p + gate_src_n + sd_dst_p + sd_dst_n:
        dev_deg[int(nid)] += 1.0

    par_deg = np.zeros(len(nets), dtype=np.float32)
    cap_ground = np.zeros(len(nets), dtype=np.float32)
    cap_couple = np.zeros(len(nets), dtype=np.float32)
    res_sum = np.zeros(len(nets), dtype=np.float32)
    res_gsum = np.zeros(len(nets), dtype=np.float32)

    for elem in passives:
        n1 = nets[elem["n1"]]
        n2 = nets[elem["n2"]]
        par_deg[n1] += 1.0
        par_deg[n2] += 1.0
        if elem["kind"] == "res":
            value = max(float(elem["value"]), 1e-12)
            res_sum[n1] += value
            res_sum[n2] += value
            conductance = 1.0 / value
            res_gsum[n1] += conductance
            res_gsum[n2] += conductance
        elif elem["kind"] == "cap":
            value = max(float(elem["value"]), 0.0)
            a1 = _net_alias(elem["n1"])
            a2 = _net_alias(elem["n2"])
            if a1 in ("VDD", "VSS") and a2 not in ("VDD", "VSS"):
                cap_ground[n2] += value
            elif a2 in ("VDD", "VSS") and a1 not in ("VDD", "VSS"):
                cap_ground[n1] += value
            else:
                cap_couple[n1] += value
                cap_couple[n2] += value

    dev_deg_norm = _norm_degree(dev_deg)
    par_deg_norm = _norm_degree(par_deg)
    top_pin_set = {str(p).upper() for p in (top_pins or [])}

    f_net = []
    for name, nid in sorted(nets.items(), key=lambda x: x[1]):
        alias = _net_alias(name)
        is_vdd = 1.0 if alias == "VDD" else 0.0
        is_vss = 1.0 if alias == "VSS" else 0.0
        is_top_pin = 1.0 if str(name).upper() in top_pin_set else 0.0
        is_internal = 1.0 if (is_vdd == 0.0 and is_vss == 0.0 and is_top_pin == 0.0) else 0.0
        f_net.append(
            [
                is_vdd,
                is_vss,
                is_top_pin,
                is_internal,
                float(dev_deg_norm[nid]),
                float(par_deg_norm[nid]),
                float(np.log10(cap_ground[nid] + 1e-18)),
                float(np.log10(cap_couple[nid] + 1e-18)),
                float(np.log10(res_sum[nid] + 1e-12)),
                float(np.log10(res_gsum[nid] + 1e-12)),
            ]
        )
    if f_net:
        f_net = torch.tensor(np.array(f_net, dtype=np.float32))
    else:
        f_net = torch.zeros((0, HGAT_IN_DIM_MAP["NET"]), dtype=torch.float32)

    def mos_feats(list_dev):
        arr = []
        for dev in list_dev:
            raw_w = float(dev.get("W") or 0.0)
            raw_l = float(dev.get("L") or 0.0)
            raw_nfin = max(float(dev.get("nfin") or 0.0), 0.0)
            raw_m = max(float(dev.get("m") or 1.0), 1.0)
            arr.append(
                [
                    float(np.log10(raw_w + 1e-12)),
                    float(np.log10(raw_l + 1e-12)),
                    float(np.log1p(raw_nfin)),
                    float(np.log(max(raw_m, 1.0))),
                ]
            )
        if not arr:
            return torch.zeros((0, HGAT_IN_DIM_MAP["PMOS"]), dtype=torch.float32)
        return torch.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats(p_devs)
    f_n = mos_feats(n_devs)
    feats = {"NET": f_net, "PMOS": f_p, "NMOS": f_n}
    if return_meta:
        return g, feats, get_hgat_in_dim_map(), _build_graph_meta(nets, top_pins)
    return g, feats, get_hgat_in_dim_map()


def build_dgl_graph_from_devs_rich(devs, top_pins, passives=None, return_meta=False):
    """Compatibility helper for step5a/step13/step14 rich-feature experiments.

    NET features: 7 dims
    MOS features: 8 dims
    """
    passives = passives or []
    nets = {}

    def net_id(name):
        if name not in nets:
            nets[name] = len(nets)
        return nets[name]

    p_devs = []
    n_devs = []
    gate_src_p, gate_dst_p = [], []
    gate_src_n, gate_dst_n = [], []
    sd_src_p, sd_dst_p = [], []
    sd_src_n, sd_dst_n = [], []

    for d in devs:
        dev_type = str(d.get("type", "")).lower()
        if dev_type.startswith("p"):
            idx = len(p_devs)
            p_devs.append(d)
            gate_src_p.append(net_id(d["g"]))
            gate_dst_p.append(idx)
            sd_src_p.extend([idx, idx])
            sd_dst_p.extend([net_id(d["s"]), net_id(d["d"])])
        elif dev_type.startswith("n"):
            idx = len(n_devs)
            n_devs.append(d)
            gate_src_n.append(net_id(d["g"]))
            gate_dst_n.append(idx)
            sd_src_n.extend([idx, idx])
            sd_dst_n.extend([net_id(d["s"]), net_id(d["d"])])

    p_count, n_count = len(p_devs), len(n_devs)
    res_src, res_dst = [], []
    cap_src, cap_dst = [], []
    for elem in passives:
        n1 = net_id(elem["n1"])
        n2 = net_id(elem["n2"])
        if n1 == n2:
            continue
        kind = str(elem.get("kind", "")).lower()
        if kind == "res":
            res_src.extend([n1, n2])
            res_dst.extend([n2, n1])
        elif kind == "cap":
            cap_src.extend([n1, n2])
            cap_dst.extend([n2, n1])
    data_dict = {}
    if p_count > 0:
        data_dict[("NET", "gate_of", "PMOS")] = (
            torch.tensor(gate_src_p, dtype=torch.int64),
            torch.tensor(gate_dst_p, dtype=torch.int64),
        )
        data_dict[("PMOS", "sd_to", "NET")] = (
            torch.tensor(sd_src_p, dtype=torch.int64),
            torch.tensor(sd_dst_p, dtype=torch.int64),
        )
        data_dict[("NET", "back_sd", "PMOS")] = (
            torch.tensor(sd_dst_p, dtype=torch.int64),
            torch.tensor(sd_src_p, dtype=torch.int64),
        )
    if n_count > 0:
        data_dict[("NET", "gate_of", "NMOS")] = (
            torch.tensor(gate_src_n, dtype=torch.int64),
            torch.tensor(gate_dst_n, dtype=torch.int64),
        )
        data_dict[("NMOS", "sd_to", "NET")] = (
            torch.tensor(sd_src_n, dtype=torch.int64),
            torch.tensor(sd_dst_n, dtype=torch.int64),
        )
        data_dict[("NET", "back_sd", "NMOS")] = (
            torch.tensor(sd_dst_n, dtype=torch.int64),
            torch.tensor(sd_src_n, dtype=torch.int64),
        )
    if res_src:
        data_dict[("NET", "res_to", "NET")] = (
            torch.tensor(res_src, dtype=torch.int64),
            torch.tensor(res_dst, dtype=torch.int64),
        )
    if cap_src:
        data_dict[("NET", "cap_to", "NET")] = (
            torch.tensor(cap_src, dtype=torch.int64),
            torch.tensor(cap_dst, dtype=torch.int64),
        )

    g = dgl.heterograph(
        data_dict,
        num_nodes_dict={"NET": len(nets), "PMOS": p_count, "NMOS": n_count},
    )

    gate_deg = np.zeros(len(nets), dtype=np.float32)
    sd_deg = np.zeros(len(nets), dtype=np.float32)
    for nid in gate_src_p + gate_src_n:
        gate_deg[int(nid)] += 1.0
    for nid in sd_dst_p + sd_dst_n:
        sd_deg[int(nid)] += 1.0
    par_deg = np.zeros(len(nets), dtype=np.float32)
    for elem in passives:
        n1 = nets.get(elem["n1"])
        n2 = nets.get(elem["n2"])
        if n1 is None or n2 is None:
            continue
        par_deg[n1] += 1.0
        par_deg[n2] += 1.0
    deg = gate_deg + sd_deg + par_deg

    max_deg = float(max(1.0, deg.max() if deg.size > 0 else 1.0))
    max_gate_deg = float(max(1.0, gate_deg.max() if gate_deg.size > 0 else 1.0))
    max_sd_deg = float(max(1.0, sd_deg.max() if sd_deg.size > 0 else 1.0))

    top_pin_set = {str(p).upper() for p in (top_pins or [])}
    rail_set = {"VDD", "VPWR", "VCC", "VSS", "VGND", "GND"}
    f_net = []
    for name, nid in sorted(nets.items(), key=lambda kv: kv[1]):
        up = str(name).upper()
        is_vdd = 1.0 if up in ("VDD", "VPWR", "VCC") else 0.0
        is_vss = 1.0 if up in ("VSS", "VGND", "GND") else 0.0
        is_top_pin = 1.0 if up in top_pin_set else 0.0
        is_internal = 1.0 if (up not in top_pin_set and up not in rail_set) else 0.0
        deg_norm = float(np.log1p(deg[nid]) / np.log1p(max_deg))
        gate_norm = float(np.log1p(gate_deg[nid]) / np.log1p(max_gate_deg))
        sd_norm = float(np.log1p(sd_deg[nid]) / np.log1p(max_sd_deg))
        f_net.append([is_vdd, is_vss, is_top_pin, deg_norm, gate_norm, sd_norm, is_internal])
    if f_net:
        f_net = torch.tensor(np.array(f_net, dtype=np.float32))
    else:
        f_net = torch.zeros((0, 7), dtype=torch.float32)

    def safe_log10(v, eps=1e-12):
        return float(np.log10(max(float(v), eps)))

    def mos_feats(dev_list):
        arr = []
        rail_set_local = {"VDD", "VPWR", "VCC", "VSS", "VGND", "GND"}
        for d in dev_list:
            raw_w = float(d.get("W")) if d.get("W") is not None else 2.0e-8
            raw_l = float(d.get("L")) if d.get("L") is not None else 2.0e-8
            nfin = d.get("nfin")
            nf = d.get("nf")
            m_mult = d.get("m")
            nfin_eff = float(nfin if nfin is not None else (nf if nf is not None else 1.0))
            nfin_eff = max(nfin_eff, 1.0)
            m_eff = float(m_mult if m_mult is not None else 1.0)
            m_eff = max(m_eff, 1.0)

            wl_ratio = raw_w / max(raw_l, 1e-12)
            area = raw_w * raw_l

            b = str(d.get("b", "")).upper()
            s = str(d.get("s", "")).upper()
            body_tied_source = 1.0 if b == s else 0.0
            body_is_rail = 1.0 if b in rail_set_local else 0.0

            arr.append(
                [
                    safe_log10(raw_w),
                    safe_log10(raw_l),
                    safe_log10(wl_ratio),
                    safe_log10(area),
                    safe_log10(nfin_eff),
                    safe_log10(m_eff),
                    body_tied_source,
                    body_is_rail,
                ]
            )

        if not arr:
            return torch.zeros((0, 8), dtype=torch.float32)
        return torch.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats(p_devs)
    f_n = mos_feats(n_devs)
    feats = {"NET": f_net, "PMOS": f_p, "NMOS": f_n}
    in_dim_map = {"NET": 7, "PMOS": 8, "NMOS": 8}
    if return_meta:
        return g, feats, in_dim_map, _build_graph_meta(nets, top_pins)
    return g, feats, in_dim_map


def build_graph_from_spice(spi_path):
    text = open(spi_path, "r", encoding="utf-8", errors="ignore").read()
    return build_graph_from_spice_text(text)


def build_graph_from_spice_text(sp_text, root_subckt=None, return_meta=False):
    if root_subckt:
        devs, passives, pins = flatten_subckt_hierarchy(sp_text, root_subckt)
        if devs or passives:
            return build_dgl_graph_from_devs(devs, pins, passives=passives, return_meta=return_meta)
        sub_text = extract_subckt_with_dependencies(sp_text, root_subckt)
        if sub_text:
            devs = parse_transistors_spice(sub_text)
            passives = parse_passive_parasitics_spice(sub_text)
            _, pins = parse_top_subckt_pins(sub_text)
            return build_dgl_graph_from_devs(devs, pins, passives=passives, return_meta=return_meta)
        return build_dgl_graph_from_devs([], [], passives=[], return_meta=return_meta)
    devs = parse_transistors_spice(sp_text)
    passives = parse_passive_parasitics_spice(sp_text)
    _, pins = parse_top_subckt_pins(sp_text)
    return build_dgl_graph_from_devs(devs, pins, passives=passives, return_meta=return_meta)
