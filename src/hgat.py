import torch
import torch.nn as nn
from torch.nn import functional as F
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
try:
    import dgl
    from dgl.nn import HeteroGraphConv, GATConv
except Exception as e:
    raise ImportError("DGL is required for HGAT. Please install dgl (CPU/GPU).")


class HGATBlock(nn.Module):
    def __init__(self, hid, rels, num_heads=1, dropout=0.1):
        super().__init__()
        self.conv = HeteroGraphConv(
            {r: GATConv(hid, hid, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout) for r in rels},
            aggregate='sum'
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
    ):
        super().__init__()
        rels = ["gate_of", "sd_to", "back_sd"]
        self.embed = nn.ModuleDict({nt: nn.Linear(in_dim_map[nt], hid) for nt in in_dim_map})
        self.use_net_readout = bool(use_net_readout)
        self.type_attn_readout = bool(type_attn_readout)
        self.summary_dim = 3 * hid
        self.layers = nn.ModuleList([
            HGATBlock(hid=hid, rels=rels, num_heads=num_heads, dropout=dropout)
            for _ in range(max(1, int(num_layers)))
        ])
        if self.type_attn_readout:
            self.type_gate = nn.Linear(self.summary_dim, 1)
        readout_in = self.summary_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_in, out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out, out)
        )

    def _summarize_hidden(self, h):
        summaries = []
        ntypes = ["PMOS", "NMOS"] + (["NET"] if self.use_net_readout else [])
        for nt in ntypes:
            if nt in h and h[nt].shape[0] > 0:
                x = h[nt]
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

    def forward(self, g, feats):
        h = {nt: self.embed[nt](feats[nt]) for nt in feats}
        for layer in self.layers:
            h = layer(g, h)
        z = self._summarize_hidden(h)
        z = self.readout(z)
        z = F.normalize(z, dim=0)
        return z
# hgat.py - 追加以下辅助函数（直接复用 train_hgat.py 的逻辑）
def build_dgl_graph_from_devs(devs, top_pins):
    import dgl, torch, numpy as np
    nets = {}
    def net_id(n):
        if n not in nets: nets[n] = len(nets)
        return nets[n]

    p_count=n_count=0
    gate_src_p= []; gate_dst_p= []
    gate_src_n= []; gate_dst_n= []
    sd_src_p=   []; sd_dst_p=   []
    sd_src_n=   []; sd_dst_n=   []

    for d in devs:
        if d["type"].startswith("p"):
            mid = p_count; p_count += 1
            gate_src_p.append(net_id(d["g"])); gate_dst_p.append(mid)
            sd_src_p.extend([mid, mid]); sd_dst_p.extend([net_id(d["s"]), net_id(d["d"])])
        else:
            mid = n_count; n_count += 1
            gate_src_n.append(net_id(d["g"])); gate_dst_n.append(mid)
            sd_src_n.extend([mid, mid]); sd_dst_n.extend([net_id(d["s"]), net_id(d["d"])])

    data_dict = {}
    if p_count > 0:
        data_dict[('NET','gate_of','PMOS')] = (torch.tensor(gate_src_p), torch.tensor(gate_dst_p))
        data_dict[('PMOS','sd_to','NET')]   = (torch.tensor(sd_src_p),  torch.tensor(sd_dst_p))
        data_dict[('NET','back_sd','PMOS')] = (torch.tensor(sd_dst_p),  torch.tensor(sd_src_p))
    if n_count > 0:
        data_dict[('NET','gate_of','NMOS')] = (torch.tensor(gate_src_n), torch.tensor(gate_dst_n))
        data_dict[('NMOS','sd_to','NET')]   = (torch.tensor(sd_src_n),  torch.tensor(sd_dst_n))
        data_dict[('NET','back_sd','NMOS')] = (torch.tensor(sd_dst_n),  torch.tensor(sd_src_n))

    g = dgl.heterograph(data_dict, num_nodes_dict={'NET': len(nets), 'PMOS': p_count, 'NMOS': n_count})

    deg = np.zeros(len(nets), dtype=np.float32)
    for src, dst in ((sd_src_p, sd_dst_p), (sd_src_n, sd_dst_n)):
        for nid in dst:
            deg[int(nid)] += 1.0
    for src, dst in ((gate_src_p, gate_dst_p), (gate_src_n, gate_dst_n)):
        for nid in src:
            deg[int(nid)] += 1.0
    max_deg = float(max(1.0, deg.max() if deg.size > 0 else 1.0))

    top_pin_set = set(str(p).upper() for p in (top_pins or []))
    f_net = []
    for name, nid in sorted(nets.items(), key=lambda x:x[1]):
        up = name.upper()
        is_vdd = 1.0 if up in ("VDD", "VPWR", "VCC") else 0.0
        is_vss = 1.0 if up in ("VSS", "VGND", "GND") else 0.0
        is_top_pin = 1.0 if up in top_pin_set else 0.0
        deg_norm = float(np.log1p(deg[nid]) / np.log1p(max_deg))
        f_net.append([is_vdd, is_vss, is_top_pin, deg_norm])
    f_net = torch.tensor(np.array(f_net, dtype=np.float32))

    def mos_feats(list_dev):
        arr = []
        for d in list_dev:
            raw_W = d["W"] if d["W"] is not None else 1e-7
            raw_L = d["L"] if d["L"] is not None else 1e-7
            w_feat = np.log10(raw_W + 1e-9)
            l_feat = np.log10(raw_L + 1e-9)
            arr.append([w_feat, l_feat])
        if len(arr)==0:
            return torch.zeros((0,2), dtype=torch.float32)
        return torch.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats([d for d in devs if d["type"].startswith("p")])
    f_n = mos_feats([d for d in devs if d["type"].startswith("n")])

    feats = {'NET': f_net, 'PMOS': f_p, 'NMOS': f_n}
    in_dim_map = {'NET': f_net.shape[1] if f_net.numel() else 4, 'PMOS': 2, 'NMOS': 2}
    return g, feats, in_dim_map

def build_graph_from_spice(spi_path):
    text = open(spi_path,"r",encoding="utf-8",errors="ignore").read()
    devs = parse_transistors_spice(text)
    _, pins = parse_top_subckt_pins(text)
    return build_dgl_graph_from_devs(devs, pins)
