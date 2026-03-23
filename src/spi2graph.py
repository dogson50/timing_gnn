import re
from typing import Dict, List, Tuple


_SI_SCALE = {
    "": 1.0,
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
    "t": 1e12,
}
_SUPPLY_ALIASES = {
    "0": "VSS",
    "GND": "VSS",
    "VGND": "VSS",
    "VSS": "VSS",
    "VCC": "VDD",
    "VDD": "VDD",
    "VPWR": "VDD",
}


def _read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _to_si(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.match(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+)?$",
        s,
    )
    if not m:
        return None
    base = float(m.group(1))
    unit = (m.group(2) or "").lower()
    scale = _SI_SCALE.get(unit)
    if scale is None:
        return base
    return base * scale


def _to_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        parsed = _to_si(value)
        if parsed is None:
            return float(default)
        return float(parsed)


def _merge_continuations(text: str) -> List[str]:
    merged: List[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                merged.append(current)
                current = ""
            continue
        if line.startswith("+"):
            addon = line[1:].strip()
            current = f"{current} {addon}".strip()
        else:
            if current:
                merged.append(current)
            current = line
    if current:
        merged.append(current)
    return merged


def _parse_param_tokens(rest: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for tok in [t for t in re.split(r"[,\s]+", rest.strip()) if t]:
        if "=" not in tok:
            continue
        key, value = tok.split("=", 1)
        params[key.lower().lstrip("$")] = value
    return params


def _classify_device(model: str) -> str:
    low = str(model).lower()
    if re.search(r"(?:^|_)nmos|nfet", low):
        return "nmos"
    if re.search(r"(?:^|_)pmos|pfet", low):
        return "pmos"
    if low.startswith("n"):
        return "nmos"
    if low.startswith("p"):
        return "pmos"
    return low


def _parse_transistor_line(line: str):
    m = re.match(
        r"^[Mm](\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$",
        line,
        re.I,
    )
    if not m:
        return None
    rest = m.group(7)
    params = _parse_param_tokens(rest)
    return {
        "name": "M" + m.group(1),
        "d": m.group(2),
        "g": m.group(3),
        "s": m.group(4),
        "b": m.group(5),
        "model": m.group(6),
        "type": _classify_device(m.group(6)),
        "W": _to_si(params.get("w")),
        "L": _to_si(params.get("l")),
        "nfin": _to_float(params.get("nfin"), 0.0),
        "m": _to_float(params.get("m"), 1.0),
        "x": _to_float(params.get("x"), 0.0),
        "y": _to_float(params.get("y"), 0.0),
    }


def _parse_passive_line(line: str):
    m = re.match(r"^([RrCc]\S*)\s+(\S+)\s+(\S+)\s+([^\s]+)", line)
    if not m:
        return None
    name = m.group(1)
    kind = "res" if name[0].lower() == "r" else "cap"
    value = _to_si(m.group(4))
    if value is None:
        return None
    return {
        "name": name,
        "kind": kind,
        "n1": m.group(2),
        "n2": m.group(3),
        "value": float(value),
    }


def _parse_instance_line(line: str):
    if not re.match(r"^[Xx]\S*", line):
        return None
    toks = line.split()
    if len(toks) < 3:
        return None
    return {
        "name": toks[0],
        "actuals": toks[1:-1],
        "subckt": toks[-1],
    }


def parse_transistors_spice(text):
    devs = []
    for line in _merge_continuations(text):
        if not line or line.startswith(("*", ";", "//")) or line.startswith("."):
            continue
        dev = _parse_transistor_line(line)
        if dev is not None:
            devs.append(dev)
    return devs


def parse_passive_parasitics_spice(text):
    elems = []
    for line in _merge_continuations(text):
        if not line or line.startswith(("*", ";", "//")) or line.startswith("."):
            continue
        elem = _parse_passive_line(line)
        if elem is not None:
            elems.append(elem)
    return elems


def _parse_subckt_blocks(text: str):
    blocks = {}
    current_name = None
    current_pins = []
    current_lines = []
    for raw in text.splitlines():
        begin = re.match(r"(?i)^\s*\.subckt\s+([^\s]+)\s*(.*)$", raw)
        if current_name is None and begin:
            current_name = begin.group(1)
            pin_str = begin.group(2).strip()
            current_pins = [p for p in pin_str.split() if p]
            current_lines = []
            continue
        if current_name is None:
            continue
        if re.match(r"(?i)^\s*\.ends\b", raw):
            body_text = "\n".join(current_lines)
            blocks[current_name] = {
                "name": current_name,
                "pins": current_pins,
                "body_lines": _merge_continuations(body_text),
            }
            current_name = None
            current_pins = []
            current_lines = []
            continue
        current_lines.append(raw.rstrip("\n"))
    return blocks


def parse_top_subckt_pins(text, cell_hint_regex=r"INV.*1"):
    blocks = _parse_subckt_blocks(text)
    items = list(blocks.items())
    for name, block in items:
        if re.match(cell_hint_regex, name, re.I):
            return name, list(block["pins"])
    if items:
        name, block = items[-1]
        return name, list(block["pins"])
    return None, []


def _lookup_pin_mapping(node: str, pin_map: Dict[str, str]):
    return pin_map.get(node) or pin_map.get(node.upper())


def _map_node_name(node: str, pin_map: Dict[str, str], scope: str) -> str:
    mapped = _lookup_pin_mapping(node, pin_map)
    if mapped is not None:
        return mapped
    alias = _SUPPLY_ALIASES.get(str(node).upper())
    if alias is not None:
        return alias
    return f"{scope}/{node}"


def flatten_subckt_hierarchy(sp_text: str, root_subckt: str):
    blocks = _parse_subckt_blocks(sp_text)
    root = blocks.get(root_subckt)
    if root is None:
        return [], [], []

    devs = []
    passives = []

    def walk(subckt_name: str, pin_map: Dict[str, str], scope: str):
        block = blocks.get(subckt_name)
        if block is None:
            return
        for line in block["body_lines"]:
            if not line or line.startswith(("*", ";", "//")) or line.startswith("."):
                continue

            dev = _parse_transistor_line(line)
            if dev is not None:
                mapped_dev = dict(dev)
                mapped_dev["name"] = f"{scope}/{dev['name']}"
                for key in ("d", "g", "s", "b"):
                    mapped_dev[key] = _map_node_name(dev[key], pin_map, scope)
                devs.append(mapped_dev)
                continue

            elem = _parse_passive_line(line)
            if elem is not None:
                mapped_elem = dict(elem)
                mapped_elem["name"] = f"{scope}/{elem['name']}"
                mapped_elem["n1"] = _map_node_name(elem["n1"], pin_map, scope)
                mapped_elem["n2"] = _map_node_name(elem["n2"], pin_map, scope)
                passives.append(mapped_elem)
                continue

            inst = _parse_instance_line(line)
            if inst is None:
                continue
            child = blocks.get(inst["subckt"])
            if child is None:
                continue
            child_pin_map: Dict[str, str] = {}
            for formal, actual in zip(child["pins"], inst["actuals"]):
                mapped_actual = _map_node_name(actual, pin_map, scope)
                child_pin_map[formal] = mapped_actual
                child_pin_map[formal.upper()] = mapped_actual
            child_scope = f"{scope}/{inst['name']}"
            walk(inst["subckt"], child_pin_map, child_scope)

    root_pin_map = {}
    for pin in root["pins"]:
        root_pin_map[pin] = pin
        root_pin_map[pin.upper()] = pin
    walk(root_subckt, root_pin_map, root_subckt)
    return devs, passives, list(root["pins"])


def extract_subckt_with_dependencies(sp_text: str, root_subckt: str) -> str:
    blocks = _parse_subckt_blocks(sp_text)
    if root_subckt not in blocks:
        return ""

    ordered = []
    visited = set()

    def dfs(name: str):
        if name in visited:
            return
        visited.add(name)
        block = blocks.get(name)
        if block is None:
            return
        body_text = "\n".join(block["body_lines"])
        for line in block["body_lines"]:
            inst = _parse_instance_line(line)
            if inst is not None and inst["subckt"] in blocks:
                dfs(inst["subckt"])
        ordered.append(f".subckt {name} {' '.join(block['pins'])}\n{body_text}\n.ends")

    dfs(root_subckt)
    return "\n\n".join(ordered)


def extract_wl_features(devs):
    if not devs:
        return {
            "wp_sum": 0.0,
            "wn_sum": 0.0,
            "wp_over_wn": 0.0,
            "num_pmos": 0.0,
            "num_nmos": 0.0,
            "p_nfin_sum": 0.0,
            "n_nfin_sum": 0.0,
        }

    wp = 0.0
    wn = 0.0
    num_p = 0.0
    num_n = 0.0
    p_nfin_sum = 0.0
    n_nfin_sum = 0.0

    for dev in devs:
        mult = max(float(dev.get("m", 1.0)), 0.0)
        if mult == 0.0:
            mult = 1.0
        w_eff = float(dev.get("W") or 0.0) * mult
        nfin_eff = float(dev.get("nfin") or 0.0) * mult
        if str(dev.get("type", "")).startswith("p"):
            wp += w_eff
            num_p += mult
            p_nfin_sum += nfin_eff
        elif str(dev.get("type", "")).startswith("n"):
            wn += w_eff
            num_n += mult
            n_nfin_sum += nfin_eff

    ratio = (wp / wn) if (wp > 0.0 and wn > 0.0) else 0.0
    return {
        "wp_sum": wp,
        "wn_sum": wn,
        "wp_over_wn": ratio,
        "num_pmos": num_p,
        "num_nmos": num_n,
        "p_nfin_sum": p_nfin_sum,
        "n_nfin_sum": n_nfin_sum,
    }
