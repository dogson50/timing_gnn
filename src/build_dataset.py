# === Python´úÂëÎÄ¼þ: build_dataset.py (ÒÑÐÞ¸´ Scaler Éú³É) ===

import argparse
import os
import json
import re
import pickle
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Set, Optional

import numpy as np
import pandas as pd

# ¼ÙÉèÕâÐ©¿âÎÄ¼þºÍÄã±¾µØ»·¾³Ò»ÖÂ
from parse_lib import cell_type_to_topology_group, normalize_arc_condition, parse_cell_arcs
from spi2graph import extract_wl_features, flatten_subckt_hierarchy, parse_transistors_spice
from options import get_options

# ======================================================
# ÅäÖÃ£ºÒª±£ÁôµÄ cell ÀàÐÍ
# ======================================================

TARGET_CELL_TYPES = [
    "AND2X2", "AND2X4", "AND3X1", "AND3X2", "AND3X4", "AND4X1", "AND4X2",
    "BUFX2", "BUFX4", "BUFX8", "BUFX16",
    "INVX1", "INVX2", "INVX4", "INVX8",
    "NAND2X1", "NAND2X2", "NAND3X1", "NAND3X2",
    "NOR2X1", "NOR2X2", "NOR3X1", "NOR3X2",
    "OR2X2", "OR2X4", "OR3X1", "OR3X2", "OR3X4", "OR4X1", "OR4X2",
    "XNOR2X2", "XOR2X2",
]

# ±ØÐëÓë train_hgat.py ÖÐµÄÁÐ±í±£³ÖÒ»ÖÂ£¬ÓÃÓÚÉú³É scaler_stats.json
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

# -------- Ô´Óò£¨Nangate£©Ã¿¸ö cell ¶ÔÓ¦Ò»¸ö SPI ÎÄ¼þ --------
SRC_CELL_SPI_FILES = {
    "AND2X2": "AND2_X2_lpe.spi",
    "AND2X4": "AND2_X4_lpe.spi",
    "AND3X1": "AND3_X1_lpe.spi",
    "AND3X2": "AND3_X2_lpe.spi",
    "AND3X4": "AND3_X4_lpe.spi",
    "AND4X1": "AND4_X1_lpe.spi",
    "AND4X2": "AND4_X2_lpe.spi",
    "BUFX2": "BUF_X2_lpe.spi",
    "BUFX4": "BUF_X4_lpe.spi",
    "BUFX8": "BUF_X8_lpe.spi",
    "BUFX16": "BUF_X16_lpe.spi",
    "INVX1": "INV_X1_lpe.spi",
    "INVX2": "INV_X2_lpe.spi",
    "INVX4": "INV_X4_lpe.spi",
    "INVX8": "INV_X8_lpe.spi",
    "NAND2X1": "NAND2_X1_lpe.spi",
    "NAND2X2": "NAND2_X2_lpe.spi",
    "NAND3X1": "NAND3_X1_lpe.spi",
    "NAND3X2": "NAND3_X2_lpe.spi",
    "NOR2X1": "NOR2_X1_lpe.spi",
    "NOR2X2": "NOR2_X2_lpe.spi",
    "NOR3X1": "NOR3_X1_lpe.spi",
    "NOR3X2": "NOR3_X2_lpe.spi",
    "OR2X2": "OR2_X2_lpe.spi",
    "OR2X4": "OR2_X4_lpe.spi",
    "OR3X1": "OR3_X1_lpe.spi",
    "OR3X2": "OR3_X2_lpe.spi",
    "OR3X4": "OR3_X4_lpe.spi",
    "OR4X1": "OR4_X1_lpe.spi",
    "OR4X2": "OR4_X2_lpe.spi",
    "XOR2X2": "XOR2_X2_lpe.spi",
    "XNOR2X2": "XNOR2_X2_lpe.spi",
}

# -------- Ä¿±êÓò£¨ASAP7£©´ó SP ÎÄ¼þÀïµÄ subckt Ãû --------
ASAP7_CELL_SUBCKT = {
    "AND2X2": "AND2x2_ASAP7_6t_L",
    "AND2X4": "AND2x4_ASAP7_6t_L",
    "AND3X1": "AND3x1_ASAP7_6t_L",
    "AND3X2": "AND3x2_ASAP7_6t_L",
    "AND3X4": "AND3x4_ASAP7_6t_L",
    "AND4X1": "AND4x1_ASAP7_6t_L",
    "AND4X2": "AND4x2_ASAP7_6t_L",
    "BUFX2": "BUFx2_ASAP7_6t_L",
    "BUFX4": "BUFx4_ASAP7_6t_L",
    "BUFX8": "BUFx8_ASAP7_6t_L",
    "BUFX16": "BUFx16q_ASAP7_6t_L",
    "INVX1": "INVx1_ASAP7_6t_L",
    "INVX2": "INVx2_ASAP7_6t_L",
    "INVX4": "INVx4_ASAP7_6t_L",
    "INVX8": "INVx8_ASAP7_6t_L",
    "NAND2X1": "NAND2x1_ASAP7_6t_L",
    "NAND2X2": "NAND2x2_ASAP7_6t_L",
    "NAND3X1": "NAND3x1_ASAP7_6t_L",
    "NAND3X2": "NAND3x2_ASAP7_6t_L",
    "NOR2X1": "NOR2x1_ASAP7_6t_L",
    "NOR2X2": "NOR2x2_ASAP7_6t_L",
    "NOR3X1": "NOR3x1_ASAP7_6t_L",
    "NOR3X2": "NOR3x2_ASAP7_6t_L",
    "OR2X2": "OR2x2_ASAP7_6t_L",
    "OR2X4": "OR2x4_ASAP7_6t_L",
    "OR3X1": "OR3x1_ASAP7_6t_L",
    "OR3X2": "OR3x2_ASAP7_6t_L",
    "OR3X4": "OR3x4_ASAP7_6t_L",
    "OR4X1": "OR4x1_ASAP7_6t_L",
    "OR4X2": "OR4x2_ASAP7_6t_L",
    "XOR2X2": "XOR2x2_ASAP7_6t_L",
    "XNOR2X2": "XNOR2x2_ASAP7_6t_L",
}


def _format_target_group_id(
        cell_name: str,
        from_pin: str,
        timing_sense: str,
        arc_cond: str,
        pol: str,
        tech: str = "ASAP7",
        to_pin: str = "Y",
        voltage: float = 0.7,
        temp: float = 25.0,
) -> str:
    return "|".join([
        str(tech),
        str(cell_name),
        str(from_pin),
        str(to_pin),
        str(timing_sense),
        str(arc_cond),
        str(pol),
        f"v={float(voltage):.6g}",
        f"t={float(temp):.6g}",
    ])


CURATED_TABLE_GROUP_MANUAL_NAME = "asap7_curated_v1"
CURATED_TABLE_GROUP_MANUAL_SPECS = {
    "train": [
        ("AND2x4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
        ("AND3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "fall"),
        ("AND4x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
        ("BUFx4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
        ("INVx4_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
        ("NAND2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
        ("NAND3x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
        ("NOR2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
        ("NOR3x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
        ("OR2x4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
        ("OR3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
        ("OR4x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
        ("XNOR2x2_ASAP7_6t_L", "A", "negative_unate", "!B", "fall"),
        ("XOR2x2_ASAP7_6t_L", "A", "negative_unate", "B", "fall"),
        ("AND4x2_ASAP7_6t_L", "D", "positive_unate", "<NONE>", "rise"),
        ("NAND3x2_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
        ("NOR3x2_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "fall"),
        ("OR3x4_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
        ("OR4x2_ASAP7_6t_L", "D", "positive_unate", "<NONE>", "fall"),
        ("XNOR2x2_ASAP7_6t_L", "B", "positive_unate", "A", "rise"),
        ("XOR2x2_ASAP7_6t_L", "B", "positive_unate", "!A", "rise"),
    ],
    "val": [
        ("AND2x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
        ("AND3x1_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
        ("AND4x1_ASAP7_6t_L", "D", "positive_unate", "<NONE>", "rise"),
        ("BUFx2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
        ("INVx2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
        ("NAND2x1_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "rise"),
        ("NAND3x1_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
        ("NOR2x1_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "rise"),
        ("NOR3x1_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
        ("OR2x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "fall"),
        ("OR3x1_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
        ("OR4x1_ASAP7_6t_L", "D", "positive_unate", "<NONE>", "fall"),
        ("XNOR2x2_ASAP7_6t_L", "B", "negative_unate", "!A", "fall"),
        ("XOR2x2_ASAP7_6t_L", "A", "positive_unate", "!B", "fall"),
        ("AND3x4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
        ("AND4x1_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
        ("NAND3x1_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
        ("NOR3x1_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
        ("OR3x4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
        ("XNOR2x2_ASAP7_6t_L", "A", "positive_unate", "B", "rise"),
        ("XOR2x2_ASAP7_6t_L", "B", "negative_unate", "A", "fall"),
    ],
}
CURATED_TABLE_GROUP_MANUAL_GROUPS = {
    split: [_format_target_group_id(*spec) for spec in specs]
    for split, specs in CURATED_TABLE_GROUP_MANUAL_SPECS.items()
}

CURATED_TABLE_GROUP_MANUAL_TRAIN_NAME = "asap7_curated_train_v2"
CURATED_TABLE_GROUP_MANUAL_TRAIN_SPECS = [
    ("AND2x4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("AND3x1_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("AND3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
    ("AND4x1_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("BUFx2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("BUFx8_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("INVx1_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
    ("INVx4_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
    ("NAND2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
    ("NAND3x1_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("NAND3x2_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
    ("NOR2x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("NOR3x1_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
    ("OR2x2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("OR3x1_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("OR3x4_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
    ("OR4x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "fall"),
    ("XNOR2x2_ASAP7_6t_L", "A", "negative_unate", "!B", "fall"),
    ("XNOR2x2_ASAP7_6t_L", "A", "positive_unate", "B", "rise"),
    ("XOR2x2_ASAP7_6t_L", "B", "negative_unate", "A", "fall"),
    ("XOR2x2_ASAP7_6t_L", "B", "positive_unate", "!A", "rise"),
]
CURATED_TABLE_GROUP_MANUAL_TRAIN_GROUPS = [
    _format_target_group_id(*spec) for spec in CURATED_TABLE_GROUP_MANUAL_TRAIN_SPECS
]

CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_NAME = "asap7_curated_train_test_v1"
CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_RATIOS = (1.0 / 6.0, 0.0, 5.0 / 6.0)
CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_SPECS = [
    ("AND2x2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("AND2x4_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
    ("AND3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "fall"),
    ("AND3x1_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
    ("AND4x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
    ("AND4x1_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("BUFx4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("BUFx2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("INVx4_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
    ("INVx2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
    ("NAND2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
    ("NAND2x1_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "rise"),
    ("NAND3x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("NAND3x1_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
    ("NOR2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
    ("NOR2x1_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("NOR3x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("OR2x2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("OR2x4_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "fall"),
    ("OR3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
    ("OR3x1_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
    ("OR4x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
    ("XNOR2x2_ASAP7_6t_L", "A", "positive_unate", "B", "rise"),
    ("XNOR2x2_ASAP7_6t_L", "B", "negative_unate", "!A", "fall"),
    ("XOR2x2_ASAP7_6t_L", "B", "negative_unate", "A", "fall"),
    ("XOR2x2_ASAP7_6t_L", "A", "positive_unate", "!B", "rise"),
]
CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_GROUPS = [
    _format_target_group_id(*spec) for spec in CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_SPECS
]

CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_NAME = "asap7_curated_dense_train_test_v1"
CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_RATIOS = (1.0 / 6.0, 0.0, 5.0 / 6.0)
CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_SPECS = [
    ("AND2x2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("AND3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "fall"),
    ("AND4x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
    ("BUFx4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "fall"),
    ("INVx4_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
    ("NAND2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
    ("NAND3x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("NOR2x2_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "rise"),
    ("NOR3x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "fall"),
    ("OR2x2_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("OR3x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
    ("OR4x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
    ("XNOR2x2_ASAP7_6t_L", "A", "positive_unate", "B", "rise"),
    ("XOR2x2_ASAP7_6t_L", "B", "negative_unate", "A", "fall"),
    ("XNOR2x2_ASAP7_6t_L", "B", "negative_unate", "!A", "fall"),
    ("XNOR2x2_ASAP7_6t_L", "A", "negative_unate", "!B", "rise"),
    ("XOR2x2_ASAP7_6t_L", "A", "positive_unate", "!B", "rise"),
    ("XOR2x2_ASAP7_6t_L", "B", "positive_unate", "!A", "rise"),
    ("AND3x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "rise"),
    ("OR3x2_ASAP7_6t_L", "C", "positive_unate", "<NONE>", "fall"),
    ("NAND3x2_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
    ("NOR3x2_ASAP7_6t_L", "C", "negative_unate", "<NONE>", "rise"),
    ("INVx4_ASAP7_6t_L", "A", "negative_unate", "<NONE>", "fall"),
    ("BUFx4_ASAP7_6t_L", "A", "positive_unate", "<NONE>", "rise"),
    ("AND2x2_ASAP7_6t_L", "B", "positive_unate", "<NONE>", "rise"),
    ("NAND2x2_ASAP7_6t_L", "B", "negative_unate", "<NONE>", "rise"),
]
CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_GROUPS = [
    _format_target_group_id(*spec) for spec in CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_SPECS
]

ZERO_SPI_FEATS = {
    "wp_sum": 0.0,
    "wn_sum": 0.0,
    "wp_over_wn": 0.0,
    "num_pmos": 0.0,
    "num_nmos": 0.0,
    "p_nfin_sum": 0.0,
    "n_nfin_sum": 0.0,
}


def summarize_data_sources(data_root: str) -> Dict[str, object]:
    root = Path(data_root)
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    ext_counts: Dict[str, int] = {}
    bytes_total = 0
    subckt_total = 0
    mos_total = 0
    nfin_files = 0
    for p in files:
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        bytes_total += int(p.stat().st_size)
        if ext in {".sp", ".spi", ".lib"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            subckt_total += len(re.findall(r"(?im)^\s*\.subckt\b", txt))
            mos_total += len(re.findall(r"(?im)^\s*M\S*\s+", txt))
            if re.search(r"(?im)\bnfin\s*=", txt):
                nfin_files += 1
    return {
        "files_total": len(files),
        "bytes_total": bytes_total,
        "extension_counts": ext_counts,
        "nfin_files": nfin_files,
        "subckt_total": subckt_total,
        "mos_line_total": mos_total,
    }


# ======================================================
# Slew/Delay thresholds normalization
# ======================================================

def parse_slew_thresholds(lib_text: str) -> Dict[str, Dict[str, float]]:
    def _find_pct(name: str):
        m = re.search(rf"{name}\s*:\s*([\d.]+)", lib_text, re.I)
        return float(m.group(1)) if m else None

    lower_rise = _find_pct("slew_lower_threshold_pct_rise")
    upper_rise = _find_pct("slew_upper_threshold_pct_rise")
    lower_fall = _find_pct("slew_lower_threshold_pct_fall")
    upper_fall = _find_pct("slew_upper_threshold_pct_fall")

    derate = _find_pct("slew_derate_from_library")
    if derate is None:
        derate = 1.0

    def _fill(lower, upper, fallback_lower, fallback_upper):
        if lower is None:
            lower = fallback_lower
        if upper is None:
            upper = fallback_upper
        if lower is None:
            lower = 10.0
        if upper is None:
            upper = 90.0
        return float(lower), float(upper)

    lower_rise, upper_rise = _fill(lower_rise, upper_rise, lower_fall, upper_fall)
    lower_fall, upper_fall = _fill(lower_fall, upper_fall, lower_rise, upper_rise)

    return {
        "rise": {"lower": lower_rise, "upper": upper_rise},
        "fall": {"lower": lower_fall, "upper": upper_fall},
        "derate": float(derate),
    }


def _thresholds_close(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], eps: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    keys = ["rise", "fall"]
    for k in keys:
        for t in ("lower", "upper"):
            if abs(a[k][t] - b[k][t]) > eps:
                return False
    if abs(a.get("derate", 1.0) - b.get("derate", 1.0)) > eps:
        return False
    return True


# ======================================================
# ¹¤¾ßº¯Êý£ºSPICE ÌØÕ÷
# ======================================================

def parse_spi_features_from_text(text: str) -> Dict[str, float]:
    devs = parse_transistors_spice(text)
    feats = extract_wl_features(devs)

    # »ñÈ¡Ô­Ê¼µ¥Î» (Ã×)
    raw_wp_sum = float(feats.get("wp_sum", 0.0))
    raw_wn_sum = float(feats.get("wn_sum", 0.0))

    # ¼ÆËã±ÈÂÊ
    if raw_wn_sum > 1e-15:
        wp_over_wn = raw_wp_sum / raw_wn_sum
    else:
        wp_over_wn = 0.0

    # ×ª»»ÎªÎ¢Ã×
    wp_sum_um = raw_wp_sum * 1e6
    wn_sum_um = raw_wn_sum * 1e6

    # È¡¶ÔÊý
    wp_sum_log = wp_sum_um
    wn_sum_log = wn_sum_um

    return {
        "wp_sum": wp_sum_log,
        "wn_sum": wn_sum_log,
        "wp_over_wn": wp_over_wn,
        "num_pmos": float(feats.get("num_pmos", 0.0)),
        "num_nmos": float(feats.get("num_nmos", 0.0)),
        "p_nfin_sum": float(feats.get("p_nfin_sum", 0.0)),
        "n_nfin_sum": float(feats.get("n_nfin_sum", 0.0)),
    }


def parse_spi_features(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return dict(ZERO_SPI_FEATS)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_spi_features_from_text(text)


# ======================================================
# É¨Ãè lib
# ======================================================

def collect_libs(root: str):
    root_path = Path(root)
    if not root_path.exists():
        return []
    found = []
    for x in root_path.rglob("*.lib"):
        if x.is_file():
            found.append(str(x))
    return sorted(set(found))


# ======================================================
# ASAP7 SP ´¦Àí
# ======================================================

def choose_asap7_sp(root_or_file: str) -> str:
    p = Path(root_or_file)
    if p.is_file():
        return str(p)

    root_path = p
    if not root_path.exists():
        return ""

    for name in ["asap7sc6t_26_L_211010.sp", "asap7sc6t_26_L.sp"]:
        cand = list(root_path.rglob(name))
        if cand:
            return str(sorted(cand)[0])

    cand = list(root_path.rglob("*.sp"))
    if cand:
        return str(sorted(cand)[0])

    cand = list(root_path.rglob("*.spi"))
    if cand:
        return str(sorted(cand)[0])

    return ""


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


# ======================================================
# ¹¹½¨ÌØÕ÷ÓëÐÐ
# ======================================================

def build_src_spi_feats(src_spi_root: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    root_path = Path(src_spi_root)
    if not root_path.exists():
        return {}, {}

    feats_map: Dict[str, Dict[str, float]] = {}
    mapping: Dict[str, str] = {}

    for cell_type in TARGET_CELL_TYPES:
        rel_name = SRC_CELL_SPI_FILES.get(cell_type)
        if rel_name is None:
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        cands = list(root_path.rglob(rel_name))
        if not cands:
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        path = str(sorted(cands)[0])
        mapping[cell_type] = path
        feats_map[cell_type] = parse_spi_features(path)

    return feats_map, mapping


def build_tgt_spi_feats_from_big_sp(tgt_sp_root_or_file: str) -> Tuple[
    Dict[str, Dict[str, float]], Dict[str, str], str]:
    sp_file = choose_asap7_sp(tgt_sp_root_or_file)
    if not sp_file or not os.path.exists(sp_file):
        print(f"[warn] TGT: asap7 SP file not found under {tgt_sp_root_or_file}")
        return {}, {}, ""

    with open(sp_file, "r", encoding="utf-8", errors="ignore") as f:
        sp_text = f.read()

    feats_map: Dict[str, Dict[str, float]] = {}
    subckt_map: Dict[str, str] = {}

    for cell_type in TARGET_CELL_TYPES:
        subckt = ASAP7_CELL_SUBCKT.get(cell_type)
        if subckt is None:
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        devs, _, _ = flatten_subckt_hierarchy(sp_text, subckt)
        if not devs:
            sub_text = extract_subckt_text(sp_text, subckt)
            if not sub_text.strip():
                feats_map[cell_type] = dict(ZERO_SPI_FEATS)
                continue
            feats_map[cell_type] = parse_spi_features_from_text(sub_text)
            subckt_map[cell_type] = subckt
            continue
        feats_map[cell_type] = parse_spi_features_from_text(
            "\n".join(
                f"M {d['d']} {d['g']} {d['s']} {d['b']} {d.get('model', d['type'])} "
                f"W={d.get('W', 0.0)} L={d.get('L', 0.0)} nfin={d.get('nfin', 0.0)} m={d.get('m', 1.0)}"
                for d in devs
            )
        )
        subckt_map[cell_type] = subckt

    return feats_map, subckt_map, sp_file


def _build_enhanced_row(
        tech: str,
        cell_type: str,
        cell_name: str,
        from_pin: str,
        to_pin: str,
        when_cond: str,
        sdf_cond: str,
        timing_sense: str,
        pol: str,
        slew: float,
        cap: float,
        voltage: float,
        temp: float,
        delay: float,
        spi_feats: Dict[str, float],
) -> Dict[str, float]:
    wp_sum_um = float(spi_feats.get("wp_sum", 0.0))
    wn_sum_um = float(spi_feats.get("wn_sum", 0.0))
    wp_over_wn = float(spi_feats.get("wp_over_wn", 0.0))

    slew_val = float(slew)
    cap_val = float(cap)

    eps = 1e-15
    req_p_linear = 1.0 / max(wp_sum_um, eps) if wp_sum_um > 0 else 0.0
    req_n_linear = 1.0 / max(wn_sum_um, eps) if wn_sum_um > 0 else 0.0

    rc_p_linear = req_p_linear * cap_val
    rc_n_linear = req_n_linear * cap_val

    if pol == "rise":
        rc_eff_linear = rc_p_linear
        req_eff_linear = req_p_linear
    else:
        rc_eff_linear = rc_n_linear
        req_eff_linear = req_n_linear

    if (wp_sum_um + wn_sum_um) > 0:
        pn_balance = (wp_sum_um - wn_sum_um) / (wp_sum_um + wn_sum_um)
    else:
        pn_balance = 0.0

    inv_v = 1.0 / max(voltage, 1e-12) if voltage > 0 else 0.0
    inv_temp = 1.0 / max(temp, 1e-12) if temp != 0 else 0.0

    log_eps = 1e-9
    arc_cond = normalize_arc_condition(when_cond, sdf_cond)
    group_id = "|".join([
        str(tech),
        str(cell_name),
        str(from_pin),
        str(to_pin),
        str(timing_sense),
        str(arc_cond),
        str(pol),
        f"v={float(voltage):.6g}",
        f"t={float(temp):.6g}",
    ])

    row = {
        "tech": tech,
        "cell_type": cell_type,
        "topology_group": cell_type_to_topology_group(cell_type) or "UNKNOWN",
        "cell_name": cell_name,
        "from_pin": from_pin,
        "to_pin": to_pin,
        "when_cond": when_cond,
        "sdf_cond": sdf_cond,
        "timing_sense": timing_sense,
        "arc_cond": arc_cond,
        "group_id": group_id,

        "pol": pol,
        "voltage": float(voltage),
        "temp": float(temp),
        "delay": float(delay),

        "wp_sum": float(np.log10(wp_sum_um + log_eps)),
        "wn_sum": float(np.log10(wn_sum_um + log_eps)),

        "slew": float(np.log10(slew_val + log_eps)),
        "cap": float(np.log10(cap_val + log_eps)),

        "log_slew": float(np.log10(slew_val + log_eps)),
        "log_cap": float(np.log10(cap_val + log_eps)),

        "req_p": float(np.log10(req_p_linear + log_eps)),
        "req_n": float(np.log10(req_n_linear + log_eps)),
        "rc_p": float(np.log10(rc_p_linear + log_eps)),
        "rc_n": float(np.log10(rc_n_linear + log_eps)),
        "req_eff": float(np.log10(req_eff_linear + log_eps)),
        "rc_eff": float(np.log10(rc_eff_linear + log_eps)),

        "wp_over_wn": wp_over_wn,
        "pn_balance": pn_balance,
        "is_inv": 1 if "INV" in cell_type else 0,
        "stack_pd": 1,
        "inv_v": inv_v,
        "inv_temp": inv_temp,
    }
    return row


def to_rows(tech: str, arc_dict: dict, spi_feats: Dict[str, float], slew_thresholds: Dict[str, Dict[str, float]]):
    rows = []
    v = float(arc_dict["nom_voltage"])
    t = float(arc_dict["nom_temperature"])
    grid_slew = arc_dict["slew"]
    grid_cap = arc_dict["cap"]

    cell_type = arc_dict["cell_type"]
    cell_name = arc_dict["cell_name"]
    from_pin = arc_dict["from_pin"]
    to_pin = arc_dict["to_pin"]
    when_cond = arc_dict.get("when_cond", "")
    sdf_cond = arc_dict.get("sdf_cond", "")
    timing_sense = arc_dict.get("timing_sense", "unknown")

    for pol, M in [("rise", arc_dict["cell_rise"]), ("fall", arc_dict["cell_fall"])]:
        th = slew_thresholds.get(pol, {}) if slew_thresholds else {}
        lower = float(th.get("lower", 10.0))
        upper = float(th.get("upper", 90.0))
        derate = float(slew_thresholds.get("derate", 1.0)) if slew_thresholds else 1.0
        span = max((upper - lower) / 100.0, 1e-6)
        for i, s in enumerate(grid_slew):
            for j, c in enumerate(grid_cap):
                delay = float(M[i, j])
                if not np.isfinite(delay):
                    continue
                if delay < -1e-6:
                    continue

                slew_norm = float(s) * derate / span

                row = _build_enhanced_row(
                    tech=tech,
                    cell_type=cell_type,
                    cell_name=cell_name,
                    from_pin=from_pin,
                    to_pin=to_pin,
                    when_cond=when_cond,
                    sdf_cond=sdf_cond,
                    timing_sense=timing_sense,
                    pol=pol,
                    slew=slew_norm,
                    cap=float(c),
                    voltage=v,
                    temp=t,
                    delay=delay,
                    spi_feats=spi_feats,
                )
                rows.append(row)
    return rows


# ======================================================
# ÇÐ·ÖÂß¼­
# ======================================================

def split_by_cell_type(df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42) -> Tuple[List[str], List[str], List[str]]:
    cell_types = df["cell_type"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(cell_types)

    n = len(cell_types)
    if n < 3:
        return cell_types.tolist(), [], []

    n_train = int(np.floor(ratios[0] * n))
    n_val = int(np.floor(ratios[1] * n))
    n_test = n - n_train - n_val

    if n_test < 1 and n > 2:
        n_test = 1
        n_train = n - n_val - n_test

    c_train = cell_types[:n_train]
    c_val = cell_types[n_train:n_train + n_val]
    c_test = cell_types[n_train + n_val:]

    return c_train.tolist(), c_val.tolist(), c_test.tolist()


def split_by_random(df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42) -> Tuple[List[int], List[int], List[int]]:
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    if n < 3:
        return idx.tolist(), [], []

    n_train = int(np.floor(ratios[0] * n))
    n_val = int(np.floor(ratios[1] * n))
    n_test = n - n_train - n_val

    if n_test < 1 and n > 2:
        n_test = 1
        n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def split_by_cell_type_stratified(
        df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    for ct in df["cell_type"].unique():
        idx = df.index[df["cell_type"] == ct].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            train_idx.extend(idx.tolist())
            continue
        n_train = int(np.floor(ratios[0] * n))
        n_val = int(np.floor(ratios[1] * n))
        n_test = n - n_train - n_val
        if n_test < 1 and n > 2:
            n_test = 1
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = max(0, n - n_train - n_test)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    return train_idx, val_idx, test_idx


def split_by_table_group(
        df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str]]:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe.")

    groups = df[group_col].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    n_train, n_val, _ = compute_split_sizes(len(groups), ratios)
    if len(groups) < 3:
        return groups.tolist(), [], []

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]
    train_groups = train_groups.tolist()
    val_groups = val_groups.tolist()
    test_groups = test_groups.tolist()
    _validate_group_partition(groups.tolist(), train_groups, val_groups, test_groups, split_mode="table_group")
    return train_groups, val_groups, test_groups


def split_by_table_group_train_test(
        df: pd.DataFrame, seed=42, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str]]:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe.")

    groups = df[group_col].dropna().astype(str).unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    if len(groups) < 2:
        return groups.tolist(), [], []

    n_train = int(np.floor((len(groups) / 6.0) + 0.5))
    n_train = max(1, min(n_train, len(groups) - 1))

    train_groups = groups[:n_train].tolist()
    val_groups: List[str] = []
    test_groups = groups[n_train:].tolist()
    _validate_group_partition(groups.tolist(), train_groups, val_groups, test_groups, split_mode="table_group_train_test")
    return train_groups, val_groups, test_groups


def split_by_table_group_manual(
        df: pd.DataFrame, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str]]:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe.")

    all_groups = df[group_col].dropna().astype(str).unique().tolist()
    all_group_set = set(all_groups)
    train_groups = list(CURATED_TABLE_GROUP_MANUAL_GROUPS["train"])
    val_groups = list(CURATED_TABLE_GROUP_MANUAL_GROUPS["val"])
    requested = train_groups + val_groups

    missing = sorted(set(requested) - all_group_set)
    if missing:
        raise ValueError(
            "table_group_manual requested groups that do not exist in current parsed target data. "
            f"missing={missing[:5]}"
        )

    selected = set(requested)
    test_groups = [g for g in all_groups if g not in selected]
    _validate_group_partition(all_groups, train_groups, val_groups, test_groups, split_mode="table_group_manual")
    return train_groups, val_groups, test_groups


def group_manual_train_auto_items(row) -> Set[str]:
    cell_type = str(row.get("cell_type", "UNKNOWN"))
    topo = str(row.get("topology_group", "UNKNOWN"))
    sense = str(row.get("timing_sense", "unknown"))
    pol = str(row.get("pol", "unknown"))
    arc_cond = str(row.get("arc_cond", "<NONE>"))
    return {
        f"cell:{cell_type}",
        f"topo:{topo}",
        f"sense:{sense}",
        f"pol:{pol}",
        f"arc:{arc_cond}",
    }


def greedy_select_groups_by_items(
        group_meta: pd.DataFrame,
        take_n: int,
        seed: int,
        item_builder: Callable,
        group_col: str = "group_id",
) -> Tuple[List[str], pd.DataFrame]:
    if take_n <= 0 or group_meta.empty:
        return [], group_meta.copy().reset_index(drop=True)

    rng = np.random.RandomState(seed)
    group_meta = group_meta.copy().reset_index(drop=True)
    group_meta["cover_items"] = group_meta.apply(item_builder, axis=1)

    uncovered: Set[str] = set()
    for items in group_meta["cover_items"].tolist():
        uncovered.update(items)

    remaining = group_meta.index.tolist()
    selected = []

    while remaining and len(selected) < take_n:
        best_gain = -1
        candidates = []

        for idx in remaining:
            gain = len(uncovered & group_meta.at[idx, "cover_items"])
            if gain > best_gain:
                best_gain = gain
                candidates = [idx]
            elif gain == best_gain:
                candidates.append(idx)

        if best_gain <= 0:
            rng.shuffle(remaining)
            selected.extend(remaining[: take_n - len(selected)])
            break

        max_rows = max(int(group_meta.at[idx, "row_count"]) for idx in candidates)
        candidates = [idx for idx in candidates if int(group_meta.at[idx, "row_count"]) == max_rows]
        chosen = candidates[rng.randint(len(candidates))]

        selected.append(chosen)
        remaining.remove(chosen)
        uncovered -= group_meta.at[chosen, "cover_items"]

    selected_set = set(selected)
    remaining_meta = group_meta.loc[[idx for idx in group_meta.index if idx not in selected_set]].copy()
    remaining_meta = remaining_meta.drop(columns=["cover_items"], errors="ignore").reset_index(drop=True)
    selected_groups = group_meta.loc[selected, group_col].astype(str).tolist()
    return selected_groups, remaining_meta


def split_by_table_group_manual_train(
        df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str], Dict[str, object]]:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe.")

    all_groups = df[group_col].dropna().astype(str).unique().tolist()
    all_group_set = set(all_groups)
    train_groups = list(CURATED_TABLE_GROUP_MANUAL_TRAIN_GROUPS)

    missing = sorted(set(train_groups) - all_group_set)
    if missing:
        raise ValueError(
            "table_group_manual_train requested train groups that do not exist in current parsed target data. "
            f"missing={missing[:5]}"
        )

    n_train, n_val, _ = compute_split_sizes(len(all_groups), ratios)
    if len(train_groups) != n_train:
        raise ValueError(
            "table_group_manual_train expects target train-group count to match the curated train spec. "
            f"expected={len(train_groups)}, ratio_train={n_train}, total_groups={len(all_groups)}, ratios={ratios}"
        )

    group_meta = build_group_meta(df, group_col=group_col)
    train_mask = group_meta[group_col].astype(str).isin(train_groups)
    train_cell_types = set(group_meta.loc[train_mask, "cell_type"].astype(str))
    remaining_meta = group_meta.loc[~train_mask].copy().reset_index(drop=True)

    seen_meta = remaining_meta[remaining_meta["cell_type"].astype(str).isin(train_cell_types)].copy()
    unseen_meta = remaining_meta[~remaining_meta["cell_type"].astype(str).isin(train_cell_types)].copy()

    n_val_seen = int(np.floor((2.0 * n_val) / 3.0))
    n_val_unseen = n_val - n_val_seen
    if len(seen_meta) < n_val_seen or len(unseen_meta) < n_val_unseen:
        raise ValueError(
            "table_group_manual_train could not satisfy the requested val seen/unseen split. "
            f"need_seen={n_val_seen}, have_seen={len(seen_meta)}, "
            f"need_unseen={n_val_unseen}, have_unseen={len(unseen_meta)}"
        )

    val_seen_groups, _ = greedy_select_groups_by_items(
        seen_meta,
        take_n=n_val_seen,
        seed=seed,
        item_builder=group_manual_train_auto_items,
        group_col=group_col,
    )
    val_unseen_groups, _ = greedy_select_groups_by_items(
        unseen_meta,
        take_n=n_val_unseen,
        seed=seed + 1,
        item_builder=group_manual_train_auto_items,
        group_col=group_col,
    )

    train_set = set(train_groups)
    val_seen_set = set(val_seen_groups)
    val_unseen_set = set(val_unseen_groups)
    val_set = val_seen_set | val_unseen_set

    val_groups = [g for g in all_groups if g in val_set]
    test_groups = [g for g in all_groups if g not in train_set and g not in val_set]
    _validate_group_partition(all_groups, train_groups, val_groups, test_groups, split_mode="table_group_manual_train")

    extra_info = {
        "manual_train_spec": CURATED_TABLE_GROUP_MANUAL_TRAIN_NAME,
        "val_seen_group_target": n_val_seen,
        "val_unseen_group_target": n_val_unseen,
        "val_seen_group_count": len(val_seen_groups),
        "val_unseen_group_count": len(val_unseen_groups),
        "val_seen_groups": list(val_seen_groups),
        "val_unseen_groups": list(val_unseen_groups),
    }
    return train_groups, val_groups, test_groups, extra_info


def split_by_table_group_manual_train_test(
        df: pd.DataFrame, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str], Dict[str, object]]:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe.")

    all_groups = df[group_col].dropna().astype(str).unique().tolist()
    all_group_set = set(all_groups)
    train_groups = list(CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_GROUPS)

    missing = sorted(set(train_groups) - all_group_set)
    if missing:
        raise ValueError(
            "table_group_manual_train_test requested train groups that do not exist in current parsed target data. "
            f"missing={missing[:5]}"
        )

    train_set = set(train_groups)
    val_groups: List[str] = []
    test_groups = [g for g in all_groups if g not in train_set]
    _validate_group_partition(
        all_groups,
        train_groups,
        val_groups,
        test_groups,
        split_mode="table_group_manual_train_test",
    )

    extra_info = {
        "manual_train_test_spec": CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_NAME,
        "manual_train_test_group_count": len(train_groups),
    }
    return train_groups, val_groups, test_groups, extra_info


def split_by_table_group_manual_dense_train_test(
        df: pd.DataFrame, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str], Dict[str, object]]:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe.")

    all_groups = df[group_col].dropna().astype(str).unique().tolist()
    all_group_set = set(all_groups)
    train_groups = list(CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_GROUPS)

    missing = sorted(set(train_groups) - all_group_set)
    if missing:
        raise ValueError(
            "table_group_manual_dense_train_test requested train groups that do not exist in current parsed target data. "
            f"missing={missing[:5]}"
        )

    train_set = set(train_groups)
    val_groups: List[str] = []
    test_groups = [g for g in all_groups if g not in train_set]
    _validate_group_partition(
        all_groups,
        train_groups,
        val_groups,
        test_groups,
        split_mode="table_group_manual_dense_train_test",
    )

    extra_info = {
        "manual_dense_train_test_spec": CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_NAME,
        "manual_dense_train_test_group_count": len(train_groups),
    }
    return train_groups, val_groups, test_groups, extra_info


def compute_split_sizes(n_total: int, ratios=(0.7, 0.2, 0.1)) -> Tuple[int, int, int]:
    if n_total < 3:
        return n_total, 0, 0

    n_train = int(np.floor(ratios[0] * n_total))
    n_val = int(np.floor(ratios[1] * n_total))
    n_test = n_total - n_train - n_val

    if n_test < 1 and n_total > 2:
        n_test = 1
        n_train = n_total - n_val - n_test

    return n_train, n_val, n_test


def _validate_group_partition(
        all_groups: List[str],
        train_groups: List[str],
        val_groups: List[str],
        test_groups: List[str],
        split_mode: str,
):
    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError(f"{split_mode} produced overlapping group assignments.")

    all_set = set(all_groups)
    union = train_set | val_set | test_set
    if union != all_set:
        missing = sorted(all_set - union)
        extra = sorted(union - all_set)
        raise ValueError(
            f"{split_mode} did not partition groups correctly. "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def build_group_meta(df: pd.DataFrame, group_col: str = "group_id") -> pd.DataFrame:
    required_cols = [
        group_col,
        "cell_type",
        "cell_name",
        "from_pin",
        "to_pin",
        "timing_sense",
        "arc_cond",
        "pol",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for group metadata: {missing}")

    work = df[df[group_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                group_col,
                "cell_type",
                "cell_name",
                "from_pin",
                "to_pin",
                "timing_sense",
                "arc_cond",
                "pol",
                "topology_group",
                "row_count",
            ]
        )

    if "topology_group" not in work.columns:
        work["topology_group"] = work["cell_type"].map(cell_type_to_topology_group)
    work["topology_group"] = work["topology_group"].fillna("UNKNOWN").astype(str)

    counts = work.groupby(group_col, dropna=False).size().rename("row_count").reset_index()
    group_meta = (
        work.groupby(group_col, dropna=False)
        .agg(
            {
                "cell_type": "first",
                "cell_name": "first",
                "from_pin": "first",
                "to_pin": "first",
                "timing_sense": "first",
                "arc_cond": "first",
                "pol": "first",
                "topology_group": "first",
            }
        )
        .reset_index()
    )
    group_meta = group_meta.merge(counts, on=group_col, how="left")
    group_meta["row_count"] = group_meta["row_count"].fillna(0).astype(int)
    return group_meta


def group_cover_items(row) -> Set[str]:
    topo = str(row.get("topology_group", "UNKNOWN"))
    sense = str(row.get("timing_sense", "unknown"))
    pol = str(row.get("pol", "unknown"))
    return {
        f"topo:{topo}",
        f"topo_sense:{topo}|{sense}",
        f"topo_pol:{topo}|{pol}",
        f"sense:{sense}",
        f"pol:{pol}",
    }


def greedy_select_cover_groups(
        group_meta: pd.DataFrame, take_n: int, seed: int, group_col: str = "group_id"
) -> Tuple[List[str], pd.DataFrame]:
    return greedy_select_groups_by_items(
        group_meta,
        take_n=take_n,
        seed=seed,
        item_builder=group_cover_items,
        group_col=group_col,
    )


def split_by_table_group_cover(
        df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42, group_col: str = "group_id"
) -> Tuple[List[str], List[str], List[str]]:
    group_meta = build_group_meta(df, group_col=group_col)
    n_train, n_val, _ = compute_split_sizes(len(group_meta), ratios)

    if len(group_meta) < 3:
        groups = group_meta[group_col].astype(str).tolist()
        return groups, [], []

    train_groups, remaining_meta = greedy_select_cover_groups(
        group_meta, take_n=n_train, seed=seed, group_col=group_col
    )
    val_groups, remaining_meta = greedy_select_cover_groups(
        remaining_meta, take_n=n_val, seed=seed + 1, group_col=group_col
    )
    test_groups = remaining_meta[group_col].astype(str).tolist()
    _validate_group_partition(
        group_meta[group_col].astype(str).tolist(),
        train_groups,
        val_groups,
        test_groups,
        split_mode="table_group_cover",
    )
    return train_groups, val_groups, test_groups


def _sorted_unique_values(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns or df.empty:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def _collect_cover_items_for_split(df: pd.DataFrame, group_col: str = "group_id") -> List[str]:
    if df.empty or group_col not in df.columns:
        return []
    group_meta = build_group_meta(df, group_col=group_col)
    items: Set[str] = set()
    for _, row in group_meta.iterrows():
        items.update(group_cover_items(row))
    return sorted(items)


def build_group_split_info(
        split_mode: str,
        ratios,
        seed: int,
        train_groups: List[str],
        val_groups: List[str],
        test_groups: List[str],
        df_tgt_train_pool: pd.DataFrame,
        df_tgt_val: pd.DataFrame,
        df_tgt_test: pd.DataFrame,
        group_col: str = "group_id",
        extra_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    info: Dict[str, object] = {
        "split_mode": split_mode,
        "ratios": list(ratios),
        "seed": seed,
        "num_train_groups": len(train_groups),
        "num_val_groups": len(val_groups),
        "num_test_groups": len(test_groups),
        "train_groups": list(train_groups),
        "val_groups": list(val_groups),
        "test_groups": list(test_groups),
    }

    for prefix, df_split in [
        ("train", df_tgt_train_pool),
        ("val", df_tgt_val),
        ("test", df_tgt_test),
    ]:
        info[f"{prefix}_cell_types"] = _sorted_unique_values(df_split, "cell_type")
        info[f"{prefix}_topologies"] = _sorted_unique_values(df_split, "topology_group")
        info[f"{prefix}_timing_sense"] = _sorted_unique_values(df_split, "timing_sense")
        info[f"{prefix}_pol"] = _sorted_unique_values(df_split, "pol")
        info[f"{prefix}_cover_items"] = _collect_cover_items_for_split(df_split, group_col=group_col)

    if extra_info:
        info.update(extra_info)

    return info


def print_group_split_summary(
        title: str,
        train_groups: List[str],
        val_groups: List[str],
        test_groups: List[str],
        df_tgt_train_pool: pd.DataFrame,
        df_tgt_val: pd.DataFrame,
        df_tgt_test: pd.DataFrame,
):
    train_cell_types = _sorted_unique_values(df_tgt_train_pool, "cell_type")
    val_cell_types = _sorted_unique_values(df_tgt_val, "cell_type")
    test_cell_types = _sorted_unique_values(df_tgt_test, "cell_type")
    train_topologies = _sorted_unique_values(df_tgt_train_pool, "topology_group")
    val_topologies = _sorted_unique_values(df_tgt_val, "topology_group")
    test_topologies = _sorted_unique_values(df_tgt_test, "topology_group")

    print("\n" + "=" * 50)
    print(title)
    print(
        f"  Train groups: {len(train_groups)}, rows: {len(df_tgt_train_pool)}, "
        f"cell types: {len(train_cell_types)}, topologies: {len(train_topologies)}"
    )
    print(
        f"  Val   groups: {len(val_groups)}, rows: {len(df_tgt_val)}, "
        f"cell types: {len(val_cell_types)}, topologies: {len(val_topologies)}"
    )
    print(
        f"  Test  groups: {len(test_groups)}, rows: {len(df_tgt_test)}, "
        f"cell types: {len(test_cell_types)}, topologies: {len(test_topologies)}"
    )
    print("=" * 50 + "\n")


# ======================================================
# ¡¾ºËÐÄÐÂÔö¡¿¼ÆËã Scalers
# ======================================================

def compute_and_save_scalers(df_train_pool: pd.DataFrame, out_dir: str):
    """
    »ùÓÚ Target Training Pool ¼ÆËãÌØÕ÷ºÍ±êÇ©µÄ¾ùÖµ/·½²î²¢±£´æ¡£
    ×¢Òâ£ºÐèÒª´¦Àí pol_bit¡£
    """
    print("[info] Computing scalers from Target Training Pool...")

    # 1. ÁÙÊ±´´½¨ pol_bit ÓÃÓÚÍ³¼Æ£¬²»ÐÞ¸ÄÔ­ df
    df_temp = df_train_pool.copy()
    if "pol_bit" not in df_temp.columns:
        if "pol" in df_temp.columns:
            df_temp["pol_bit"] = (df_temp["pol"] == "rise").astype(float)
        else:
            df_temp["pol_bit"] = 0.0

    # 2. ¼ÆËãÌØÕ÷Í³¼ÆÁ¿
    stats = {"mean": {}, "std": {}}
    for col in NUMERIC_COLS:
        if col in df_temp.columns:
            # ×ª»»Îª float ÒÔÈ·±£ JSON ¿ÉÐòÁÐ»¯
            stats["mean"][col] = float(df_temp[col].mean())
            stats["std"][col] = float(df_temp[col].std())
        else:
            print(f"[warn] Feature {col} not found in dataframe, using default 0/1.")
            stats["mean"][col] = 0.0
            stats["std"][col] = 1.0

    # 3. ¼ÆËã±êÇ©Í³¼ÆÁ¿ (Delay)
    if "delay" in df_temp.columns:
        y_mean = float(df_temp["delay"].mean())
        y_std = float(df_temp["delay"].std())
    else:
        y_mean = 0.0
        y_std = 1.0

    y_info = {"mean": y_mean, "std": y_std}

    # 4. ±£´æ
    with open(os.path.join(out_dir, "scaler_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(out_dir, "y_scaler.json"), "w") as f:
        json.dump(y_info, f, indent=2)

    print(f"[info] Scalers saved to {out_dir}")
    print(f"       Y Stats: mean={y_mean:.4f}, std={y_std:.4f}")
    return stats, y_info


# ======================================================
# Ö÷Á÷³Ì
# ======================================================

def main():
    args = get_options()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) ÕÒµ½ËùÓÐ lib ----------
    src_libs = collect_libs(args.src_lib)
    tgt_libs = collect_libs(args.tgt_lib)

    print(f"[info] Nangate45 libs found: {len(src_libs)}")
    print(f"[info] ASAP7 libs found    : {len(tgt_libs)}")

    if len(tgt_libs) == 0:
        raise SystemExit("[error] ASAP7 ÖÐÃ»ÓÐÕÒµ½ÈÎºÎ .lib£¬ÇëÈ·ÈÏÄ¿Â¼¡£")

    # ---------- 2) SPI ÌØÕ÷ ----------
    src_spi_feats_map, src_spi_map = build_src_spi_feats(args.src_spi)
    tgt_spi_feats_map, tgt_subckt_map, tgt_sp_file = build_tgt_spi_feats_from_big_sp(args.tgt_sp)

    # ---------- 3) ±éÀú lib ½âÎöÊý¾Ý ----------
    all_src_rows = []
    all_tgt_rows = []

    # Nangate45
    src_slew_thresholds = None
    for path in src_libs:
        print(f"[info] parse SRC lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        cur_th = parse_slew_thresholds(text)
        if src_slew_thresholds is None:
            src_slew_thresholds = cur_th
        elif not _thresholds_close(src_slew_thresholds, cur_th):
            print(f"[warn] SRC slew thresholds differ across libs: {path}")
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = src_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_src_rows += to_rows("Nangate45", arc, spi_feats, src_slew_thresholds)

    # ASAP7
    tgt_slew_thresholds = None
    for path in tgt_libs:
        print(f"[info] parse TGT lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        cur_th = parse_slew_thresholds(text)
        if tgt_slew_thresholds is None:
            tgt_slew_thresholds = cur_th
        elif not _thresholds_close(tgt_slew_thresholds, cur_th):
            print(f"[warn] TGT slew thresholds differ across libs: {path}")
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = tgt_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_tgt_rows += to_rows("ASAP7", arc, spi_feats, tgt_slew_thresholds)

    if len(all_tgt_rows) == 0:
        raise SystemExit("[error] ¹¹½¨Ê§°Ü£ºASAP7 Ä¿±êÓòÃ»ÓÐÈÎºÎÓÐÐ§Ñù±¾¡£")

    df_src = pd.DataFrame(all_src_rows) if len(all_src_rows) > 0 else pd.DataFrame()
    df_tgt = pd.DataFrame(all_tgt_rows)

    # È«¾Ö´òÂÒ
    df_tgt = df_tgt.sample(frac=1, random_state=args.split_seed).reset_index(drop=True)
    if not df_src.empty:
        df_src = df_src.sample(frac=1, random_state=args.split_seed).reset_index(drop=True)

    # ---------- 4) Êý¾Ý¼¯»®·Ö ----------
    split_mode = getattr(args, "tgt_split_mode", "cell_type")
    ratios = tuple(args.tgt_split_ratios)
    if split_mode == "cell_type":
        train_cells, val_cells, test_cells = split_by_cell_type(
            df_tgt,
            ratios=ratios,
            seed=args.split_seed,
        )

        print("\n" + "=" * 50)
        print("Target split (By Cell Type)")
        print(f"  Train Cells ({len(train_cells)}): {train_cells}")
        print(f"  Val   Cells ({len(val_cells)}): {val_cells}")
        print(f"  Test  Cells ({len(test_cells)}): {test_cells}")
        print("=" * 50 + "\n")

        df_tgt_train_pool = df_tgt[df_tgt["cell_type"].isin(train_cells)].copy()
        df_tgt_val = df_tgt[df_tgt["cell_type"].isin(val_cells)].copy()
        df_tgt_test = df_tgt[df_tgt["cell_type"].isin(test_cells)].copy()
        split_info = {
            "split_mode": "cell_type",
            "ratios": list(ratios),
            "seed": args.split_seed,
            "train_cells": train_cells,
            "val_cells": val_cells,
            "test_cells": test_cells,
        }
    elif split_mode == "random":
        train_idx, val_idx, test_idx = split_by_random(
            df_tgt, ratios=ratios, seed=args.split_seed
        )
        df_tgt_train_pool = df_tgt.iloc[train_idx].copy()
        df_tgt_val = df_tgt.iloc[val_idx].copy()
        df_tgt_test = df_tgt.iloc[test_idx].copy()
        print("\n" + "=" * 50)
        print("Target split (Random)")
        print(f"  Train rows: {len(df_tgt_train_pool)}")
        print(f"  Val   rows: {len(df_tgt_val)}")
        print(f"  Test  rows: {len(df_tgt_test)}")
        print("=" * 50 + "\n")
        split_info = {
            "split_mode": "random",
            "ratios": list(ratios),
            "seed": args.split_seed,
        }
    elif split_mode == "stratified":
        train_idx, val_idx, test_idx = split_by_cell_type_stratified(
            df_tgt, ratios=ratios, seed=args.split_seed
        )
        df_tgt_train_pool = df_tgt.iloc[train_idx].copy()
        df_tgt_val = df_tgt.iloc[val_idx].copy()
        df_tgt_test = df_tgt.iloc[test_idx].copy()
        print("\n" + "=" * 50)
        print("Target split (Stratified by Cell Type)")
        print(f"  Train rows: {len(df_tgt_train_pool)}")
        print(f"  Val   rows: {len(df_tgt_val)}")
        print(f"  Test  rows: {len(df_tgt_test)}")
        print("=" * 50 + "\n")
        split_info = {
            "split_mode": "stratified",
            "ratios": list(ratios),
            "seed": args.split_seed,
        }
    elif split_mode == "table_group":
        train_groups, val_groups, test_groups = split_by_table_group(
            df_tgt, ratios=ratios, seed=args.split_seed, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt[df_tgt["group_id"].isin(val_groups)].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            "Target split (Table-Level Group)",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group",
            ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
        )
    elif split_mode == "table_group_train_test":
        fixed_ratios = (1.0 / 6.0, 0.0, 5.0 / 6.0)
        train_groups, val_groups, test_groups = split_by_table_group_train_test(
            df_tgt, seed=args.split_seed, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt.iloc[0:0].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            "Target split (Table-Level Group Train/Test 1:5)",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group_train_test",
            fixed_ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
        )
    elif split_mode == "table_group_cover":
        train_groups, val_groups, test_groups = split_by_table_group_cover(
            df_tgt, ratios=ratios, seed=args.split_seed, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt[df_tgt["group_id"].isin(val_groups)].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            "Target split (Table-Level Group Cover)",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group_cover",
            ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
        )
    elif split_mode == "table_group_manual":
        train_groups, val_groups, test_groups = split_by_table_group_manual(
            df_tgt, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt[df_tgt["group_id"].isin(val_groups)].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            f"Target split (Manual Table-Level Group: {CURATED_TABLE_GROUP_MANUAL_NAME})",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group_manual",
            ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
            extra_info={"manual_spec": CURATED_TABLE_GROUP_MANUAL_NAME},
        )
    elif split_mode == "table_group_manual_train":
        train_groups, val_groups, test_groups, manual_extra_info = split_by_table_group_manual_train(
            df_tgt, ratios=ratios, seed=args.split_seed, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt[df_tgt["group_id"].isin(val_groups)].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            f"Target split (Manual Train + Auto Val/Test: {CURATED_TABLE_GROUP_MANUAL_TRAIN_NAME})",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group_manual_train",
            ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
            extra_info=manual_extra_info,
        )
    elif split_mode == "table_group_manual_train_test":
        fixed_ratios = CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_RATIOS
        train_groups, val_groups, test_groups, manual_extra_info = split_by_table_group_manual_train_test(
            df_tgt, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt.iloc[0:0].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            f"Target split (Manual Train/Test 1:5: {CURATED_TABLE_GROUP_MANUAL_TRAIN_TEST_NAME})",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group_manual_train_test",
            fixed_ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
            extra_info=manual_extra_info,
        )
    elif split_mode == "table_group_manual_dense_train_test":
        fixed_ratios = CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_RATIOS
        train_groups, val_groups, test_groups, manual_extra_info = split_by_table_group_manual_dense_train_test(
            df_tgt, group_col="group_id"
        )
        df_tgt_train_pool = df_tgt[df_tgt["group_id"].isin(train_groups)].copy()
        df_tgt_val = df_tgt.iloc[0:0].copy()
        df_tgt_test = df_tgt[df_tgt["group_id"].isin(test_groups)].copy()
        print_group_split_summary(
            f"Target split (Manual Dense Train/Test 1:5: {CURATED_TABLE_GROUP_MANUAL_DENSE_TRAIN_TEST_NAME})",
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
        )
        split_info = build_group_split_info(
            "table_group_manual_dense_train_test",
            fixed_ratios,
            args.split_seed,
            train_groups,
            val_groups,
            test_groups,
            df_tgt_train_pool,
            df_tgt_val,
            df_tgt_test,
            group_col="group_id",
            extra_info=manual_extra_info,
        )
    else:
        raise ValueError(f"Unknown tgt_split_mode: {split_mode}")

    split_info["num_tgt_train_pool"] = len(df_tgt_train_pool)
    split_info["num_tgt_val"] = len(df_tgt_val)
    split_info["num_tgt_test"] = len(df_tgt_test)

    # =========================================================
    # ¡¾ÐÂÔö²½Öè¡¿¼ÆËã²¢±£´æ Scalers (»ùÓÚ Target Train Pool)
    # =========================================================
    scaler_stats, y_scaler = compute_and_save_scalers(df_tgt_train_pool, args.out_dir)

    # ¼ÌÐø´¦Àí Labeled / Unlabeled
    df_tgt_train_pool = df_tgt_train_pool.sample(frac=1, random_state=args.split_seed).reset_index(drop=True)
    n_train_total = len(df_tgt_train_pool)
    n_train_labeled = int(n_train_total * args.target_label_ratio)

    df_tgt_train = df_tgt_train_pool.iloc[:n_train_labeled].copy()
    df_tgt_unlabeled = df_tgt_train_pool.iloc[n_train_labeled:].copy()

    df_tgt_train["is_labeled"] = 1
    df_tgt_unlabeled["is_labeled"] = 0
    df_tgt_val["is_labeled"] = 1
    df_tgt_test["is_labeled"] = 1

    # ===== Êä³ö CSV =====
    if not df_src.empty:
        df_src.to_csv(os.path.join(args.out_dir, "src_delay.csv"), index=False)

    df_tgt_train.to_csv(os.path.join(args.out_dir, "tgt_train.csv"), index=False)
    df_tgt_val.to_csv(os.path.join(args.out_dir, "tgt_val.csv"), index=False)
    df_tgt_test.to_csv(os.path.join(args.out_dir, "tgt_test.csv"), index=False)

    df_tgt_u_safe = df_tgt_unlabeled.drop(columns=["delay"], errors="ignore")
    df_tgt_u_safe.to_csv(os.path.join(args.out_dir, "tgt_unlabeled.csv"), index=False)
    df_tgt_unlabeled.to_csv(os.path.join(args.out_dir, "tgt_unlabeled_debug.csv"), index=False)

    df_tgt.to_csv(os.path.join(args.out_dir, "tgt_delay_full.csv"), index=False)

    # ÔªÊý¾Ý
    feature_cols = [
        c for c in df_tgt_train.columns
        if c not in ["delay", "tech", "is_labeled",
                     "cell_name", "from_pin", "to_pin", "when_cond", "sdf_cond",
                     "timing_sense", "arc_cond", "group_id", "topology_group"]
    ]
    
    meta = {
        "src_spi_by_cell": src_spi_map,
        "tgt_sp_file": tgt_sp_file,
        "tgt_subckt_by_cell": tgt_subckt_map,
        "slew_thresholds": {
            "src": src_slew_thresholds,
            "tgt": tgt_slew_thresholds,
        },
        "split_info": split_info,
        "stats": {
            "num_src": len(df_src),
            "num_tgt_train_labeled": len(df_tgt_train),
            "num_tgt_train_unlabeled": len(df_tgt_unlabeled),
            "num_tgt_val": len(df_tgt_val),
            "num_tgt_test": len(df_tgt_test),
        },
        "feature_cols": feature_cols,
        "cell_types": TARGET_CELL_TYPES,
        "data_source_summary": summarize_data_sources("data"),
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    dataset_obj = {
        "meta": meta,
        "scaler_stats": scaler_stats,
        "y_scaler": y_scaler,
        "tgt_train_df": df_tgt_train,
        "tgt_val_df": df_tgt_val,
        "tgt_test_df": df_tgt_test,
        "src_df": df_src,
    }
    with open(os.path.join(args.out_dir, args.dataset_pkl_name), "wb") as f:
        pickle.dump(dataset_obj, f)
    print("[info] DONE dataset build completed successfully.")
    print(f"[info] Check output in: {args.out_dir}")
    return

    print("[info] DONE ¡ª Êý¾Ý¼¯¹¹½¨³É¹¦£¨Strict Cell-Based Split + Scalers£©£¡")
    print(f"[info] Check output in: {args.out_dir}")


if __name__ == "__main__":
    main()
