"""
Train cell delay regressor (HGAT + MLP) using dataset.pkl.
"""
import os
import pickle
import random
import re
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

    print(f"[Info] Parsing Target SPICE: {tgt_spice}")
    sp_text = open(tgt_spice, "r", encoding="utf-8", errors="ignore").read()

    graph_cache = {}
    for ctype, sub_name in mapping.items():
        sub_txt = extract_subckt_text(sp_text, sub_name)
        if not sub_txt:
            continue
        devs = parse_transistors_spice(sub_txt)
        _, pins = parse_top_subckt_pins(sub_txt)
        if not devs:
            continue
        g, feats, _ = build_dgl_graph_from_devs(devs, pins)
        graph_cache[str(ctype)] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
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


def train(options, seed):
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

    if options.task != "reg":
        raise ValueError("Only regression task is supported in this training script.")

    data_dir = options.data_save_path
    print(data_dir)

    obj = load_dataset_pkl(data_dir, getattr(options, "dataset_pkl_name", "dataset.pkl"))
    meta = obj.get("meta", {})
    scaler_stats = obj.get("scaler_stats", {})
    y_scaler = obj.get("y_scaler", {})
    df_train = obj.get("tgt_train_df")
    df_val = obj.get("tgt_val_df")
    if df_train is None or df_val is None:
        raise RuntimeError("dataset.pkl must include tgt_train_df and tgt_val_df")

    x_mean = np.array([scaler_stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([scaler_stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean = float(y_scaler.get("mean", 0.0))
    y_std = float(y_scaler.get("std", 1.0))
    if abs(y_std) < 1e-12:
        y_std = 1.0

    train_ds = CellDelayDataset(df_train, x_mean, x_std, y_mean, y_std)
    val_ds = CellDelayDataset(df_val, x_mean, x_std, y_mean, y_std)

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    train_dl = DataLoader(train_ds, batch_size=options.batch_size, shuffle=True, collate_fn=my_collate)
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

    print("----------------Loading HGAT embeddings----------------")
    z_dict = None
    graph_cache = None
    if options.freeze_hgat:
        print("[Info] freeze_hgat=True, precomputing z")
        z_dict = precompute_z_from_tgt_spice(data_dir, meta, enc, device, design_dim)
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
        optimizer = th.optim.Adam(model.parameters(),
                                  lr=options.learning_rate, weight_decay=options.weight_decay)
    else:
        print("[Info] freeze_hgat=False, building graph cache")
        graph_cache = build_tgt_graph_cache(data_dir, meta, device)
        optimizer = th.optim.Adam(list(enc.parameters()) + list(model.parameters()),
                                  lr=options.learning_rate, weight_decay=options.weight_decay)
    loss_fn = nn.MSELoss()
    r2_score = R2Score().to(device)

    print("----------------Start training---------------")
    best_val = float("inf")
    for epoch in range(options.num_epoch):
        if options.freeze_hgat:
            enc.eval()
        else:
            enc.train()
        model.train()
        train_loss = 0.0
        train_n = 0

        for xb, yb, cts in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            zb = build_z_batch(
                cts, device, design_dim,
                z_dict=z_dict, graph_cache=graph_cache, enc=enc,
                dedup=options.dedup_z
            )

            optimizer.zero_grad()
            pred = model(xb, zb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_n += len(yb)

        train_loss = train_loss / max(1, train_n)

        # validation
        enc.eval()
        model.eval()
        val_loss = 0.0
        val_n = 0
        r2_score.reset()
        with th.no_grad():
            for xb, yb, cts in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                zb = build_z_batch(
                    cts, device, design_dim,
                    z_dict=z_dict, graph_cache=graph_cache, enc=enc,
                    dedup=options.dedup_z
                )
                pred = model(xb, zb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * len(yb)
                val_n += len(yb)
                r2_score.update(pred, yb)

        val_loss = val_loss / max(1, val_n)
        val_r2 = r2_score.compute().item()

        if (epoch + 1) % 5 == 0:
            print(f"[Ep {epoch+1:3d}] TrainLoss {train_loss:.6f} | ValLoss {val_loss:.6f} | ValR2 {val_r2:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            if os.path.exists(options.model_saving_dir) is False:
                os.makedirs(options.model_saving_dir)
            th.save({
                "enc": enc.state_dict(),
                "model": model.state_dict(),
                "design_dim": design_dim,
                "hgat_hid": hgat_hid,
                "hgat_heads": hgat_heads,
                "scaler_stats": scaler_stats,
                "y_scaler": y_scaler,
                "epoch": epoch + 1,
                "best_val_loss": best_val,
            }, os.path.join(options.model_saving_dir, "ckpt_best.pt"))

        # end = time()
        # runtime += end-start
        # Train_loss = total_loss / total_num
        # Train_r2 = total_r2 / total_num
        # # calculate accuracy, recall, precision and F1-score
        # Train_acc = correct / total_num
        # Train_recall = 0
        # Train_precision = 0
        # if tp != 0:
        #     Train_recall = tp / (tp + fn)
        #     Train_precision = tp / (tp + fp)
        # Train_F1_score = 0
        # if Train_precision != 0 or Train_recall != 0:
        #     Train_F1_score = 2 * Train_recall * Train_precision / (Train_recall + Train_precision)

        # print("Task: {}, epoch[{:d}]".format('classification' if options.task=='cls' else 'regression', epoch))
        # print("training runtime: ",runtime)
        # print("  train:")
        # # print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision,3))
        # print("loss:{:.8f}, r2:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(Train_loss,Train_r2,Train_acc,Train_recall,Train_F1_score))

        # validate
        # print("  validate:")
        # val_res,val_F1_score,val_r2 = validate(val_dataset, device, model,cnn,beta,options)
        # # print("  test:")
        # # validate(testdataloader, label_name, device, model,
        # #          Loss, beta, options)

        # # save the result of current epoch
        # with open(os.path.join(options.model_saving_dir, 'res.txt'), 'a') as f:
        #     f.write(str(round(Train_loss, 8)) + " " + str(round(Train_r2, 3)) + " " + str(round(Train_acc, 3)) + " " + str(
        #         round(Train_recall, 3)) + " " + str(round(Train_precision,3))+" " + str(round(Train_F1_score, 3)) + "\n")
        #     for res in val_res:
        #         f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(res[0], res[1],res[2], res[3], res[4], res[5]) + "\n")
        #     f.write('\n')

        # if options.task == 'cls':
        #     judgement = val_F1_score > max_F1_score 
        # elif options.task == 'reg':
        #     judgement = val_r2 > max_r2
        # else:
        #     assert False
        # #judgement = True
        # if judgement:
        #    stop_score = 0
        #    max_F1_score = val_F1_score
        #    max_r2 = val_r2
        #    print("Saving model.... ", os.path.join(options.model_saving_dir))
        #    if os.path.exists(options.model_saving_dir) is False:
        #       os.makedirs(options.model_saving_dir)
        #    with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
        #       parameters = options
        #       pickle.dump((parameters, model,cnn), f)
        #    print("Model successfully saved")
        # else:
        #     stop_score += 1
        #     if stop_score >= 5:
        #         print('Early Stop!')
        #         exit(0)


if __name__ == "__main__":
    options = get_options()
    seed = options.seed
    # seed = random.randint(1, 10000)
    # seed = 9294
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # if options.reproducibility:
    #     print(f'---------Reproducibility setting')
    #     th.backends.cudnn.benchmark = False
    #     th.backends.cudnn.deterministic = True
    #     os.environ["PYTHONHASHSEED"] = str(seed)
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        train(options, seed)
