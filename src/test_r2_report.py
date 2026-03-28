"""
Lightweight test-set evaluation helper for training scripts.

Usage in each script:
    from test_r2_report import run_train_and_report_test
    ...
    run_train_and_report_test(train_fn, options, seed, script_name=__file__)
"""

import inspect
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch as th
from torch.utils.data import DataLoader as TorchDataLoader


def _safe_len(x: Any) -> int:
    try:
        return int(len(x))
    except Exception:
        return 0


def _first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _get_test_df(scope: Dict[str, Any]):
    df_tgt_test = scope.get("df_tgt_test", None)
    if df_tgt_test is not None:
        return df_tgt_test
    obj = scope.get("obj", None)
    if isinstance(obj, dict):
        return obj.get("tgt_test_df", None)
    return None


def _get_design_dim(scope: Dict[str, Any]) -> Optional[int]:
    design_dim = scope.get("design_dim", None)
    if design_dim is not None:
        return int(design_dim)
    options = scope.get("options", None)
    if options is None:
        return None
    return int(getattr(options, "design_dim", getattr(options, "out_dim", 0)))


def _build_test_dataset(scope: Dict[str, Any], df_tgt_test):
    ds_cls = scope.get("CellDelayDataset", None)
    if ds_cls is None:
        raise RuntimeError("CellDelayDataset not found in training scope.")

    x_mean = scope.get("x_mean", None)
    x_std = scope.get("x_std", None)
    y_mean = scope.get("y_mean", None)
    y_std = scope.get("y_std", None)
    if any(v is None for v in (x_mean, x_std, y_mean, y_std)):
        raise RuntimeError("x/y normalization stats not found in training scope.")

    # Most scripts use optional extra args: pin2id / cond2id / role_maps.
    sig = inspect.signature(ds_cls.__init__)
    kw = {}
    for name in ("pin2id", "cond2id", "role_maps"):
        if name in sig.parameters and name in scope:
            kw[name] = scope[name]

    try:
        return ds_cls(df_tgt_test, x_mean, x_std, y_mean, y_std, **kw)
    except TypeError:
        # Fallback to positional extras when constructor is positional-only style.
        extras = []
        for name in ("pin2id", "cond2id", "role_maps"):
            if name in scope:
                extras.append(scope[name])
        return ds_cls(df_tgt_test, x_mean, x_std, y_mean, y_std, *extras)


def _build_test_loader(scope: Dict[str, Any], test_ds):
    DataLoader = scope.get("DataLoader", TorchDataLoader)
    options = scope.get("options", None)
    batch_size = int(getattr(options, "batch_size", 1024))

    val_dl = scope.get("val_dl", None)
    kwargs = {}
    if val_dl is not None:
        if hasattr(val_dl, "collate_fn"):
            kwargs["collate_fn"] = val_dl.collate_fn
        if hasattr(val_dl, "num_workers"):
            kwargs["num_workers"] = int(val_dl.num_workers)
        if hasattr(val_dl, "pin_memory"):
            kwargs["pin_memory"] = bool(val_dl.pin_memory)
        if kwargs.get("num_workers", 0) > 0:
            if hasattr(val_dl, "persistent_workers"):
                kwargs["persistent_workers"] = bool(val_dl.persistent_workers)
            if hasattr(val_dl, "prefetch_factor"):
                try:
                    kwargs["prefetch_factor"] = int(val_dl.prefetch_factor)
                except Exception:
                    pass

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)


def _try_load_best_ckpt(scope: Dict[str, Any]):
    best_ckpt_path = scope.get("best_ckpt_path", None)
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        return

    device = scope.get("device", th.device("cpu"))
    ckpt = th.load(best_ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        return

    enc = scope.get("enc", None)
    model = scope.get("model", None)

    if enc is not None and "enc" in ckpt:
        load_enc_fn = scope.get("_load_hgat_encoder_state_dict", None)
        loaded = False
        if callable(load_enc_fn):
            try:
                load_enc_fn(enc, ckpt["enc"])
                loaded = True
            except Exception:
                loaded = False
        if not loaded:
            try:
                enc.load_state_dict(ckpt["enc"], strict=False)
            except TypeError:
                enc.load_state_dict(ckpt["enc"])

    if model is not None and "model" in ckpt:
        try:
            model.load_state_dict(ckpt["model"], strict=False)
        except TypeError:
            model.load_state_dict(ckpt["model"])


def _extract_metrics(out: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            try:
                loss = float(out[0])
            except Exception:
                loss = None
            try:
                r2 = float(out[1])
            except Exception:
                r2 = None
            return loss, r2
        if len(out) == 1:
            try:
                return None, float(out[0])
            except Exception:
                return None, None
    if isinstance(out, dict):
        r2 = _first_non_none(out.get("test_r2", None), out.get("r2", None), out.get("val_r2", None))
        loss = _first_non_none(out.get("test_loss", None), out.get("loss", None), out.get("val_loss", None))
        try:
            r2 = float(r2) if r2 is not None else None
        except Exception:
            r2 = None
        try:
            loss = float(loss) if loss is not None else None
        except Exception:
            loss = None
        return loss, r2
    try:
        return None, float(out)
    except Exception:
        return None, None


def _call_validate(scope: Dict[str, Any], test_dl):
    validate_fn = _first_non_none(
        scope.get("validate_cell", None),
        scope.get("validate_cell_proc", None),
        scope.get("validate_deeper", None),
        scope.get("validate", None),
    )
    if not callable(validate_fn):
        raise RuntimeError("No validate function found in training scope.")

    enc = scope.get("enc", None)
    model = scope.get("model", None)
    device = scope.get("device", None)
    design_dim = _get_design_dim(scope)
    options = scope.get("options", None)

    dedup = bool(getattr(options, "dedup_z", False)) if options is not None else False
    use_amp = bool(scope.get("use_amp", False))
    z_dict_tgt = _first_non_none(scope.get("z_dict_tgt", None), scope.get("z_dict", None))
    graph_cache_tgt = _first_non_none(scope.get("graph_cache_tgt", None), scope.get("graph_cache", None))

    # Try common modern signatures first.
    attempts = [
        lambda: validate_fn(
            test_dl,
            enc,
            model,
            device,
            design_dim,
            z_dict=z_dict_tgt,
            graph_cache=graph_cache_tgt,
            dedup=dedup,
            use_amp=use_amp,
        ),
        lambda: validate_fn(
            test_dl,
            enc,
            model,
            device,
            design_dim,
            z_dict=z_dict_tgt,
            graph_cache=graph_cache_tgt,
            dedup=dedup,
        ),
        lambda: validate_fn(test_dl, enc, model, device, design_dim),
        # legacy train.py style fallback
        lambda: validate_fn(
            test_dl,
            device,
            model,
            scope.get("cnn", None),
            scope.get("beta", 0.0),
            options,
        ),
        lambda: validate_fn(test_dl, device, model, options),
        lambda: validate_fn(test_dl, model, device),
    ]

    last_err = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to call validate function.")


def maybe_report_test_r2(scope: Dict[str, Any], script_name: str = ""):
    try:
        df_tgt_test = _get_test_df(scope)
        if df_tgt_test is None or _safe_len(df_tgt_test) == 0:
            print("[Test] dataset.pkl has no tgt_test_df, skip test evaluation.")
            return

        test_ds = _build_test_dataset(scope, df_tgt_test)
        test_dl = _build_test_loader(scope, test_ds)
        _try_load_best_ckpt(scope)
        out = _call_validate(scope, test_dl)
        test_loss, test_r2 = _extract_metrics(out)

        if test_r2 is None:
            print("[Test] Evaluation ran but test_r2 was not parsed.")
            return

        if test_loss is None:
            print(f"[Test] test_r2:{test_r2:.4f}")
        else:
            print(f"[Test] test_loss:{test_loss:.6f}, test_r2:{test_r2:.4f}")
    except Exception as e:
        tag = f" ({script_name})" if script_name else ""
        print(f"[Warn] Test evaluation failed{tag}: {e}")


def run_train_and_report_test(train_fn, *args, script_name: str = "", **kwargs):
    captured = {}
    target_code = train_fn.__code__

    def tracer(frame, event, arg):
        nonlocal captured
        if event == "return" and frame.f_code is target_code:
            captured = dict(frame.f_locals)
        return tracer

    old_profiler = sys.getprofile()
    sys.setprofile(tracer)
    try:
        result = train_fn(*args, **kwargs)
    finally:
        sys.setprofile(old_profiler)

    if captured:
        maybe_report_test_r2(captured, script_name=script_name)
    else:
        tag = f" ({script_name})" if script_name else ""
        print(f"[Warn] No train scope captured{tag}; skip test evaluation.")
    return result

