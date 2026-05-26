# -*- coding: utf-8 -*-
"""Main training / evaluation script for Tirano"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import load_pickle, save_json, set_seed, setup_logger, NeighborFinder
from Tirano_multi import Tirano
from eval import eval_link_prediction


@dataclass
class Batch:
    src_idx: torch.Tensor
    rel_idx: torch.Tensor
    target_idx: torch.Tensor
    ts: torch.Tensor


class QuadDataset(Dataset):
    def __init__(self, data: List[Tuple[int, int, int, int, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def collate_quads(batch) -> Batch:
    # tuples: (s, r, o, t, event)
    s = torch.tensor([x[0] for x in batch], dtype=torch.long)
    r = torch.tensor([x[1] for x in batch], dtype=torch.long)
    o = torch.tensor([x[2] for x in batch], dtype=torch.long)
    t = torch.tensor([x[3] for x in batch], dtype=torch.long)
    return Batch(src_idx=s, rel_idx=r, target_idx=o, ts=t)


def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def _get_ckpt_paths(save_dir: str, run_key: str) -> Tuple[str, str]:
    """Return (best_ckpt_path, last_ckpt_path).

    We keep the filename convention:
      <save_dir>/<run_key>_best.pt
      <save_dir>/<run_key>_last.pt
    where run_key is typically: <run_name>_<dataset_lower>[_<tag>]
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, str(run_key))
    return base + "_best.pt", base + "_last.pt"


def _make_run_key(run_name: str, dataset_lower: str, tag: str) -> str:
    tag = (tag or "").strip()
    if tag:
        return f"{run_name}_{dataset_lower}_{tag}"
    return f"{run_name}_{dataset_lower}"


def _resolve_results_dir(save_dir: str, results_dir: str) -> str:
    """Resolve results directory.

    - If results_dir is empty: use save_dir.
    - If results_dir is absolute: use it.
    - Else: treat as relative to save_dir.
    """
    results_dir = (results_dir or "").strip()
    if not results_dir:
        return save_dir
    if os.path.isabs(results_dir):
        return results_dir
    return os.path.join(save_dir, results_dir)


def _append_csv_row(path: str, fieldnames: List[str], row: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    best_val_mrr: float,
    args: argparse.Namespace,
) -> None:
    """Save a full training checkpoint (model + optimizer + scaler + rng)."""

    # Move model weights to CPU for saving (prevents GPU serialization spikes)
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "format": "tirano_v2",
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_val_mrr": float(best_val_mrr),
        "args": vars(args),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    *,
    strict_model: bool = False,
    reset_optimizer: bool = False,
    reset_rng: bool = False,
) -> Dict:
    """Load a checkpoint. Supports both new (tirano_v2) and older formats."""
    # NOTE (PyTorch>=2.6): torch.load now defaults to weights_only=True which may
    # refuse to unpickle objects (e.g., numpy RNG states) stored in our checkpoints.
    # These checkpoints are produced by this codebase, so we explicitly set
    # weights_only=False for compatibility. If you are loading an UNTRUSTED
    # checkpoint, do NOT do this.
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions (<2.0 / <2.6) may not support weights_only.
        ckpt = torch.load(path, map_location=device)

    # --- model ---
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=strict_model)
    elif "state_dict" in ckpt:
        # older code format
        model.load_state_dict(ckpt["state_dict"], strict=strict_model)
    else:
        raise KeyError(f"Unknown checkpoint format (no model_state/state_dict): {path}")

    # --- optimizer / scaler (optional) ---
    if (not reset_optimizer) and (optimizer is not None):
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        elif "optimizer" in ckpt and ckpt["optimizer"] is not None:
            # older code format
            optimizer.load_state_dict(ckpt["optimizer"])
        # else: no optimizer in ckpt (fine)

    if (not reset_optimizer) and (scaler is not None):
        if "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                # older torch versions / changed scaler type
                pass

    # --- rng ---
    if (not reset_rng) and ("rng_state" in ckpt):
        rs = ckpt["rng_state"]
        try:
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and (rs.get("cuda") is not None):
                torch.cuda.set_rng_state_all(rs["cuda"])
        except Exception:
            # RNG restoration is best-effort
            pass

    return ckpt


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="make training more reproducible (may be slower)")

    # training
    p.add_argument("--epochs", type=int, default=30, help="total epochs to train (not 'additional')")
    p.add_argument("--batch_size", type=int, default=32)  # <<< safer default
    p.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    # model
    p.add_argument("--embed_dim", type=int, default=200)
    p.add_argument("--score_func", type=str, default="distmult", choices=["distmult", "complex", "bique"])

    # entity embedding split (for the final query representation)
    # - shared: use the (trainable) entity embedding table everywhere (original behavior)
    # - pretrained_frozen: keep a separate frozen table for the 'static entity embedding' term
    #   in the final query representation; initialize it from --pretrained_ckpt.
    p.add_argument(
        "--static_entity_mode",
        type=str,
        default="pretrained_frozen",
        choices=["shared", "pretrained_frozen"],
        help="how to form the 'static entity embedding' term in the final query representation",
    )

    p.add_argument("--num_neighbors", type=int, default=50, help="K in RTNS")
    p.add_argument("--max_entities", type=int, default=60, help="slot budget per query (>=K recommended)")
    p.add_argument("--max_relations", type=int, default=60, help="slot budget per query")

    p.add_argument("--window_size", type=int, default=50, help="m=n time window")
    p.add_argument("--window_future", type=int, default=None, help="future window size n (default: same as --window_size). For extrapolation/forecasting, set this to 0 to use only past context.")

    p.add_argument("--bin_width", type=int, default=1, help="Δ")
    p.add_argument("--beta", type=float, default=1.0, help="beta in temporal decay")

    # ST-CNN (sparse)
    p.add_argument("--hr_c", type=int, default=128, help="bottleneck feature dim for snapshot channels")
    p.add_argument("--c_out", type=int, default=128)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dilations", type=str, default="1,2,4")
    p.add_argument("--num_cnn_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    # sampling
    p.add_argument("--sampling", type=str, default="rtns", choices=["rtns", "uniform", "recent"])
    p.add_argument("--deterministic_rtns", action="store_true", help="use top-k instead of stochastic RTNS")
    p.add_argument("--context_mode", type=str, default="auto", choices=["auto","all","past","past_inclusive"], help="Neighbor time filtering. auto: use past for chronological(extrapolation) splits, else all. past: ts < t_q (forecasting-safe). past_inclusive: ts <= t_q.")


    # pretrained init
    p.add_argument("--pretrained_ckpt", type=str, default="", help="path to pretrained_init*.pt")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default="tirano")
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="optional tag appended to run_name (useful to avoid overwriting results when doing many runs)",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="where to save logs/metrics (relative to --save_dir unless absolute; use '' to write into save_dir)",
    )
    p.add_argument(
        "--save_val_json",
        action="store_true",
        help="if set, also save per-epoch validation metrics as JSON files (in addition to history CSV)",
    )

    # resume
    p.add_argument(
        "--resume_from",
        type=str,
        default="none",
        choices=["none", "best", "last", "path"],
        help="resume training from checkpoint: none/best/last/path",
    )
    p.add_argument("--resume_ckpt", type=str, default="", help="explicit ckpt path when --resume_from path")
    p.add_argument("--resume_strict_model", action="store_true", help="strict state_dict loading (debug)")
    p.add_argument("--reset_optimizer", action="store_true", help="when resuming, DO NOT load optimizer/scaler states")
    p.add_argument("--reset_rng", action="store_true", help="when resuming, DO NOT restore RNG states")
    p.add_argument("--reset_epoch_counter", action="store_true", help="when resuming, start epoch counter from 1")
    p.add_argument("--keep_lr_on_resume", action="store_true", help="keep checkpoint LR instead of CLI --lr")

    # device
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp", action="store_true", help="enable AMP mixed precision")

    # eval-only
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint to load for test_only")

    args = p.parse_args()

    # Reproducibility
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # speed-ups (safe)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dname = args.dataset.lower()
    proc_dir = os.path.join(args.data_dir, dname, "processed")
    if not os.path.exists(proc_dir):
        raise FileNotFoundError(f"{proc_dir} not found. Run preprocess.py first.")

    # -------------------------
    # Results / logging outputs
    # -------------------------
    run_key = _make_run_key(args.run_name, dname, args.tag)
    results_dir = _resolve_results_dir(args.save_dir, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    run_base = os.path.join(results_dir, run_key)
    log_file = run_base + ".log"
    args_path = run_base + "_args.json"
    history_path = run_base + "_history.csv"
    best_json_path = run_base + "_best.json"
    test_json_path = run_base + "_test.json"
    summary_path = run_base + "_summary.json"

    logger = setup_logger("tirano", log_file=log_file)
    logger.info(f"Dataset: {args.dataset}  device={device}")
    logger.info(f"run_key={run_key}")
    logger.info(f"results_dir={results_dir}")
    logger.info(str(vars(args)))

    # save args + environment info once
    env_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cmd": " ".join(sys.argv),
        "python": sys.version,
        "torch": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "device": str(device),
    }
    save_json({"args": vars(args), "env": env_info}, args_path)

    if args.static_entity_mode == "pretrained_frozen" and (not args.pretrained_ckpt):
        logger.warning(
            "static_entity_mode=pretrained_frozen but --pretrained_ckpt is empty. "
            "The frozen static entity embedding will stay random-initialized unless it is present in the loaded checkpoint. "
            "For best results, pass --pretrained_ckpt <pretrained_init_dim*.pt>."
        )

    train_data = load_pickle(os.path.join(proc_dir, "train_data.pkl"))
    valid_data = load_pickle(os.path.join(proc_dir, "valid_data.pkl"))
    test_data = load_pickle(os.path.join(proc_dir, "test_data.pkl"))

    has_valid = len(valid_data) > 0
    logger.info(f"train={len(train_data)} valid={len(valid_data)} test={len(test_data)}")

    # ---------------------------
    # Auto-detect extrapolation setting (chronological splits)
    # ---------------------------
    def _time_range(data_split):
        if not data_split:
            return None
        ts = [int(x[3]) for x in data_split]
        return (min(ts), max(ts))

    tr_rng = _time_range(train_data)
    va_rng = _time_range(valid_data)
    te_rng = _time_range(test_data)

    is_extrapolation = False
    try:
        if tr_rng and te_rng:
            if va_rng:
                is_extrapolation = (tr_rng[1] < va_rng[0]) and (va_rng[1] < te_rng[0])
            else:
                is_extrapolation = (tr_rng[1] < te_rng[0])
    except Exception:
        is_extrapolation = False

    # Resolve neighbor time filtering mode
    if args.context_mode == 'auto':
        context_mode = 'past' if is_extrapolation else 'all'
    else:
        context_mode = args.context_mode

    # Resolve future window size n
    if args.window_future is None:
        window_future = 0 if context_mode in ('past', 'past_inclusive') else args.window_size
    else:
        window_future = int(args.window_future)

    logger.info(f"Split time ranges: train={tr_rng} valid={va_rng} test={te_rng}  -> is_extrapolation={is_extrapolation}")
    logger.info(f"Effective context_mode={context_mode}  window_past(m)={args.window_size}  window_future(n)={window_future}")


    sr2o = load_pickle(os.path.join(proc_dir, "sr2o.pkl"))
    srt2o = load_pickle(os.path.join(proc_dir, "srt2o.pkl"))

    adj_train = load_pickle(os.path.join(proc_dir, "o2srt_train.pkl"))
    adj_train_val = load_pickle(os.path.join(proc_dir, "o2srt_train_val.pkl"))

    relation2alpha = load_pickle(os.path.join(proc_dir, "relation2alpha.pkl"))

    # ---------------------------
    # Dataset stats (E / R counts)
    # ---------------------------
    meta = None
    meta_path = os.path.join(proc_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None

    max_ent = 0
    max_rel = 0
    for data_split in (train_data, valid_data, test_data):
        for (s, r, o, _, _) in data_split:
            if s > max_ent:
                max_ent = s
            if o > max_ent:
                max_ent = o
            if r > max_rel:
                max_rel = r

    num_entities = int(max_ent) + 1
    num_relations = int(max_rel) + 1

    if isinstance(meta, dict):
        try:
            num_entities = max(num_entities, int(meta.get("num_entities", 0)))
            num_relations = max(num_relations, int(meta.get("num_relations_aug", 0)))
        except Exception:
            pass

    # Ensure relation2alpha is sized to num_relations (NeighborFinder and Tirano index by relation id)
    relation2alpha = np.asarray(relation2alpha, dtype=np.float32).reshape(-1)
    if relation2alpha.shape[0] != num_relations:
        if relation2alpha.shape[0] > num_relations:
            logger.warning(
                f"relation2alpha has more entries than num_relations: {relation2alpha.shape[0]} > {num_relations}. "
                "Truncating to match."
            )
            relation2alpha = relation2alpha[:num_relations]
        else:
            pad_n = int(num_relations - relation2alpha.shape[0])
            pad_val = float(relation2alpha.mean()) if relation2alpha.size > 0 else 0.1
            logger.warning(
                f"relation2alpha has fewer entries than num_relations: {relation2alpha.shape[0]} < {num_relations}. "
                f"Padding {pad_n} values with mean={pad_val:.6f}."
            )
            relation2alpha = np.concatenate(
                [relation2alpha, np.full((pad_n,), pad_val, dtype=np.float32)], axis=0
            )

    logger.info(f"num_entities={num_entities}, num_relations={num_relations}")

    nf = NeighborFinder(
        adj=adj_train,
        sampling=args.sampling,
        relation2alpha=relation2alpha,
        beta=args.beta,
        deterministic=args.deterministic_rtns,
        seed=args.seed,
        time_mode=context_mode,
    )

    dilations = [int(x.strip()) for x in args.dilations.split(",") if x.strip()]

    model = Tirano(
        num_entities=num_entities,
        num_relations=num_relations,
        nf=nf,
        embed_dim=args.embed_dim,
        score_func=args.score_func,
        static_entity_mode=args.static_entity_mode,
        max_entities=args.max_entities,
        max_relations=args.max_relations,
        window_size=args.window_size,
        window_future=window_future,
        bin_width=args.bin_width,
        beta=args.beta,
        hr_c=args.hr_c,
        c_out=args.c_out,
        kernel_size=args.kernel_size,
        dilations=dilations,
        num_cnn_layers=args.num_cnn_layers,
        dropout=args.dropout,
        #relation2alpha=relation2alpha,
        relation2alpha=None,
    ).to(device)

    if args.pretrained_ckpt and (args.resume_from == "none"):
        # Only load pretrained init for "fresh" training runs.
        # If resuming, we always load from checkpoint to avoid overwriting learned weights.
        logger.info(f"Loading pretrained: {args.pretrained_ckpt}")
        model.load_pretrained_embeddings(args.pretrained_ckpt, strict=False)

    best_path, last_path = _get_ckpt_paths(args.save_dir, run_key)

    # dataloaders
    train_dl = DataLoader(
        QuadDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_quads,
        pin_memory=(device.type == "cuda"),
    )
    valid_dl = DataLoader(
        QuadDataset(valid_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_quads,
        pin_memory=(device.type == "cuda"),
    )
    test_dl = DataLoader(
        QuadDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_quads,
        pin_memory=(device.type == "cuda"),
    )

    # test-only mode
    if args.test_only:
        ckpt_path = args.ckpt if args.ckpt else best_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        # Reuse the robust loader so PyTorch>=2.6 doesn't fail on weights_only.
        load_checkpoint(
            ckpt_path,
            model=model,
            optimizer=None,
            scaler=None,
            device=device,
            strict_model=False,
            reset_optimizer=True,
            reset_rng=True,
        )

        # If you are using the split static embedding mode, you may want the frozen
        # static table to always come from the pretrained KG init, even when evaluating
        # an older checkpoint that doesn't include it.
        if args.pretrained_ckpt and args.static_entity_mode == "pretrained_frozen":
            model.load_pretrained_embeddings(args.pretrained_ckpt, load_dynamic=False, load_static=True)

        model.eval()
        nf.set_adj(adj_train_val)
        res = eval_link_prediction(model, test_dl, sr2o, srt2o, num_entities, device=device, verbose=True, num_neighbors=args.num_neighbors)
        logger.info("[TestOnly] " + "  ".join([f"{k}={v:.4f}" for k, v in res.items()]))

        # Save results
        save_json(
            {
                "mode": "test_only",
                "dataset": args.dataset,
                "run_key": run_key,
                "ckpt": ckpt_path,
                "results": res,
            },
            test_json_path,
        )
        save_json(
            {
                "dataset": args.dataset,
                "run_key": run_key,
                "ckpt_used": ckpt_path,
                "best_val": None,
                "test": res,
                "note": "test_only run",
            },
            summary_path,
        )
        logger.info(f"Saved test results: {test_json_path}")
        return

    # optimizer / AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = bool(args.use_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume if requested ----
    start_epoch = 1
    best_val_mrr = -1.0
    best_epoch = -1

    if args.resume_from != "none":
        if args.resume_from == "path":
            if not args.resume_ckpt:
                raise ValueError("--resume_from path requires --resume_ckpt <path>")
            resume_path = args.resume_ckpt
        elif args.resume_from == "best":
            resume_path = best_path
        elif args.resume_from == "last":
            resume_path = last_path
        else:
            raise ValueError(f"Unknown resume_from={args.resume_from}")

        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}\n"
                f"Hint: check --run_name/--save_dir, or pass --resume_from path --resume_ckpt <file>."
            )

        logger.info(f"[Resume] Loading checkpoint: {resume_path}")
        ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=opt,
            scaler=scaler,
            device=device,
            strict_model=bool(args.resume_strict_model),
            reset_optimizer=bool(args.reset_optimizer),
            reset_rng=bool(args.reset_rng),
        )

        # For split static embedding mode: keep the frozen static table pinned to
        # the pretrained KG init (useful when resuming from older checkpoints).
        if args.pretrained_ckpt and args.static_entity_mode == "pretrained_frozen":
            model.load_pretrained_embeddings(args.pretrained_ckpt, load_dynamic=False, load_static=True)

        # infer epoch / best metric from various formats
        ckpt_epoch = int(ckpt.get("epoch", ckpt.get("best_epoch", 0)))
        if "best_val_mrr" in ckpt:
            best_val_mrr = float(ckpt["best_val_mrr"])
        elif "best_val" in ckpt and isinstance(ckpt["best_val"], dict) and ("mrr" in ckpt["best_val"]):
            best_val_mrr = float(ckpt["best_val"]["mrr"])
        else:
            best_val_mrr = -1.0

        if args.reset_epoch_counter:
            start_epoch = 1
        else:
            start_epoch = ckpt_epoch + 1

        # If a separate best checkpoint exists, sync best_val_mrr from it
        # (useful when resuming from last).
        if has_valid and os.path.exists(best_path):
            try:
                try:
                    b = torch.load(best_path, map_location="cpu", weights_only=False)
                except TypeError:
                    b = torch.load(best_path, map_location="cpu")
                if isinstance(b, dict):
                    if "best_val_mrr" in b:
                        best_val_mrr = max(best_val_mrr, float(b["best_val_mrr"]))
                    if "epoch" in b:
                        best_epoch = int(b["epoch"])
            except Exception:
                pass

        # If we resumed from the best checkpoint itself, the checkpoint epoch is the best epoch.
        if args.resume_from == "best" and ckpt_epoch > 0:
            best_epoch = int(ckpt_epoch)

        if not args.keep_lr_on_resume:
            _set_optimizer_lr(opt, args.lr)

        logger.info(f"[Resume] start_epoch={start_epoch}  best_val_mrr={best_val_mrr:.4f}")

        # If resuming, we intentionally DO NOT load pretrained_ckpt (it would overwrite).
        # If user wants to start from pretrained but "continue", they should train from scratch.

    # ---- training loop ----
    if start_epoch > args.epochs:
        logger.info(
            f"Nothing to do: start_epoch({start_epoch}) > epochs({args.epochs}). "
            f"Change --epochs to a larger number or resume_from none."
        )
        return

    # History CSV (per-epoch metrics)
    history_fields = [
        "epoch",
        "train_loss",
        "val_mrr",
        "val_hits1",
        "val_hits10",
        "best_val_mrr",
        "lr",
        "epoch_time_sec",
        "timestamp",
    ]

    # Fresh run: overwrite stale result files (avoid mixing old runs)
    if args.resume_from == "none":
        for pth in [history_path, best_json_path, test_json_path, summary_path]:
            if os.path.exists(pth):
                try:
                    os.remove(pth)
                except Exception:
                    pass
        logger.info(
            "Fresh run detected (resume_from=none): existing history/best/test/summary files will be overwritten. "
            "Tip: use --tag to keep multiple runs side-by-side."
        )

    if has_valid and os.path.exists(best_path):
        # If a best checkpoint exists already (common in resume_from last), record its epoch for summaries.
        try:
            try:
                b = torch.load(best_path, map_location="cpu", weights_only=False)
            except TypeError:
                b = torch.load(best_path, map_location="cpu")
            if isinstance(b, dict) and ("epoch" in b):
                best_epoch = int(b["epoch"])
        except Exception:
            pass

    for epoch in range(start_epoch, args.epochs + 1):
        t0_epoch = time.time()
        model.train()
        nf.set_adj(adj_train)

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", leave=False)
        for it, batch in enumerate(pbar, start=1):
            batch = Batch(
                src_idx=batch.src_idx.to(device, non_blocking=True),
                rel_idx=batch.rel_idx.to(device, non_blocking=True),
                target_idx=batch.target_idx.to(device, non_blocking=True),
                ts=batch.ts.to(device, non_blocking=True),
            )

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                scores = model(batch, num_neighbors=args.num_neighbors)
                loss = model.loss(scores, batch.target_idx)
                loss = loss / float(max(1, args.accum_steps))

            scaler.scale(loss).backward()

            if (it % args.accum_steps) == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            n_steps += 1
            pbar.set_postfix(loss=total_loss / max(1, n_steps))

        train_loss_epoch = float(total_loss / max(1, n_steps))
        logger.info(f"Epoch {epoch}: train_loss={train_loss_epoch:.4f}")

        val_res = None

        # validation
        if has_valid and (epoch % args.eval_every == 0):
            model.eval()
            nf.set_adj(adj_train)  # context from train only (avoid leakage into validation)

            val_res = eval_link_prediction(
                model=model,
                data_loader=valid_dl,
                sr2o=sr2o,
                srt2o=srt2o,
                num_entities=num_entities,
                device=device,
                verbose=False,
                num_neighbors=args.num_neighbors,
            )
            logger.info(
                f"[Valid][Epoch {epoch}] MRR={val_res['mrr']:.4f}  Hits@1={val_res['hits1']:.4f}  Hits@10={val_res['hits10']:.4f}"
            )

            if args.save_val_json:
                save_json(val_res, run_base + f"_val_epoch{epoch}.json")

            if val_res["mrr"] > best_val_mrr:
                best_val_mrr = float(val_res["mrr"])
                best_epoch = int(epoch)
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=opt if not args.reset_optimizer else None,
                    scaler=scaler if (use_amp and (not args.reset_optimizer)) else None,
                    epoch=epoch,
                    best_val_mrr=best_val_mrr,
                    args=args,
                )
                logger.info(f"Saved best checkpoint: {best_path} (MRR={best_val_mrr:.4f})")

                # persist best metrics
                try:
                    save_json(
                        {
                            "dataset": args.dataset,
                            "run_key": run_key,
                            "best_epoch": best_epoch,
                            "best_val_mrr": best_val_mrr,
                            "val": val_res,
                            "best_ckpt": best_path,
                        },
                        best_json_path,
                    )
                except Exception:
                    pass

        # always save "last" checkpoint after each epoch (so epoch-20 resume works)
        save_checkpoint(
            last_path,
            model=model,
            optimizer=opt if not args.reset_optimizer else None,
            scaler=scaler if (use_amp and (not args.reset_optimizer)) else None,
            epoch=epoch,
            best_val_mrr=best_val_mrr,
            args=args,
        )

        # append history row
        epoch_time = float(time.time() - t0_epoch)
        cur_lr = float(opt.param_groups[0].get("lr", args.lr))
        row = {
            "epoch": int(epoch),
            "train_loss": train_loss_epoch,
            "val_mrr": float(val_res["mrr"]) if isinstance(val_res, dict) and ("mrr" in val_res) else "",
            "val_hits1": float(val_res["hits1"]) if isinstance(val_res, dict) and ("hits1" in val_res) else "",
            "val_hits10": float(val_res["hits10"]) if isinstance(val_res, dict) and ("hits10" in val_res) else "",
            "best_val_mrr": float(best_val_mrr),
            "lr": cur_lr,
            "epoch_time_sec": epoch_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_csv_row(history_path, history_fields, row)

    # ---- final test ----
    if has_valid and os.path.exists(best_path):
        # use best for testing
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            raise KeyError(f"Unknown checkpoint format: {best_path}")
        model.eval()

    nf.set_adj(adj_train_val)
    model.eval()

    test_res = eval_link_prediction(
        model=model,
        data_loader=test_dl,
        sr2o=sr2o,
        srt2o=srt2o,
        num_entities=num_entities,
        device=device,
        verbose=True,
        num_neighbors=args.num_neighbors,
    )
    logger.info("[Test] " + "  ".join([f"{k}={v:.4f}" for k, v in test_res.items()]))

    # Save test results + summary
    ckpt_used = best_path if (has_valid and os.path.exists(best_path)) else last_path
    try:
        save_json(
            {
                "dataset": args.dataset,
                "run_key": run_key,
                "ckpt_used": ckpt_used,
                "results": test_res,
            },
            test_json_path,
        )
    except Exception:
        pass

    try:
        save_json(
            {
                "dataset": args.dataset,
                "run_key": run_key,
                "best_epoch": int(best_epoch) if best_epoch >= 0 else None,
                "best_val_mrr": float(best_val_mrr) if best_val_mrr >= 0 else None,
                "best_ckpt": best_path if os.path.exists(best_path) else None,
                "last_ckpt": last_path if os.path.exists(last_path) else None,
                "ckpt_used_for_test": ckpt_used,
                "test": test_res,
                "artifacts": {
                    "args_json": args_path,
                    "history_csv": history_path,
                    "log_file": log_file,
                    "best_json": best_json_path if os.path.exists(best_json_path) else None,
                    "test_json": test_json_path,
                },
            },
            summary_path,
        )
    except Exception:
        pass

    logger.info(f"Saved history: {history_path}")
    logger.info(f"Saved test results: {test_json_path}")
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
