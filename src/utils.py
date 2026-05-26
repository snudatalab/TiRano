# -*- coding: utf-8 -*-
"""
Utilities for Tirano (Temporal KG completion).

This file includes:
- deterministic seeding
- dataset IO helpers (txt <-> pickle)
- preprocessing helpers (inverse relations, filter dicts, RTNS alpha computation)
- NeighborFinder with Relation-adaptive Temporal Neighbor Sampling (RTNS)
"""

from __future__ import annotations

import os
import json
import math
import pickle
import random
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -------------------------
# Reproducibility / logging
# -------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (best-effort).

    Args:
        seed: random seed
        deterministic: if True, set cudnn deterministic flags (may reduce speed)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # local import to keep preprocessing lightweight
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass



def setup_logger(name: str = "tirano", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        # Ensure the parent directory exists (common when logs are written to a results folder)
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -------------
# Pickle / JSON
# -------------

def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------
# Dataset stat + raw parser
# ------------------------

def get_dataset_stat(stat_path: str) -> Tuple[int, int, int]:
    """
    Expected format: "<num_entities> <num_relations> <num_timestamps>"
    where num_relations is the number of *base* relations (before inverse augmentation).

    Returns:
        num_entities, num_relations_aug (= 2 * num_relations_base), num_timestamps (may be 0 if unknown)
    """
    with open(stat_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid stat file: {stat_path}")
    num_e = int(parts[0])
    num_r_base = int(parts[1])
    num_t = int(parts[2]) if len(parts) >= 3 else 0
    return num_e, 2 * num_r_base, num_t


# -------------------------
# Timestamp parsing (WIKI/YAGO-friendly)
# -------------------------

def parse_timestamp(ts: str, mode: str = "auto") -> int:
    """Parse a timestamp token into an integer.

    Tirano's pipeline expects timestamps as integers. Many TKG datasets already
    provide integer time indices (e.g., ICEWS / GDELT). However, interval-style
    benchmarks (e.g., Wikidata12k / YAGO11k) often use date strings such as:

      - "1926-##-##"  (unknown month/day)
      - "1952-05-##"  (unknown day)
      - "2014-05-23"  (full date)
      - "2014-05-23^^xsd:date" or "2014-05-23T00:00:00Z"

    This helper converts such tokens into an integer based on `mode`:
      - "auto"   : int if possible; otherwise, extract leading year
      - "int"    : require int
      - "year"   : always extract the leading year and return it as int
      - "ordinal": parse as YYYY-MM-DD (## treated as 01) and return date.toordinal()

    NOTE: Missing timestamps like "####-##-##" are not valid integers/years.
    Use :func:`parse_timestamp_optional` when missing values are expected.
    """
    ts = str(ts).strip()
    mode = (mode or "auto").lower()

    # Fast path: pure integer tokens.
    if mode in ("auto", "int"):
        try:
            return int(ts)
        except Exception:
            if mode == "int":
                raise

    # Strip common suffixes.
    if "^^" in ts:
        ts = ts.split("^^", 1)[0]
    if "t" in ts.lower() and "-" in ts:
        ts = ts.split("T", 1)[0]

    if mode in ("auto", "year"):
        import re
        m = re.match(r"^([+-]?\d{1,6})", ts)
        if m is None:
            raise ValueError(f"Cannot parse timestamp token: {ts!r}")
        return int(m.group(1))

    if mode == "ordinal":
        if "-" not in ts:
            return int(ts)
        parts = ts.split("-")
        if len(parts) < 1:
            raise ValueError(f"Cannot parse timestamp token: {ts!r}")
        year = int(parts[0])
        month = 1
        day = 1
        if len(parts) >= 2 and parts[1] not in ("##", "####", ""):
            month = int(parts[1])
        if len(parts) >= 3 and parts[2] not in ("##", "####", ""):
            day = int(parts[2])
        from datetime import date
        return int(date(year, month, day).toordinal())

    raise ValueError(f"Unknown timestamp parsing mode: {mode}")


def _is_missing_time_token(tok: str) -> bool:
    tok = str(tok).strip()
    if tok == "":
        return True
    if tok.startswith("#"):
        return True
    if tok.lower() in ("none", "nan", "null"):
        return True
    return False


def parse_timestamp_optional(ts: str, mode: str = "auto") -> Optional[int]:
    """Best-effort timestamp parsing that returns None for missing/invalid tokens."""
    if _is_missing_time_token(ts):
        return None
    try:
        return parse_timestamp(ts, mode=mode)
    except Exception:
        return None


def _is_int_token(tok: str) -> bool:
    tok = str(tok).strip()
    if tok.startswith(("+", "-")):
        return tok[1:].isdigit()
    return tok.isdigit()


def _looks_like_time_token(tok: str) -> bool:
    tok = str(tok).strip()
    if _is_missing_time_token(tok):
        return True
    if "-" in tok:
        return True
    if "^^" in tok:
        return True
    if "t" in tok.lower() and ":" in tok:
        return True
    return False


def read_quadruples_txt(
    path: str,
    time_mode: str = "auto",
    file_format: str = "auto",
    interval_mode: str = "start",
) -> List[Tuple[int, int, int, int, int]]:
    """Read a split file into a list of (s, r, o, t, event_idx).

    Supported raw formats:
      1) Quadruple:              s r o t
      2) Quadruple + event id:   s r o t event_idx
      3) Interval (WIKI/YAGO):   s r o t_start t_end

    For (3), Tirano internally uses a single integer timestamp per row. We
    convert the (start, end) interval into one (or multiple) timestamps via:

      - interval_mode="start"  : use min(start, end)
      - interval_mode="end"    : use max(start, end)
      - interval_mode="mid"    : use floor((min+max)/2)
      - interval_mode="random" : sample uniformly in [min, max] (non-deterministic)
      - interval_mode="expand" : expand into all integer t in [min, max] (inclusive)

    Missing boundary tokens like "####" / "####-##-##" are handled by using
    the available boundary.
    """
    if not path or (not os.path.exists(path)):
        return []

    file_format = (file_format or "auto").lower()
    if file_format not in ("auto", "quad", "interval"):
        raise ValueError(f"file_format must be one of auto|quad|interval, got: {file_format}")

    interval_mode = (interval_mode or "start").lower()
    if interval_mode not in ("start", "end", "mid", "random", "expand"):
        raise ValueError(
            f"interval_mode must be one of start|end|mid|random|expand, got: {interval_mode}"
        )

    data: List[Tuple[int, int, int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue

            # --- Resolve which raw format to use ---
            fmt = file_format
            if fmt == "auto":
                if len(parts) == 4:
                    fmt = "quad"
                elif len(parts) >= 5:
                    # Heuristic: if the last token looks like a timestamp (date string / ####),
                    # treat as an interval format; otherwise treat as event_idx.
                    fmt = "interval" if _looks_like_time_token(parts[4]) else "quad"

            if fmt == "quad":
                if len(parts) not in (4, 5):
                    raise ValueError(f"Unexpected columns ({len(parts)}) in {path}: {ln}")
                s = int(parts[0]); r = int(parts[1]); o = int(parts[2])
                t = parse_timestamp(parts[3], mode=time_mode)
                eidx = int(parts[4]) if len(parts) == 5 else 0
                data.append((s, r, o, int(t), int(eidx)))
                continue

            # fmt == "interval"
            if len(parts) < 5:
                raise ValueError(f"Interval format expects >=5 columns in {path}: {ln}")

            s = int(parts[0]); r = int(parts[1]); o = int(parts[2])

            t1 = parse_timestamp_optional(parts[3], mode=time_mode)
            t2 = parse_timestamp_optional(parts[4], mode=time_mode)
            if (t1 is None) and (t2 is None):
                # No valid time boundary -> skip (should be rare)
                continue

            if t1 is None:
                t_min = t_max = int(t2)  # type: ignore[arg-type]
            elif t2 is None:
                t_min = t_max = int(t1)
            else:
                t_min = int(min(t1, t2))
                t_max = int(max(t1, t2))

            if interval_mode == "expand" and t_max >= t_min:
                for t in range(t_min, t_max + 1):
                    data.append((s, r, o, int(t), 0))
            else:
                if interval_mode == "end":
                    t = t_max
                elif interval_mode == "mid":
                    t = (t_min + t_max) // 2
                elif interval_mode == "random":
                    # WARNING: non-deterministic unless you fix python random seed globally.
                    t = random.randint(t_min, t_max)
                else:  # "start"
                    t = t_min
                data.append((s, r, o, int(t), 0))

    return data
def infer_num_entities_relations(*splits: Sequence[Tuple[int, int, int, int, int]]) -> Tuple[int, int]:
    max_ent = 0
    max_rel = 0
    for data in splits:
        for s, r, o, _, _ in data:
            if s > max_ent: max_ent = s
            if o > max_ent: max_ent = o
            if r > max_rel: max_rel = r
    return max_ent + 1, max_rel + 1  # base relations


# -----------------------
# Inverse augmentation etc
# -----------------------

def augment_with_inverse(
    data: Sequence[Tuple[int, int, int, int, int]],
    num_rel_base: int,
) -> List[Tuple[int, int, int, int, int]]:
    """
    For each (s, r, o, t), add inverse (o, r+R, s, t).
    """
    out: List[Tuple[int, int, int, int, int]] = []
    for s, r, o, t, eidx in data:
        out.append((s, r, o, t, eidx))
        out.append((o, r + num_rel_base, s, t, eidx))
    return out


def build_filter_dicts(
    all_data_aug: Sequence[Tuple[int, int, int, int, int]]
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int, int], List[int]]]:
    """
    Builds sr2o and srt2o for filtered ranking evaluation.
    Values are stored as sorted unique lists to keep serialization smaller.
    """
    sr2o: Dict[Tuple[int, int], set] = {}
    srt2o: Dict[Tuple[int, int, int], set] = {}

    for s, r, o, t, _ in all_data_aug:
        sr2o.setdefault((s, r), set()).add(o)
        srt2o.setdefault((s, r, t), set()).add(o)

    sr2o_list = {k: sorted(v) for k, v in sr2o.items()}
    srt2o_list = {k: sorted(v) for k, v in srt2o.items()}
    return sr2o_list, srt2o_list


def build_o2srt_adj(
    train_aug: Sequence[Tuple[int, int, int, int, int]],
    num_entities: int,
) -> List[List[Tuple[int, int, int]]]:
    """
    Builds adjacency list keyed by object: adj[o] = [(s, r, t), ...] sorted by t.
    This matches Tirano's neighbor definition N_{s_q} using inverse augmentation.
    """
    adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(num_entities)]
    for s, r, o, t, _ in train_aug:
        adj[o].append((s, r, t))
    # sort by time for each entity
    for o in range(num_entities):
        if len(adj[o]) > 1:
            adj[o].sort(key=lambda x: x[2])
    return adj


def compute_relation2alpha(
    train_original: Sequence[Tuple[int, int, int, int, int]],
    num_rel_base: int,
    alpha_min: float = 1e-4,
    alpha_max: float = 100.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute RTNS temporal decay priors alpha_r = 1 / mu_r, where mu_r is the mean
    gap between consecutive occurrences of relation r in the *training* data.
    (We compute on original relations only, then copy to inverse relations.)
    """
    # Collect timestamps per relation
    times_per_rel: List[List[int]] = [[] for _ in range(num_rel_base)]
    for _, r, _, t, _ in train_original:
        if 0 <= r < num_rel_base:
            times_per_rel[r].append(t)
        else:
            # If caller accidentally passes augmented relations, fold back.
            rr = r % num_rel_base
            times_per_rel[rr].append(t)

    # global fallback mean gap
    global_diffs: List[float] = []
    for r in range(num_rel_base):
        ts = times_per_rel[r]
        if len(ts) >= 2:
            ts_sorted = np.sort(np.asarray(ts, dtype=np.int64))
            diffs = np.diff(ts_sorted).astype(np.float64)
            if diffs.size > 0:
                global_diffs.append(float(np.mean(diffs)))
    global_mu = float(np.mean(global_diffs)) if len(global_diffs) > 0 else 1.0
    global_mu = max(global_mu, 1.0)

    alpha_base = np.zeros((num_rel_base,), dtype=np.float32)
    for r in range(num_rel_base):
        ts = times_per_rel[r]
        if len(ts) < 2:
            mu = global_mu
        else:
            ts_sorted = np.sort(np.asarray(ts, dtype=np.int64))
            diffs = np.diff(ts_sorted).astype(np.float64)
            if diffs.size == 0:
                mu = global_mu
            else:
                mu = float(np.mean(diffs))
                if mu < eps:
                    mu = eps
        alpha = 1.0 / mu
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        alpha_base[r] = alpha

    alpha_full = np.concatenate([alpha_base, alpha_base], axis=0)  # inverse shares the same prior
    return alpha_full


# -------------------
# NeighborFinder (RTNS)
# -------------------

class NeighborFinder:
    """
    NeighborFinder for temporal KG.

    `adj` is a list:
        adj[obj] = [(ngh_entity, rel, ts), ...]  (sorted by ts ascending)

    sampling:
      - "uniform": uniform random sampling
      - "recent" : take most recent `num_neighbors`
      - "rtns"   : relation-adaptive temporal neighbor sampling (Eq. 1 in Tirano)
                    p(ngh | q) ∝ exp(-alpha_{r_q} * |t_q - t|^beta)

    time_mode (important for extrapolation / forecasting):
      - "all"            : sample from all neighbors regardless of time (can use "future" facts)
      - "past"           : sample only neighbors with ts <  t_q   (strict history; recommended for extrapolation)
      - "past_inclusive" : sample only neighbors with ts <= t_q  (history including same-timestamp facts)

    Notes:
      - For forecasting/extrapolation benchmarks, using "past" avoids information leakage
        (e.g., sampling the inverse of the current training fact at the same timestamp).
      - For interpolation benchmarks, "all" can be a reasonable choice when you assume the full
        training graph is observed (including events after the query timestamp).
    """

    def __init__(
        self,
        adj: List[List[Tuple[int, int, int]]],
        sampling: str = "rtns",
        relation2alpha: Optional[np.ndarray] = None,
        beta: float = 1.0,
        deterministic: bool = False,
        seed: int = 0,
        time_mode: str = "all",
    ) -> None:
        self.adj = adj
        self.sampling = sampling.lower()
        self.relation2alpha = relation2alpha  # shape (num_rel_aug,)
        self.beta = float(beta)
        self.deterministic = bool(deterministic)
        self._rng = np.random.RandomState(seed)

        self.time_mode = (time_mode or "all").lower().strip()
        if self.time_mode not in {"all", "past", "past_inclusive"}:
            raise ValueError(f"Unknown time_mode: {time_mode} (expected all|past|past_inclusive)")

    def set_adj(self, adj: List[List[Tuple[int, int, int]]]) -> None:
        self.adj = adj

    def set_time_mode(self, time_mode: str) -> None:
        time_mode = (time_mode or "all").lower().strip()
        if time_mode not in {"all", "past", "past_inclusive"}:
            raise ValueError(f"Unknown time_mode: {time_mode} (expected all|past|past_inclusive)")
        self.time_mode = time_mode

    def _get_alpha(self, rel_q: int) -> float:
        if self.relation2alpha is None:
            return 1.0
        if rel_q < 0 or rel_q >= len(self.relation2alpha):
            return float(np.mean(self.relation2alpha))
        return float(self.relation2alpha[rel_q])

    def _apply_time_filter(self, ent_arr: np.ndarray, rel_arr: np.ndarray, ts_arr: np.ndarray, t_q: int):
        """Filter candidate neighbors by query time based on self.time_mode.

        Assumes ts_arr is sorted ascending.
        Returns filtered arrays.
        """
        if self.time_mode == "all":
            return ent_arr, rel_arr, ts_arr

        # Since ts_arr is sorted, use binary search (O(log N)) rather than masking.
        if self.time_mode == "past":
            # keep ts < t_q
            end = int(np.searchsorted(ts_arr, t_q, side="left"))
        else:  # past_inclusive
            # keep ts <= t_q
            end = int(np.searchsorted(ts_arr, t_q, side="right"))

        if end <= 0:
            return ent_arr[:0], rel_arr[:0], ts_arr[:0]
        return ent_arr[:end], rel_arr[:end], ts_arr[:end]

    def get_temporal_neighbor(
        self,
        obj_idx_l: Sequence[int],
        ts_l: Sequence[int],
        num_neighbors: int = 50,
        rel_q_l: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            ngh_ent: (B, K) int64, padded with -1
            ngh_rel: (B, K) int64, padded with -1
            ngh_ts : (B, K) int64, padded with 0 (only meaningful if ent>=0)
        """
        B = len(obj_idx_l)
        K = int(num_neighbors)
        ngh_ent = np.full((B, K), -1, dtype=np.int64)
        ngh_rel = np.full((B, K), -1, dtype=np.int64)
        ngh_ts = np.zeros((B, K), dtype=np.int64)

        if rel_q_l is None:
            rel_q_l = [0] * B

        for i, (obj, t_q, r_q) in enumerate(zip(obj_idx_l, ts_l, rel_q_l)):
            neighbors = self.adj[int(obj)]
            if len(neighbors) == 0:
                continue

            # unpack (adj[obj] is sorted by ts)
            ent_arr = np.asarray([x[0] for x in neighbors], dtype=np.int64)
            rel_arr = np.asarray([x[1] for x in neighbors], dtype=np.int64)
            ts_arr = np.asarray([x[2] for x in neighbors], dtype=np.int64)

            # --- time filter (forecasting/extrapolation) ---
            ent_arr, rel_arr, ts_arr = self._apply_time_filter(ent_arr, rel_arr, ts_arr, int(t_q))
            if ent_arr.size == 0:
                continue

            if self.sampling == "recent":
                # take most recent K
                take = min(K, ent_arr.size)
                ent_sel = ent_arr[-take:]
                rel_sel = rel_arr[-take:]
                ts_sel = ts_arr[-take:]
            elif self.sampling == "uniform":
                take = min(K, ent_arr.size)
                idx = self._rng.choice(ent_arr.size, size=take, replace=False)
                ent_sel = ent_arr[idx]
                rel_sel = rel_arr[idx]
                ts_sel = ts_arr[idx]
            elif self.sampling == "rtns":
                alpha = self._get_alpha(int(r_q))
                dt = np.abs(ts_arr.astype(np.float64) - float(t_q))
                # weights = exp(-alpha * |dt|^beta)
                # subtract max for numerical stability
                w = -alpha * np.power(dt, self.beta)
                w = np.exp(w - np.max(w))
                if not np.isfinite(w).all():
                    # fallback to uniform
                    w = np.ones_like(w, dtype=np.float64)
                w_sum = float(np.sum(w))
                if w_sum <= 0:
                    w = np.ones_like(w, dtype=np.float64) / float(len(w))
                else:
                    w = w / w_sum

                take = min(K, ent_arr.size)
                if self.deterministic:
                    # top-k by weight (deterministic, less variance)
                    idx = np.argsort(-w)[:take]
                else:
                    # np.random.choice(..., replace=False, p=w) requires >=take non-zero entries in w.
                    # With large timestamps or aggressive decay, exp() can underflow to exact zeros.
                    if int(np.count_nonzero(w)) < int(take):
                        w = np.maximum(w, 1e-12)
                        w = w / float(np.sum(w))
                    idx = self._rng.choice(ent_arr.size, size=take, replace=False, p=w)
                ent_sel = ent_arr[idx]
                rel_sel = rel_arr[idx]
                ts_sel = ts_arr[idx]
            else:
                raise ValueError(f"Unknown sampling: {self.sampling}")

            # pad (left-aligned)
            take = int(ent_sel.size)
            ngh_ent[i, :take] = ent_sel
            ngh_rel[i, :take] = rel_sel
            ngh_ts[i, :take] = ts_sel

        return ngh_ent, ngh_rel, ngh_ts
