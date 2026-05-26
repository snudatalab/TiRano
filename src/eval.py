# -*- coding: utf-8 -*-
"""Evaluation utilities for temporal knowledge graph completion.

We intentionally keep evaluation **simple and paper-aligned**:

- **Raw** ranking (no filtering): computed optionally.
- **Time-filtered** ranking: filters other true entities for the same pair.

Assumes ``model(batch, ...)`` returns scores over all candidate tail entities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm


def _to_set_dict(d):
    # stored as list for serialization -> convert to set for faster membership
    if not d:
        return {}
    k = next(iter(d.keys()))
    if isinstance(d[k], set):
        return d
    return {kk: set(vv) for kk, vv in d.items()}


@torch.no_grad()
def eval_link_prediction(
    model,
    data_loader,
    sr2o: Dict[Tuple[int, int], List[int]],
    srt2o: Dict[Tuple[int, int, int], List[int]],
    num_entities: int,
    device: torch.device,
    num_neighbors: Optional[int] = None,
    eval_time_filtered: bool = True,
    return_raw: bool = True,
    verbose: bool = True,
):
    sr2o_s = _to_set_dict(sr2o)

    ranks_raw: List[int] = []
    ranks_tf: List[int] = []  # time-filtered 

    iterator = tqdm(data_loader, desc="Eval", leave=False) if verbose else data_loader

    for batch in iterator:
        model.eval()
        if num_neighbors is None:
            scores = model(batch)  # (B, num_entities)
        else:
            try:
                scores = model(batch, num_neighbors=int(num_neighbors))  # (B, num_entities)
            except TypeError:
                scores = model(batch)
        scores = scores.detach().cpu()

        B = scores.shape[0]
        src = batch.src_idx.detach().cpu().numpy()
        rel = batch.rel_idx.detach().cpu().numpy()
        ts = batch.ts.detach().cpu().numpy()
        tgt = batch.target_idx.detach().cpu().numpy()

        for i in range(B):
            s_i = int(src[i]); r_i = int(rel[i]); t_i = int(ts[i]); o_i = int(tgt[i])

            # ---- raw rank----
            if return_raw:
                s_vec_raw = scores[i]  # (num_entities,)
                o_score = float(s_vec_raw[o_i])
                rank_raw = int((s_vec_raw > o_score).sum().item()) + 1
                ranks_raw.append(rank_raw)

            # ---- time-filtered rank ----
            if eval_time_filtered:
                s_vec_f = scores[i].clone()
                filt = sr2o_s.get((s_i, r_i), set())
                if filt:
                    # keep the true target
                    for cand in filt:
                        if cand != o_i:
                            s_vec_f[cand] = -1e9
                o_score_f = float(s_vec_f[o_i])
                rank_tf = int((s_vec_f > o_score_f).sum().item()) + 1
                ranks_tf.append(rank_tf)

    def _metrics(ranks: List[int]):
        if len(ranks) == 0:
            return dict(hits1=0.0, hits3=0.0, hits10=0.0, mrr=0.0, mar=0.0)
        r = np.asarray(ranks, dtype=np.float64)
        return dict(
            hits1=float(np.mean(r <= 1)),
            hits3=float(np.mean(r <= 3)),
            hits10=float(np.mean(r <= 10)),
            mrr=float(np.mean(1.0 / r)),
            mar=float(np.mean(r)),
        )

    results = {}
    if eval_time_filtered:
        results.update(_metrics(ranks_tf))
    if return_raw:
        results.update({f"{k}_raw": v for k, v in _metrics(ranks_raw).items()})

    return results
