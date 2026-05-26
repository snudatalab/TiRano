# -*- coding: utf-8 -*-
"""
Pretrain neighbor-aware KG embeddings to initialize Tirano's embedding tables.

Goal:
  - Fill entity/relation embedding tables with features that encode neighborhood information
  - Save a checkpoint usable by Tirano.load_pretrained_embeddings()

We pretrain a simple Neighbor-Aware DistMult:
  h_e = LN( E[e] + mean_{(e, r, n) in N(e)} ( R[r] ⊙ E[n] ) )
and DistMult score:
  score(s,r,o) = < h_s ⊙ R[r], h_o >

Training:
  - Use augmented train triples (including inverse relations) from processed/train_data.pkl
  - Ignore timestamps (static pretraining)
  - Negative sampling on tails

Outputs:
  <out_dir>/pretrained_init.pt containing:
    - entity_init: (num_entities, embed_dim)  (neighbor-aware features)
    - relation_init: (num_relations_aug, embed_dim)
    - meta
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import load_pickle, save_json, save_pickle, set_seed, setup_logger


class TripleDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, int, int]]):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        return self.triples[idx]


def collate_triples(batch):
    s = torch.tensor([x[0] for x in batch], dtype=torch.long)
    r = torch.tensor([x[1] for x in batch], dtype=torch.long)
    o = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return s, r, o


class NeighborAwareDistMult(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embed_dim: int, adj_out):
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embed_dim = int(embed_dim)

        self.E = nn.Embedding(self.num_entities, self.embed_dim)
        self.R = nn.Embedding(self.num_relations, self.embed_dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)

        self.ln = nn.LayerNorm(self.embed_dim)
        self.adj_out = adj_out  # list of (neighbors, rels)
        self.dropout = nn.Dropout(0.1)

    @torch.no_grad()
    def _sample_neighbors(self, e: int, k: int, rng: np.random.RandomState):
        neigh, rel = self.adj_out[e]
        if len(neigh) == 0:
            return [], []
        if len(neigh) <= k:
            idx = np.arange(len(neigh))
        else:
            idx = rng.choice(len(neigh), size=k, replace=False)
        return [neigh[i] for i in idx], [rel[i] for i in idx]

    def aggregate(self, ent_ids: torch.Tensor, num_neighbors: int = 20, rng: Optional[np.random.RandomState] = None) -> torch.Tensor:
        """
        ent_ids: (N,)
        returns: h: (N, d) neighbor-aware
        """
        device = ent_ids.device
        N = ent_ids.numel()
        base = self.E(ent_ids)  # (N,d)

        if rng is None:
            rng = np.random.RandomState(0)

        # sample neighbors per entity (python loop, N is small per step)
        neigh_ids = []
        rel_ids = []
        row_ptr = [0]

        ent_ids_cpu = ent_ids.detach().cpu().tolist()
        for e in ent_ids_cpu:
            nlist, rlist = self._sample_neighbors(int(e), num_neighbors, rng)
            neigh_ids.extend(nlist)
            rel_ids.extend(rlist)
            row_ptr.append(len(neigh_ids))

        if len(neigh_ids) == 0:
            return self.ln(base)

        neigh_ids_t = torch.tensor(neigh_ids, dtype=torch.long, device=device)
        rel_ids_t = torch.tensor(rel_ids, dtype=torch.long, device=device)

        neigh_emb = self.E(neigh_ids_t)  # (M,d)
        rel_emb = self.R(rel_ids_t)      # (M,d)
        msg = neigh_emb * rel_emb        # (M,d)

        # mean per row
        out = torch.zeros_like(base)
        for i in range(N):
            st = row_ptr[i]
            ed = row_ptr[i + 1]
            if st == ed:
                continue
            out[i] = msg[st:ed].mean(dim=0)

        h = self.ln(base + self.dropout(out))
        return h

    def score(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor, rng: np.random.RandomState, neigh_k: int = 20) -> torch.Tensor:
        # compute neighbor-aware embeddings for unique entities to reduce repeats
        ids = torch.cat([s, o], dim=0)
        uniq, inv = torch.unique(ids, return_inverse=True)
        h_uniq = self.aggregate(uniq, num_neighbors=neigh_k, rng=rng)
        h_s = h_uniq[inv[: s.numel()]]
        h_o = h_uniq[inv[s.numel():]]
        h_r = self.R(r)
        return (h_s * h_r * h_o).sum(dim=-1)  # (B,)


def build_outgoing_adj(triples: List[Tuple[int, int, int]], num_entities: int):
    """
    Build outgoing adjacency for neighbor-aware aggregation:
      adj_out[s] contains outgoing (o, r)
    """
    neigh = [[] for _ in range(num_entities)]
    rels = [[] for _ in range(num_entities)]
    for s, r, o in triples:
        neigh[s].append(o)
        rels[s].append(r)
    adj_out = [(neigh[i], rels[i]) for i in range(num_entities)]
    return adj_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--embed_dim", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--num_neighbors", type=int, default=20, help="neighbors sampled in aggregation during training")
    ap.add_argument("--num_neighbors_export", type=int, default=50, help="neighbors sampled when exporting entity_init")
    ap.add_argument("--neg_ratio", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="where to save logs/metrics (default: dataset/<ds>/processed)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    dname = args.dataset.lower()
    proc_dir = os.path.join(args.data_dir, dname, "processed")
    train_path = os.path.join(proc_dir, "train_data.pkl")
    meta_path = os.path.join(proc_dir, "meta.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found. Run preprocess.py first.")

    # logging / artifacts
    log_dir = args.log_dir.strip() if args.log_dir else ""
    if not log_dir:
        log_dir = proc_dir
    os.makedirs(log_dir, exist_ok=True)
    run_key = f"pretrain_{dname}_dim{args.embed_dim}"
    log_file = os.path.join(log_dir, run_key + ".log")
    hist_path = os.path.join(log_dir, run_key + "_history.csv")
    args_path = os.path.join(log_dir, run_key + "_args.json")
    logger = setup_logger("pretrain", log_file=log_file)
    logger.info(f"Dataset: {args.dataset}  device={args.device}")
    logger.info(str(vars(args)))
    save_json(
        {
            "args": vars(args),
            "env": {
                "python": sys.version,
                "torch": getattr(torch, "__version__", ""),
                "cuda_available": bool(torch.cuda.is_available()),
            },
        },
        args_path,
    )

    def append_hist(epoch: int, loss: float):
        file_exists = os.path.exists(hist_path)
        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss"])
            if not file_exists:
                w.writeheader()
            w.writerow({"epoch": int(epoch), "loss": float(loss)})

    train_aug = load_pickle(train_path)
    meta = None
    if os.path.exists(meta_path):
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Convert to static triples (s,r,o)
    triples = [(int(s), int(r), int(o)) for (s, r, o, _, _) in train_aug]
    if meta is not None:
        num_entities = int(meta["num_entities"])
        num_relations = int(meta["num_relations_aug"])
    else:
        num_entities = max(max(s, o) for s, _, o in triples) + 1
        num_relations = max(r for _, r, _ in triples) + 1

    adj_out = build_outgoing_adj(triples, num_entities=num_entities)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = NeighborAwareDistMult(num_entities, num_relations, args.embed_dim, adj_out=adj_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = TripleDataset(triples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_triples)

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(dl, desc=f"Pretrain epoch {epoch}", leave=False)
        for s, r, o in pbar:
            s = s.to(device)
            r = r.to(device)
            o = o.to(device)

            # positives
            score_pos = model.score(s, r, o, rng=rng, neigh_k=args.num_neighbors)  # (B,)
            y_pos = torch.ones_like(score_pos)

            # negatives (corrupt tail)
            B = s.size(0)
            neg_o = torch.randint(low=0, high=num_entities, size=(B * args.neg_ratio,), device=device)
            s_rep = s.repeat_interleave(args.neg_ratio)
            r_rep = r.repeat_interleave(args.neg_ratio)

            score_neg = model.score(s_rep, r_rep, neg_o, rng=rng, neigh_k=args.num_neighbors)
            y_neg = torch.zeros_like(score_neg)

            loss = bce(score_pos, y_pos) + bce(score_neg, y_neg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.item())
            n_steps += 1
            pbar.set_postfix(loss=total_loss / max(1, n_steps))

        epoch_loss = float(total_loss / max(1, n_steps))
        logger.info(f"[Pretrain] epoch={epoch} loss={epoch_loss:.4f}")
        try:
            append_hist(epoch, epoch_loss)
        except Exception:
            pass

    # Export neighbor-aware entity features as initialization
    model.eval()
    with torch.no_grad():
        entity_init = torch.zeros((num_entities, args.embed_dim), dtype=torch.float32)
        batch = 512
        for st in tqdm(range(0, num_entities, batch), desc="Export entity_init"):
            ed = min(num_entities, st + batch)
            ent_ids = torch.arange(st, ed, dtype=torch.long, device=device)
            h = model.aggregate(ent_ids, num_neighbors=args.num_neighbors_export, rng=rng).detach().cpu()
            entity_init[st:ed] = h

        relation_init = model.R.weight.detach().cpu().float()

    out_ckpt = os.path.join(proc_dir, f"pretrained_init_dim{args.embed_dim}.pt")
    torch.save(
        dict(
            entity_init=entity_init,
            relation_init=relation_init,
            meta=dict(
                dataset=args.dataset,
                num_entities=num_entities,
                num_relations=num_relations,
                embed_dim=args.embed_dim,
                score_func="distmult",
            ),
        ),
        out_ckpt,
    )
    logger.info("Saved pretrained checkpoint: %s", out_ckpt)
    logger.info("Saved pretrain history: %s", hist_path)


if __name__ == "__main__":
    main()
