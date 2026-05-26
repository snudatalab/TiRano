# -*- coding: utf-8 -*-
"""Tirano.

We build only the *observed* (pair,time) sequences:
- For each query in the batch, we sample K temporal neighbors (RTNS).
- For the observed (entity, relation) pairs, we build a tensor
    seq:  (P, C_in, T)
  where P is the number of unique (b, e_slot, r_slot) pairs that appear.

We then apply a relation-conditioned multi-scale temporal CNN over time (1D),
propagate the mask through the CNN (max-pooling), and aggregate back to a
batch-level context vector.

"""

from __future__ import annotations

import math
import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NeighborFinder


# -----------------
# Helper: masking
# -----------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Sequence[int], eps: float = 1e-9) -> torch.Tensor:
    """Masked mean over given dims."""
    mask_f = mask.to(dtype=x.dtype)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(eps)
    return num / den


# -----------------------------
# Relation-conditioned ST-CNN
# -----------------------------

class RelationGate(nn.Module):
    """\pi_{r_q} = softmax(MLP(h_{r_q})) \in R^M"""

    def __init__(self, rel_dim: int, num_scales: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        h = int(hidden_dim) if hidden_dim is not None else max(64, int(rel_dim))
        self.net = nn.Sequential(
            nn.Linear(int(rel_dim), h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h, int(num_scales)),
        )

    def forward(self, rel_emb: torch.Tensor) -> torch.Tensor:
        logits = self.net(rel_emb)
        return F.softmax(logits, dim=-1)


class MultiScaleDepthwiseTemporalConv1D(nn.Module):
    """One multi-scale depthwise temporal conv layer with dilation mixture.

    Input:
      x:    (P, C_in, T)
      mask: (P, 1,    T)  (binary)
      gate: (P, M) mixture weights for each pair

    Output:
      x':   (P, C_in, T)
      mask':(P, 1,    T)

    We use SAME padding so that length T is preserved.
    Mask is propagated by max-pooling with the same receptive fields.
    """

    def __init__(self, c_in: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        self.c_in = int(c_in)
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        assert self.kernel_size % 2 == 1, "Use odd kernel_size for same padding"
        assert len(self.dilations) >= 1

        self.dw_convs = nn.ModuleList()
        self._pads = []
        for d in self.dilations:
            pad = (d * (self.kernel_size - 1)) // 2
            self._pads.append(pad)
            self.dw_convs.append(
                nn.Conv1d(
                    in_channels=self.c_in,
                    out_channels=self.c_in,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=pad,
                    dilation=d,
                    groups=self.c_in,
                    bias=False,
                )
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # gate: (P, M)
        P, C, T = x.shape
        M = len(self.dilations)
        assert gate.shape == (P, M)

        y_mix = torch.zeros_like(x)
        # mask union across dilations
        mask_u = mask.new_zeros(mask.shape)

        # Use float mask for pooling
        mask_f = mask.to(dtype=x.dtype)

        for m, d in enumerate(self.dilations):
            y = self.dw_convs[m](x)  # (P,C,T)
            w = gate[:, m].view(P, 1, 1).to(dtype=x.dtype)
            y_mix = y_mix + w * y

            pad = self._pads[m]
                        # Propagate validity mask through the same receptive field as the dilated depthwise conv.
            # max_pool1d has a hard constraint padding <= kernel_size/2, which breaks for dilation>1.
            # Instead, use a 1D convolution with an all-ones kernel and the same dilation/padding, then threshold.
            if not hasattr(self, '_mask_kernel'):
                self._mask_kernel = None
            if self._mask_kernel is None or self._mask_kernel.device != mask_f.device or self._mask_kernel.dtype != mask_f.dtype:
                self._mask_kernel = torch.ones((1, 1, self.kernel_size), device=mask_f.device, dtype=mask_f.dtype)
            m_out = F.conv1d(mask_f, self._mask_kernel, bias=None, stride=1, padding=pad, dilation=d)
            mask_u = torch.maximum(mask_u, (m_out > 0).to(mask_u.dtype))

        return y_mix, mask_u


class RelationConditionedSparseSTCNN1D(nn.Module):
    """Relation-conditioned multi-scale temporal CNN over (pair,time) sequences.

    Inputs:
      x:    (P, C_in, T)
      mask: (P, 1,    T)
      gate: (P, M)

    Outputs:
      y:    (P, C_out, T)
      mask: (P, 1,     T)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        dilations: Sequence[int],
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.num_layers = int(num_layers)

        self.layers = nn.ModuleList([
            MultiScaleDepthwiseTemporalConv1D(self.c_in, kernel_size=kernel_size, dilations=dilations)
            for _ in range(self.num_layers)
        ])
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.pw = nn.Conv1d(self.c_in, self.c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # conv blocks in FP32 for stability with depthwise conv
        orig_dtype = x.dtype
        if torch.is_autocast_enabled():
            # autocast region: do conv in float32, then cast back
            with torch.autocast(device_type=x.device.type, enabled=False):
                x32 = x.float()
                m32 = mask.to(dtype=torch.float32)
                g32 = gate.to(dtype=torch.float32)
                for layer in self.layers:
                    x32, m32 = layer(x32, m32, g32)
                    x32 = self.act(x32)
                    x32 = self.drop(x32)
                y32 = self.pw(x32)
                y32 = self.drop(y32)
                m_out = (m32 > 0).to(dtype=mask.dtype)
                return y32.to(orig_dtype), m_out
        else:
            for layer in self.layers:
                x, mask = layer(x, mask, gate)
                x = self.act(x)
                x = self.drop(x)
            y = self.pw(x)
            y = self.drop(y)
            return y, mask


# -------------------
# KG scoring modules
# -------------------

def score_distmult(h_s: torch.Tensor, h_r: torch.Tensor, ent_all: torch.Tensor) -> torch.Tensor:
    return (h_s * h_r) @ ent_all.t()


def score_complex(h_s: torch.Tensor, h_r: torch.Tensor, ent_all: torch.Tensor) -> torch.Tensor:
    d2 = h_s.size(-1)
    if d2 % 2 != 0:
        raise ValueError("ComplEx requires even embedding dim (re||im).")
    d = d2 // 2

    s_re, s_im = h_s[..., :d], h_s[..., d:]
    r_re, r_im = h_r[..., :d], h_r[..., d:]
    o_re, o_im = ent_all[:, :d], ent_all[:, d:]

    term1 = (s_re * r_re - s_im * r_im) @ o_re.t()
    term2 = (s_re * r_im + s_im * r_re) @ o_im.t()
    return term1 + term2


def hamilton_product(a, b, c, d, ra, rb, rc, rd):
    A = a * ra - b * rb - c * rc - d * rd
    B = a * rb + b * ra + c * rd - d * rc
    C = a * rc - b * rd + c * ra + d * rb
    D = a * rd + b * rc - c * rb + d * ra
    return A, B, C, D


def score_bique(h_s: torch.Tensor, h_r: torch.Tensor, ent_all: torch.Tensor) -> torch.Tensor:
    d4 = h_s.size(-1)
    if d4 % 4 != 0:
        raise ValueError("BiQUE requires embedding dim divisible by 4 (a||b||c||d).")
    d = d4 // 4

    sa, sb, sc, sd = h_s[..., :d], h_s[..., d:2 * d], h_s[..., 2 * d:3 * d], h_s[..., 3 * d:]
    ra, rb, rc, rd = h_r[..., :d], h_r[..., d:2 * d], h_r[..., 2 * d:3 * d], h_r[..., 3 * d:]
    oa, ob, oc, od = ent_all[:, :d], ent_all[:, d:2 * d], ent_all[:, 2 * d:3 * d], ent_all[:, 3 * d:]

    pa, pb, pc, pd = hamilton_product(sa, sb, sc, sd, ra, rb, rc, rd)
    return pa @ oa.t() + pb @ ob.t() + pc @ oc.t() + pd @ od.t()


# --------------
# Tirano model
# --------------

class Tirano(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        nf: NeighborFinder,
        embed_dim: int = 200,
        score_func: str = "distmult",
        # entity embedding split
        # - "shared": (default in many codebases) one trainable embedding table is used everywhere.
        # - "pretrained_frozen": keep a separate, *frozen* entity embedding table for the
        #   "static entity embedding" term in the final query representation, initialized from
        #   the pretrained KG checkpoint; the main (trainable) entity embedding table is still
        #   updated and used for RTNS aggregation/CNN features and for candidate scoring.
        # This matches your desired behavior:
        #   final_query = [CNN_feat || AGG_feat || E_static_pretrained(frozen)]
        static_entity_mode: str = "pretrained_frozen",
        # snapshot / ST-CNN
        max_entities: int = 50,
        max_relations: int = 50,
        window_size: int = 50,     # past window size m (relative to query time)
        window_future: Optional[int] = None,  # future window size n (default: same as window_size)
        bin_width: int = 1,        # \Delta
        beta: float = 1.0,
        hr_c: int = 128,           # bottleneck for snapshot features
        c_out: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4),
        num_cnn_layers: int = 2,
        gate_hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        sigma_mask_empty: bool = True,
        # RTNS prior for gamma init
        relation2alpha: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.nf = nf

        self.embed_dim = int(embed_dim)
        self.score_func = score_func.lower()

        self.static_entity_mode = str(static_entity_mode).lower().strip()
        if self.static_entity_mode not in {"shared", "pretrained_frozen"}:
            raise ValueError(
                f"static_entity_mode must be one of ['shared','pretrained_frozen'], got: {static_entity_mode}"
            )

        self.max_entities = int(max_entities)
        self.max_relations = int(max_relations)
        self.window_past = int(window_size)
        self.window_future = int(window_future) if (window_future is not None) else int(window_size)
        self.window_size = int(window_size)  # backward-compat alias
        self.bin_width = int(bin_width)
        self.beta = float(beta)
        self.sigma_mask_empty = bool(sigma_mask_empty)

        # embedding dimensionality in real vector space
        if self.score_func == "distmult":
            self.ent_dim = self.embed_dim
        elif self.score_func == "complex":
            self.ent_dim = 2 * self.embed_dim  # re||im
        elif self.score_func == "bique":
            self.ent_dim = 4 * self.embed_dim  # a||b||c||d
        else:
            raise ValueError(f"Unknown score_func: {score_func}")

        # Entity / relation embeddings
        # - entity_emb: trainable table (updated during training)
        # - entity_emb_static: optional frozen table used ONLY for the 'static entity embedding'
        #   concatenated into the final query representation.
        self.entity_emb = nn.Embedding(self.num_entities, self.ent_dim)
        self.relation_emb = nn.Embedding(self.num_relations, self.ent_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.entity_emb_static: Optional[nn.Embedding] = None
        if self.static_entity_mode == "pretrained_frozen":
            self.entity_emb_static = nn.Embedding(self.num_entities, self.ent_dim)
            nn.init.xavier_uniform_(self.entity_emb_static.weight)
            # frozen by default (can still be overwritten by load_pretrained_embeddings)
            self.entity_emb_static.weight.requires_grad_(False)

        # Snapshot bottleneck: use ONLY the first embed_dim slice for features
        self.hr_c = int(hr_c)
        self.hr_proj = nn.Linear(2 * self.embed_dim, self.hr_c)

        # Learnable gamma for slice weights; init from RTNS alpha prior if provided.
        if relation2alpha is not None:
            alpha = torch.tensor(list(relation2alpha), dtype=torch.float32)
            if alpha.numel() != self.num_relations:
                if alpha.numel() * 2 == self.num_relations:
                    alpha = torch.cat([alpha, alpha], dim=0)
                else:
                    alpha = alpha.new_full((self.num_relations,), float(alpha.mean()))
        else:
            alpha = torch.full((self.num_relations,), 0.1, dtype=torch.float32)

        def inv_softplus(x: torch.Tensor) -> torch.Tensor:
            return torch.log(torch.expm1(x).clamp_min(1e-8))

        self.gamma_raw = nn.Parameter(inv_softplus(alpha.clamp_min(1e-4)))

        # Gate conditioned on query relation embedding
        self.gate = RelationGate(rel_dim=self.ent_dim, num_scales=len(list(dilations)), hidden_dim=gate_hidden_dim, dropout=dropout)

        # Relation-conditioned sparse ST-CNN over time
        self.c_out = int(c_out)
        self.stcnn = RelationConditionedSparseSTCNN1D(
            c_in=self.hr_c,
            c_out=self.c_out,
            kernel_size=kernel_size,
            dilations=dilations,
            num_layers=num_cnn_layers,
            dropout=dropout,
        )

        # Fuse: tanh(W [z_CNN || z_Agg || E_ent[s_q]])
        self.fuse = nn.Linear(self.c_out + self.hr_c + self.ent_dim, self.ent_dim)
        self.fuse_act = nn.Tanh()

        self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout)

    # -------------------------
    # Pretrained initialization
    # -------------------------

    def load_pretrained_embeddings(
        self,
        ckpt_path: str,
        strict: bool = False,
        *,
        load_dynamic: bool = True,
        load_static: bool = True,
    ) -> None:
        """Load a pretrained embedding checkpoint produced by pretrain_kg.py.

        Checkpoint keys:
          - entity_init: (num_entities, embed_dim)
          - relation_init: (num_relations, embed_dim)

        For complex/bique scoring embeddings, we copy the pretrained vector to the
        first component and zero-fill the rest.
        """

        # NOTE (PyTorch>=2.6): torch.load defaults to weights_only=True, which can
        # fail when our pretrained init checkpoint stores numpy arrays.
        # Our pretrain_kg.py produces this checkpoint, so we explicitly allow
        # full unpickling for compatibility.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "entity_init" not in ckpt or "relation_init" not in ckpt:
            raise ValueError(f"Checkpoint {ckpt_path} missing keys: entity_init / relation_init")

        log = logging.getLogger("tirano")

        ent = torch.as_tensor(ckpt["entity_init"], dtype=torch.float32)
        rel = torch.as_tensor(ckpt["relation_init"], dtype=torch.float32)

        def _fit_matrix(mat: torch.Tensor, target_rows: int, target_cols: int, name: str) -> torch.Tensor:
            if mat.ndim != 2:
                raise ValueError(f"{name} must be 2D, got shape={tuple(mat.shape)}")

            # Columns (embedding dim)
            if mat.shape[1] != target_cols:
                if strict:
                    raise ValueError(f"{name} dim mismatch: ckpt={mat.shape[1]} expected={target_cols}")
                if mat.shape[1] > target_cols:
                    log.warning(
                        f"{name} dim mismatch: ckpt={mat.shape[1]} > expected={target_cols}. Truncating columns."
                    )
                    mat = mat[:, :target_cols].contiguous()
                else:
                    pad = torch.empty((mat.shape[0], target_cols - mat.shape[1]), dtype=mat.dtype)
                    nn.init.xavier_uniform_(pad)
                    log.warning(
                        f"{name} dim mismatch: ckpt={mat.shape[1]} < expected={target_cols}. Padding columns with Xavier init."
                    )
                    mat = torch.cat([mat, pad], dim=1)

            # Rows (count)
            if mat.shape[0] != target_rows:
                if strict:
                    raise ValueError(f"{name} count mismatch: ckpt={mat.shape[0]} expected={target_rows}")
                if mat.shape[0] > target_rows:
                    log.warning(
                        f"{name} count mismatch: ckpt={mat.shape[0]} > expected={target_rows}. Truncating rows."
                    )
                    mat = mat[:target_rows].contiguous()
                else:
                    extra = torch.empty((target_rows - mat.shape[0], mat.shape[1]), dtype=mat.dtype)
                    nn.init.xavier_uniform_(extra)
                    log.warning(
                        f"{name} count mismatch: ckpt={mat.shape[0]} < expected={target_rows}. Padding rows with Xavier init."
                    )
                    mat = torch.cat([mat, extra], dim=0)

            return mat

        ent = _fit_matrix(ent, self.num_entities, self.embed_dim, name="entity_init")
        rel = _fit_matrix(rel, self.num_relations, self.embed_dim, name="relation_init")

        # Adapt to scoring embedding size (ent_dim)
        if self.ent_dim == self.embed_dim:
            ent_full = ent
            rel_full = rel
        else:
            ent_full = torch.zeros((self.num_entities, self.ent_dim), dtype=torch.float32)
            rel_full = torch.zeros((self.num_relations, self.ent_dim), dtype=torch.float32)
            ent_full[:, : self.embed_dim] = ent
            rel_full[:, : self.embed_dim] = rel

        with torch.no_grad():
            if load_dynamic:
                self.entity_emb.weight.copy_(ent_full)
                self.relation_emb.weight.copy_(rel_full)
            if load_static and (self.entity_emb_static is not None):
                self.entity_emb_static.weight.copy_(ent_full)

    def _get_static_entity_emb(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the 'static' entity embedding used in the final query representation."""
        if self.entity_emb_static is not None:
            return self.entity_emb_static(idx)
        return self.entity_emb(idx)

    # -------------------
    # Core forward
    # -------------------

    def _score_all(self, h_s: torch.Tensor, h_r: torch.Tensor) -> torch.Tensor:
        ent_all = self.entity_emb.weight
        if self.score_func == "distmult":
            return score_distmult(h_s, h_r, ent_all)
        if self.score_func == "complex":
            return score_complex(h_s, h_r, ent_all)
        if self.score_func == "bique":
            return score_bique(h_s, h_r, ent_all)
        raise RuntimeError("Invalid score_func")

    def loss(self, scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(scores, target_idx)

    @staticmethod
    def _assign_slots_firstk(b_t: torch.Tensor, ids_t: torch.Tensor, budget: int, id_bound: int) -> torch.Tensor:
        """Assign compact [0..budget-1] slots per-batch in first-appearance order.

        Others -> slot == budget (i.e., 'others' bucket).

        This is a vectorized version (no python loop over B).
        """
        if budget <= 0:
            return torch.zeros_like(ids_t)

        device = b_t.device
        BIG = int(id_bound + 1)
        key = b_t * BIG + ids_t  # (N,)

        idx_sorted = torch.argsort(key)
        key_sorted = key[idx_sorted]
        first_mask = torch.ones_like(key_sorted, dtype=torch.bool, device=device)
        first_mask[1:] = key_sorted[1:] != key_sorted[:-1]

        uniq_idx_in_sorted = torch.nonzero(first_mask, as_tuple=False).squeeze(-1)
        unique_keys = key_sorted[uniq_idx_in_sorted]  # (U,)
        first_pos = idx_sorted[uniq_idx_in_sorted]    # (U,) original positions
        b_unique = b_t[first_pos]                     # (U,)

        # order unique keys by (batch, first_pos)
        BIGN = int(key.numel() + 1)
        order2 = torch.argsort(b_unique * BIGN + first_pos)
        b2 = b_unique[order2]

        start_mask = torch.ones_like(b2, dtype=torch.bool, device=device)
        start_mask[1:] = b2[1:] != b2[:-1]
        group_id = torch.cumsum(start_mask.to(torch.long), dim=0) - 1
        group_starts = torch.nonzero(start_mask, as_tuple=False).squeeze(-1)
        rank = torch.arange(order2.numel(), device=device) - group_starts[group_id]
        slot_ordered = torch.where(rank < budget, rank, torch.full_like(rank, budget))

        slots_for_unique = torch.full((unique_keys.numel(),), budget, dtype=torch.long, device=device)
        slots_for_unique[order2] = slot_ordered

        inv = torch.searchsorted(unique_keys, key)
        slots = slots_for_unique[inv]
        return slots

    def forward(self, batch, num_neighbors: int = 50) -> torch.Tensor:
        """Return logits over all entities as candidate tails."""

        device = self.entity_emb.weight.device
        src = batch.src_idx.to(device)
        rel_q = batch.rel_idx.to(device)
        t_q = batch.ts.to(device)

        B = src.shape[0]
        K = int(num_neighbors)

        # 1) RTNS sampling from NeighborFinder
        ngh_ent_np, ngh_rel_np, ngh_ts_np = self.nf.get_temporal_neighbor(
            obj_idx_l=src.detach().cpu().numpy().tolist(),
            ts_l=t_q.detach().cpu().numpy().tolist(),
            num_neighbors=K,
            rel_q_l=rel_q.detach().cpu().numpy().tolist(),
        )
        ngh_ent = torch.from_numpy(ngh_ent_np).to(device=device, dtype=torch.long)  # (B,K), -1 padded
        ngh_rel = torch.from_numpy(ngh_rel_np).to(device=device, dtype=torch.long)
        ngh_ts = torch.from_numpy(ngh_ts_np).to(device=device, dtype=torch.long)

        # 2) Filter to the temporal window and valid pads
        m = self.window_past
        n = self.window_future
        dt = ngh_ts - t_q.view(B, 1)  # (B,K)
        valid = (ngh_ent >= 0) & (ngh_rel >= 0) & (dt >= -m) & (dt <= n)

        if valid.sum().item() == 0:
            # No context available: still use the same final fusion form
            # (CNN/AGG are zeros; static entity term stays meaningful).
            rel_emb_q = self.relation_emb(rel_q)
            z_cnn = torch.zeros((B, self.c_out), device=device, dtype=rel_emb_q.dtype)
            z_agg = torch.zeros((B, self.hr_c), device=device, dtype=rel_emb_q.dtype)
            h_s_static = self._get_static_entity_emb(src).to(dtype=rel_emb_q.dtype)
            fuse_in = torch.cat([z_cnn, z_agg, h_s_static], dim=-1)
            h_s = self.fuse_act(self.fuse(self.dropout(fuse_in)))
            return self._score_all(h_s, rel_emb_q)

        b_idx, k_idx = torch.nonzero(valid, as_tuple=True)
        e_f = ngh_ent[b_idx, k_idx]  # (N,)
        r_f = ngh_rel[b_idx, k_idx]
        dt_f = dt[b_idx, k_idx].to(torch.long)

        # 3) Time binning (\Delta = bin_width)
        T = int(math.ceil((m + n + 1) / self.bin_width))
        tau_f = ((dt_f + m) // self.bin_width).clamp(min=0, max=T - 1)  # (N,)
        tau_q = int((0 + m) // self.bin_width)

        # 4) Slice weights \sigma[\tau] (Eq. 2)  (counts-aware)
        counts = torch.zeros((B, T), device=device, dtype=torch.float32)
        ones = torch.ones_like(tau_f, dtype=torch.float32)
        counts.index_put_((b_idx, tau_f), ones, accumulate=True)

        gamma = F.softplus(self.gamma_raw[rel_q]).to(dtype=torch.float32)  # (B,)

        dist = torch.arange(T, device=device, dtype=torch.float32)
        dist = torch.abs(dist - float(tau_q))
        dist_beta = torch.pow(dist, self.beta)

        # (B,T)
        unnorm = torch.exp(-gamma.view(B, 1) * dist_beta.view(1, T)) / counts.clamp_min(1.0)
        if self.sigma_mask_empty:
            # Eq.(2) is undefined when |G(τ)|=0; we treat empty slices as having zero weight
            unnorm = unnorm * (counts > 0).to(dtype=unnorm.dtype)
        sigma = unnorm / unnorm.sum(dim=1, keepdim=True).clamp_min(1e-9)

        # per-event weight
        w_f = sigma[b_idx, tau_f]  # (N,)

        # 5) Build compact (pair,time) tensor seq (P, C_in, T)
        #    We assign per-batch compact slots for entity and relation IDs.
        max_E = max(1, self.max_entities)
        max_R = max(1, self.max_relations)

        # slots in [0..max_E-1], last bucket is "others" if exceeded
        e_slot = self._assign_slots_firstk(b_idx, e_f, budget=max_E - 1, id_bound=self.num_entities)
        r_slot = self._assign_slots_firstk(b_idx, r_f, budget=max_R - 1, id_bound=self.num_relations)

        # snapshot feature h_{e,r} = proj([E_ent[e] || E_rel[r]]) with embed_dim slice only
        e_base = self.entity_emb(e_f)[..., : self.embed_dim]
        r_base = self.relation_emb(r_f)[..., : self.embed_dim]
        h_er = self.hr_proj(torch.cat([e_base, r_base], dim=-1))  # (N, hr_c)

        # apply slice weight
        h_er_w = h_er * w_f.to(h_er.dtype).unsqueeze(-1)

        # pair ids
        pair_key = (b_idx * (max_E * max_R) + e_slot * max_R + r_slot).to(torch.long)  # (N,)
        uniq_pairs, pair_idx = torch.unique(pair_key, sorted=False, return_inverse=True)
        P = int(uniq_pairs.numel())
        if P == 0:
            h_s = self.entity_emb(src)
            h_r = self.relation_emb(rel_q)
            return self._score_all(h_s, h_r)

        pair_b = (uniq_pairs // (max_E * max_R)).to(torch.long)  # (P,)

        # seq init
        seq = torch.zeros((P, self.hr_c, T), device=device, dtype=h_er_w.dtype)
        mask_t = torch.zeros((P, 1, T), device=device, dtype=torch.bool)

        # scatter-add into (P*T, C)
        flat = (pair_idx * T + tau_f).to(torch.long)  # (N,)
        seq2d = seq.permute(0, 2, 1).contiguous().view(P * T, self.hr_c)
        idx2d = flat.view(-1, 1).expand(-1, self.hr_c)
        seq2d.scatter_add_(0, idx2d, h_er_w)
        seq = seq2d.view(P, T, self.hr_c).permute(0, 2, 1).contiguous()  # (P,C,T)

        mask_flat = mask_t.view(-1)
        mask_flat.index_put_((flat,), torch.ones_like(flat, dtype=torch.bool, device=device), accumulate=True)
        mask_t = mask_flat.view(P, 1, T)

        # 6) Slice-weighted interaction aggregation z_Agg (Eq. 8)
        # mean per (b, tau)
        sum_bt = torch.zeros((B * T, self.hr_c), device=device, dtype=h_er.dtype)
        bt = (b_idx * T + tau_f).to(torch.long)
        sum_bt.index_put_((bt,), h_er, accumulate=True)
        mean_bt = sum_bt / counts.view(B * T, 1).clamp_min(1.0)
        z_agg = (mean_bt.view(B, T, self.hr_c) * sigma.to(mean_bt.dtype).unsqueeze(-1)).sum(dim=1)  # (B, hr_c)

        # 7) Relation-conditioned ST-CNN z_CNN (Eq. 7)
        rel_emb_q = self.relation_emb(rel_q)  # (B, ent_dim)
        gate_b = self.gate(rel_emb_q)  # (B, M_scales)
        gate_p = gate_b[pair_b]        # (P, M_scales)

        y, mask_y = self.stcnn(seq, mask_t, gate_p)  # y:(P,c_out,T)

        # aggregate over (pair,time) to get batch-level z_cnn
        mask_y_f = mask_y.to(dtype=y.dtype)
        y_sum = (y * mask_y_f).sum(dim=2)  # (P, c_out)
        den = mask_y_f.sum(dim=2).clamp_min(1.0)  # (P,1)

        sum_b = torch.zeros((B, self.c_out), device=device, dtype=y_sum.dtype)
        den_b = torch.zeros((B, 1), device=device, dtype=den.dtype)
        sum_b.index_add_(0, pair_b, y_sum)
        den_b.index_add_(0, pair_b, den)
        z_cnn = sum_b / den_b.clamp_min(1.0)

        # 8) Final query subject representation (Eq. 9)
        # NOTE:
        #   - entity_emb: trainable table (updated)
        #   - entity_emb_static (optional): frozen pretrained table for the 'static' concat term
        h_s_static = self._get_static_entity_emb(src)  # (B, ent_dim)
        fuse_in = torch.cat([z_cnn, z_agg.to(z_cnn.dtype), h_s_static.to(z_cnn.dtype)], dim=-1)
        h_s = self.fuse_act(self.fuse(self.dropout(fuse_in)))

        # 9) Score all candidate tails (Eq. 10)
        h_r = rel_emb_q
        scores = self._score_all(h_s, h_r)  # (B, num_entities)
        return scores
