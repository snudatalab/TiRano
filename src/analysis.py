# -*- coding: utf-8 -*-
"""
Optional analysis utilities for Tirano experiments.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from utils import read_quadruples_txt, compute_relation2alpha, get_dataset_stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--time_mode", type=str, default="auto", choices=["auto","int","year","ordinal"])
    ap.add_argument("--file_format", type=str, default="auto", choices=["auto","quad","interval"])
    ap.add_argument("--interval_mode", type=str, default="start", choices=["start","end","mid","random","expand"])
    args = ap.parse_args()

    ds_lower = args.dataset.lower()
    interval_datasets = {"wiki", "wikidata12k", "yago", "yago11k"}
    if ds_lower in interval_datasets:
        if args.time_mode == "auto":
            args.time_mode = "year"
        if args.file_format == "auto":
            args.file_format = "interval"
    else:
        if args.file_format == "auto":
            args.file_format = "quad"

    dname = args.dataset.lower()
    dpath = os.path.join(args.data_dir, dname)

    train_path = os.path.join(dpath, "train.txt")
    if not os.path.exists(train_path):
        alt = os.path.join(dpath, f"{dname}_train.txt")
        if os.path.exists(alt):
            train_path = alt

    stat_path = os.path.join(dpath, "stat.txt")
    if not os.path.exists(stat_path):
        alt = os.path.join(dpath, f"{dname}_stat.txt")
        if os.path.exists(alt):
            stat_path = alt


    train = read_quadruples_txt(train_path, time_mode=args.time_mode, file_format=args.file_format, interval_mode=args.interval_mode)

    if os.path.exists(stat_path):
        num_e, num_r_aug, _ = get_dataset_stat(stat_path)
        num_r_base = num_r_aug // 2
    else:
        num_r_base = max(r for _, r, _, _, _ in train) + 1

    # alpha
    alpha = compute_relation2alpha(train, num_rel_base=num_r_base)

    # relation frequency
    freq = np.zeros((num_r_base,), dtype=np.int64)
    for _, r, _, _, _ in train:
        freq[r % num_r_base] += 1

    print(f"Dataset: {args.dataset}")
    print(f"Train facts: {len(train)}")
    print(f"Base relations: {num_r_base}")
    print(f"alpha range: min={alpha.min():.6f} max={alpha.max():.6f} mean={alpha.mean():.6f}")
    print(f"freq range:  min={freq.min()} max={freq.max()} mean={freq.mean():.2f}")

    # show top 10 relations by frequency
    top = np.argsort(-freq)[:10]
    print("Top relations by frequency:")
    for r in top:
        print(f"  r={r:4d}  freq={freq[r]:8d}  alpha={alpha[r]:.6f}")


if __name__ == "__main__":
    main()
