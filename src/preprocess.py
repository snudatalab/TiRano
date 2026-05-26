# -*- coding: utf-8 -*-
"""
Preprocess raw temporal KG dataset into pickles required by Tirano.

Expected dataset directory structure:
  dataset/<dataset_name_lower>/
    train.txt
    valid.txt   (optional)
    test.txt
    stat.txt    (optional but recommended)

Raw file formats supported:
  - 4 columns: s r o t
  - 5 columns: s r o t event_idx  (e.g., GDELT)
  - 5 columns: s r o t_start t_end (interval datasets like WIKI/YAGO)

Outputs (saved under dataset/<dataset>/processed/):
  - train_data.pkl / valid_data.pkl / test_data.pkl  (augmented with inverse relations; tuples (s,r,o,t,event))
  - sr2o.pkl  (time-independent filter dict)
  - srt2o.pkl (time-dependent filter dict)
  - o2srt_train.pkl      (adjacency for training)
  - o2srt_train_val.pkl  (adjacency for validation/testing)
  - relation2alpha.pkl   (RTNS priors α for all relations including inverse)
  - meta.json            (stats)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np

from utils import (
    augment_with_inverse,
    build_filter_dicts,
    build_o2srt_adj,
    compute_relation2alpha,
    get_dataset_stat,
    infer_num_entities_relations,
    read_quadruples_txt,
    save_json,
    save_pickle,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset", help="Root dataset dir")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ICEWS14, gdelt, icews18)")
    p.add_argument("--alpha_min", type=float, default=1e-4)
    p.add_argument("--alpha_max", type=float, default=100.0)
    p.add_argument(
        "--time_mode",
        type=str,
        default="auto",
        choices=["auto", "int", "year", "ordinal"],
        help=(
            "How to parse timestamp tokens in raw txt. "
            "auto: int if possible else extract year; "
            "year: always extract year; "
            "ordinal: YYYY-MM-DD -> date.toordinal() (## treated as 01)."
        ),
    )
    p.add_argument(
        "--file_format",
        type=str,
        default="auto",
        choices=["auto", "quad", "interval"],
        help=("Raw split file format. auto: infer; quad: s r o t [event]; interval: s r o t_start t_end"),
    )
    p.add_argument(
        "--interval_mode",
        type=str,
        default="start",
        choices=["start", "end", "mid", "random", "expand"],
        help=("How to convert (t_start, t_end) to a single timestamp (or expand)."),
    )
    args = p.parse_args()

    # Dataset-aware defaults for interval datasets (WIKI/YAGO style).
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

    # NOTE: interval_mode default is "start"; change to "expand" if you want
    # each (start,end) fact to be replicated across all timestamps in its interval.

    dname = args.dataset.lower()
    dpath = os.path.join(args.data_dir, dname)

    train_path = os.path.join(dpath, "train.txt")
    if not os.path.exists(train_path):
        alt = os.path.join(dpath, f"{dname}_train.txt")
        if os.path.exists(alt):
            train_path = alt


    valid_path = os.path.join(dpath, "valid.txt")
    if not os.path.exists(valid_path):
        alt = os.path.join(dpath, f"{dname}_valid.txt")
        if os.path.exists(alt):
            valid_path = alt


    test_path = os.path.join(dpath, "test.txt")
    if not os.path.exists(test_path):
        alt = os.path.join(dpath, f"{dname}_test.txt")
        if os.path.exists(alt):
            test_path = alt


    stat_path = os.path.join(dpath, "stat.txt")
    if not os.path.exists(stat_path):
        alt = os.path.join(dpath, f"{dname}_stat.txt")
        if os.path.exists(alt):
            stat_path = alt



    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)

    train = read_quadruples_txt(train_path, time_mode=args.time_mode, file_format=args.file_format, interval_mode=args.interval_mode)
    valid = read_quadruples_txt(valid_path, time_mode=args.time_mode, file_format=args.file_format, interval_mode=args.interval_mode) if os.path.exists(valid_path) else []
    test = read_quadruples_txt(test_path, time_mode=args.time_mode, file_format=args.file_format, interval_mode=args.interval_mode)

    if os.path.exists(stat_path):
        num_e, num_r_aug, _ = get_dataset_stat(stat_path)
        num_r_base = num_r_aug // 2
    else:
        num_e, num_r_base = infer_num_entities_relations(train, valid, test)
        num_r_aug = 2 * num_r_base

    # Augment with inverse relations
    train_aug = augment_with_inverse(train, num_rel_base=num_r_base)
    valid_aug = augment_with_inverse(valid, num_rel_base=num_r_base) if valid else []
    test_aug = augment_with_inverse(test, num_rel_base=num_r_base)

    # Filter dicts built from all splits (augmented) for filtered ranking
    all_aug = train_aug + valid_aug + test_aug
    sr2o, srt2o = build_filter_dicts(all_aug)

    # Adjacency lists for neighbor sampling
    adj_train = build_o2srt_adj(train_aug, num_entities=num_e)
    adj_train_val = build_o2srt_adj(train_aug + valid_aug, num_entities=num_e)

    # RTNS alpha priors computed from *original* training (no inverse duplication), then expand
    relation2alpha = compute_relation2alpha(
        train_original=train,
        num_rel_base=num_r_base,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )

    out_dir = os.path.join(dpath, "processed")
    os.makedirs(out_dir, exist_ok=True)

    save_pickle(train_aug, os.path.join(out_dir, "train_data.pkl"))
    save_pickle(valid_aug, os.path.join(out_dir, "valid_data.pkl"))
    save_pickle(test_aug, os.path.join(out_dir, "test_data.pkl"))

    save_pickle(sr2o, os.path.join(out_dir, "sr2o.pkl"))
    save_pickle(srt2o, os.path.join(out_dir, "srt2o.pkl"))

    save_pickle(adj_train, os.path.join(out_dir, "o2srt_train.pkl"))
    save_pickle(adj_train_val, os.path.join(out_dir, "o2srt_train_val.pkl"))

    save_pickle(relation2alpha.astype(np.float32), os.path.join(out_dir, "relation2alpha.pkl"))

    meta = dict(
        dataset=args.dataset,
        time_mode=args.time_mode,
        file_format=args.file_format,
        interval_mode=args.interval_mode,
        num_entities=int(num_e),
        num_relations_base=int(num_r_base),
        num_relations_aug=int(num_r_aug),
        num_train_original=int(len(train)),
        num_train_aug=int(len(train_aug)),
        num_valid_original=int(len(valid)),
        num_valid_aug=int(len(valid_aug)),
        num_test_original=int(len(test)),
        num_test_aug=int(len(test_aug)),
    )
    save_json(meta, os.path.join(out_dir, "meta.json"))
    print("Saved to:", out_dir)
    print(meta)


if __name__ == "__main__":
    main()
