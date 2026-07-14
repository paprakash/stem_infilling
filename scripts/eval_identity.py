#!/usr/bin/env python3
"""Identity baseline with the FULL Phase-1 metric set (pred = raw source).

Phase 0 computed PSNR/SSIM/KL only; the Phase-1 report compares models against
identity on every committed metric, so compute the rest once and cache.

Usage: eval_identity.py --split val --out results/phase1
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import build_index, load_pair, load_split, material_family
from stem_metrics import all_metrics

ROOT = "/blue/hennig/pawanprakash/ornl_stem"


def one(job):
    structure, op, lv = job
    s, t, _, _ = load_pair(structure, op, lv)
    m = all_metrics(s, t, s, with_columns=True)
    m.update(structure=structure, family=material_family(structure), op=op, level=lv)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase1"))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    pairs = build_index(load_split(args.split))
    print(f"{len(pairs)} pairs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, pairs, chunksize=8))
    df = pd.DataFrame(rows)
    df["exp"] = "identity"
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"identity_{args.split}_per_image.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(df.groupby("level")[["psnr", "ssim", "kl", "fft_err", "col_precision",
                               "col_recall", "col_rmse", "psnr_dmg", "psnr_undmg"]]
          .median().round(4).to_string())


if __name__ == "__main__":
    main()
