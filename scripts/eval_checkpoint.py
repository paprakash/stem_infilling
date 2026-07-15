#!/usr/bin/env python3
"""Evaluate a checkpoint on the FULL val (or test) split with all committed metrics.

GPU inference is serial; metric computation fans out over worker processes.
Writes <out>/<exp>_iter<It>_per_image.csv. Table/figure generation lives in
make_phase1_report.py.

Usage:
  eval_checkpoint.py --config runs/<exp>/config.yaml [--ckpt runs/<exp>/latest.pth]
                     --split val --out results/phase1
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import build_index, load_pair, load_split, material_family
from stem_metrics import all_metrics
from train_stem import build_model, infer_full

ROOT = "/blue/hennig/pawanprakash/ornl_stem"


def load_defect_mask(structure, op):
    p = os.path.join(ROOT, "data", "defect_masks", f"operation_{op}",
                     structure.replace(".png", ".npz"))
    if os.path.exists(p):
        return np.load(p)["mask"]
    return None


def one_metric_job(job):
    structure, op, lv, s, t, pred = job
    m = all_metrics(s, t, pred, with_columns=True, defect_mask=load_defect_mask(structure, op))
    m.update(structure=structure, family=material_family(structure), op=op, level=lv)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase1"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--levels", type=int, nargs="+", default=None,
                    help="restrict to these damage levels (e.g. 1 36) for fast milestone tracking")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpt_path = args.ckpt or os.path.join(ROOT, "runs", cfg["exp"], "latest.pth")
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = build_model(cfg["model"]).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    it = ck["iter"]
    print(f"{cfg['exp']} @ iter {it}, split={args.split}", flush=True)

    pairs = build_index(load_split(args.split))
    if args.levels:
        pairs = [p for p in pairs if p[2] in set(args.levels)]
    print(f"{len(pairs)} pairs", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (structure, op, lv) in enumerate(pairs):
            s, t, mean, std = load_pair(structure, op, lv)
            pred = infer_full(model, s, mean, std)
            futures.append(ex.submit(one_metric_job, (structure, op, lv, s, t, pred)))
            if (i + 1) % 200 == 0:
                print(f"  inferred {i + 1}/{len(pairs)}", flush=True)
        for i, f in enumerate(futures):
            rows.append(f.result())
            if (i + 1) % 500 == 0:
                print(f"  metrics {i + 1}/{len(pairs)}", flush=True)

    df = pd.DataFrame(rows)
    df["iter"] = it
    df["exp"] = cfg["exp"]
    out_csv = os.path.join(args.out, f"{cfg['exp']}_iter{it}_{args.split}_per_image.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}", flush=True)
    print(df.groupby("level")[["psnr", "ssim", "kl", "fft_err", "col_precision",
                               "col_recall", "col_rmse", "psnr_dmg", "psnr_undmg"]]
          .median().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
