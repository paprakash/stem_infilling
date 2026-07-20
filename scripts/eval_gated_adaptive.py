#!/usr/bin/env python3
"""Adaptive gate: per-image threshold selection by predicted damage extent.

Rule: probe mask at thr 0.5 (+400 px size filter). If its area fraction <
--switch (default 3%), the image is low-damage -> tight gate (thr 0.9, s400,
which reaches identity-level preservation and ~0 low-level inventions);
otherwise keep the permissive gate (thr 0.5, s400, whose level-36 cost is
-0.24 dB). Feathered compositing throughout.

Evaluates full val; caches subset outputs for the census.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gated_sweep import feather_alpha, load_net, make_mask, seg_prob, _load_dm
from stem_data import build_index, load_pair, load_split, material_family, val_subset_structures
from stem_metrics import all_metrics
from train_stem import infer_full

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
PERMISSIVE = (0.5, 2, 400)
TIGHT = (0.9, 2, 400)


def one_job(job):
    structure, op, lv, s, t, gated, mask, gt, regime = job
    m = all_metrics(s, t, gated, with_columns=True, defect_mask=_load_dm(structure, op))
    tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
    m["seg_precision"] = tp / max(tp + fp, 1)
    m["seg_recall"] = tp / max(tp + fn, 1)
    missed = gt & ~mask
    m["psnr_missed_dmg"] = (10 * np.log10(1.0 / max(float(np.mean((gated[missed] - t[missed]) ** 2)), 1e-12))
                            if missed.sum() >= 10 else np.nan)
    m["frac_edited"] = float(mask.mean())
    m["regime"] = regime
    m.update(structure=structure, family=material_family(structure), op=op, level=lv)
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    seg = load_net(os.path.join(ROOT, "runs", "damage_seg_v2", "config.yaml"),
                   os.path.join(ROOT, "runs", "damage_seg_v2", "latest.pth"))
    res = load_net(os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "config.yaml"),
                   os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "iter_25000.pth"))
    pairs = build_index(load_split("val"))
    subset = set(val_subset_structures())
    cache = os.path.join(ROOT, "data", "cache_phase1_preds", "gatedv2_adaptive")
    os.makedirs(cache, exist_ok=True)
    out_dir = os.path.join(ROOT, "results", "phase3")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (structure, op, lv) in enumerate(pairs):
            s01, t01, mean, std = load_pair(structure, op, lv)
            prob = seg_prob(seg, s01, mean, std)
            probe = make_mask(prob, *PERMISSIVE)
            regime = "tight" if probe.mean() < args.switch else "permissive"
            mask = make_mask(prob, *(TIGHT if regime == "tight" else PERMISSIVE))
            pred = infer_full(res, s01, mean, std)
            alpha = feather_alpha(mask)
            gated = alpha * pred + (1.0 - alpha) * s01
            gt = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2)
            if structure in subset and op == "A":
                np.save(os.path.join(cache, f"{structure.replace('.png', '')}_lvl{lv}.npy"),
                        gated.astype(np.float32))
            futures.append(ex.submit(one_job, (structure, op, lv, s01, t01, gated, mask, gt, regime)))
            if (i + 1) % 300 == 0:
                print(f"{i + 1}/{len(pairs)}", flush=True)
        for f in futures:
            rows.append(f.result())

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "gatedv2_adaptive_per_image.csv"), index=False)
    print(df.groupby("level").agg(
        psnr=("psnr", "median"), recall=("col_recall", "median"), psnr_dmg=("psnr_dmg", "median"),
        def_pres=("defect_preserved_frac", "mean"), vac_ph=("vacuum_phantom_frac", "mean"),
        psnr_missed=("psnr_missed_dmg", "median"), edited=("frac_edited", "median"),
        tight_frac=("regime", lambda r: (r == "tight").mean())).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
