#!/usr/bin/env python3
"""Blind hysteresis gate: strict-seeded loose components (training-free).

Editable mask = connected components of the LOOSE mask (prob > growth_thr)
that contain at least one STRICT seed (prob > 0.9, component >= seed_min px).
Per-region behavior: clean images converge to identity even when the loose
detector over-fires (no seeds -> nothing edited); real damage regions open
fully (a single confident core unlocks the whole loose region). Feathered
compositing. BLIND: no level information used.

Sweep: seed_min {100, 400, 800} x growth_thr {0.5, 0.7}.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gated_sweep import _load_dm, feather_alpha, load_net, seg_prob
from stem_data import build_index, load_pair, load_split, material_family, val_subset_structures
from stem_metrics import all_metrics
from train_stem import infer_full

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SETTINGS = [(0.5, 100), (0.5, 400), (0.5, 800), (0.7, 100), (0.7, 400), (0.7, 800)]
STRICT_THR = 0.9


def hysteresis_mask(prob, growth_thr, seed_min):
    seeds = prob > STRICT_THR
    lab_s, n_s = cc_label(seeds)
    if n_s:
        sizes = np.bincount(lab_s.ravel()); sizes[0] = 0
        seeds = np.isin(lab_s, np.nonzero(sizes >= seed_min)[0])
    if not seeds.any():
        return np.zeros(prob.shape, dtype=bool)
    loose = prob > growth_thr
    lab_l, n_l = cc_label(loose)
    keep = np.unique(lab_l[seeds & loose])
    keep = keep[keep > 0]
    m = np.isin(lab_l, keep)
    return binary_dilation(m, iterations=2)


def one_job(job):
    structure, op, lv, s, t, prob, pred, gt = job
    out = []
    dm = _load_dm(structure, op)
    dark_sites = ((dm == 1) | (dm == 4)) & ~gt if dm is not None else None
    for growth_thr, seed_min in SETTINGS:
        mask = hysteresis_mask(prob, growth_thr, seed_min)
        alpha = feather_alpha(mask)
        gated = alpha * pred + (1.0 - alpha) * s
        m = all_metrics(s, t, gated, with_columns=True, defect_mask=dm)
        tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
        m["seg_precision"] = tp / max(tp + fp, 1)
        m["seg_recall"] = tp / max(tp + fn, 1)
        m["seg_fp_at_defects"] = (float(mask[dark_sites].mean())
                                  if dark_sites is not None and dark_sites.sum() >= 10 else np.nan)
        missed = gt & ~mask
        m["psnr_missed_dmg"] = (10 * np.log10(1.0 / max(float(np.mean((gated[missed] - t[missed]) ** 2)), 1e-12))
                                if missed.sum() >= 10 else np.nan)
        m["frac_edited"] = float(mask.mean())
        m.update(structure=structure, family=material_family(structure), op=op, level=lv,
                 growth_thr=growth_thr, seed_min=seed_min)
        out.append(m)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    seg = load_net(os.path.join(ROOT, "runs", "damage_seg_v2", "config.yaml"),
                   os.path.join(ROOT, "runs", "damage_seg_v2", "latest.pth"))
    res = load_net(os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "config.yaml"),
                   os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "iter_25000.pth"))
    pairs = build_index(load_split("val"))
    subset = set(val_subset_structures())
    out_dir = os.path.join(ROOT, "results", "phase3")
    cache_dirs = {}
    for g, sm in SETTINGS:
        d = os.path.join(ROOT, "data", "cache_phase1_preds", f"hyst_g{g}_s{sm}")
        os.makedirs(d, exist_ok=True)
        cache_dirs[(g, sm)] = d

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (structure, op, lv) in enumerate(pairs):
            s01, t01, mean, std = load_pair(structure, op, lv)
            prob = seg_prob(seg, s01, mean, std)
            pred = infer_full(res, s01, mean, std)
            gt = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2)
            if structure in subset and op == "A":
                for (g, sm), d in cache_dirs.items():
                    mask = hysteresis_mask(prob, g, sm)
                    a = feather_alpha(mask)
                    np.save(os.path.join(d, f"{structure.replace('.png', '')}_lvl{lv}.npy"),
                            (a * pred + (1.0 - a) * s01).astype(np.float32))
            futures.append(ex.submit(one_job, (structure, op, lv, s01, t01, prob, pred, gt)))
            if (i + 1) % 300 == 0:
                print(f"{i + 1}/{len(pairs)}", flush=True)
        for f in futures:
            rows.extend(f.result())

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "hysteresis_sweep_per_image.csv"), index=False)
    summary = (df.groupby(["growth_thr", "seed_min", "level"])
               .agg(psnr=("psnr", "median"), recall=("col_recall", "median"),
                    psnr_dmg=("psnr_dmg", "median"), def_pres=("defect_preserved_frac", "mean"),
                    vac_ph=("vacuum_phantom_frac", "mean"), seg_fp_def=("seg_fp_at_defects", "mean"),
                    psnr_missed=("psnr_missed_dmg", "median"), edited=("frac_edited", "median")).round(4))
    summary.to_csv(os.path.join(out_dir, "hysteresis_sweep_summary.csv"))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
