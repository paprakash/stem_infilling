#!/usr/bin/env python3
"""Phase-3 gate sweep: threshold x dilation grid, single inference pass.

Per image: segmenter probability map and restorer prediction are computed ONCE;
each (thr, dil) setting derives its mask, gated output, and full metric set
(incl. defect preservation, segmenter P/R, miss-rate cost, edited fraction).
Val-subset gated outputs are cached per setting for the hallucination census.

The gate's dial: mask PRECISION protects the zero-invention guarantee, mask
RECALL protects restoration of true damage.

Usage: eval_gated_sweep.py [--split val] [--workers 12]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import build_index, load_pair, load_split, material_family, val_subset_structures
from stem_metrics import all_metrics
from train_stem import build_model, infer_full, pad_to_multiple

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SEG_CFG = os.path.join(ROOT, "runs", "damage_seg_v1", "config.yaml")
SEG_CKPT = os.path.join(ROOT, "runs", "damage_seg_v1", "latest.pth")
RES_CFG = os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "config.yaml")
RES_CKPT = os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "iter_25000.pth")
# (threshold, dilation, min_component_px) — min_size kills speckle false-fires
# (true damage is compact-to-extended; sub-400 px components at these
# magnifications are noise). Round 2 of the sweep; round 1 was min_size 0.
SETTINGS = [(0.7, 2, 0), (0.5, 2, 400), (0.7, 2, 400), (0.9, 2, 400), (0.95, 2, 400), (0.9, 2, 0)]


def load_net(cfg_path, ckpt_path):
    cfg = yaml.safe_load(open(cfg_path))
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    m = build_model(cfg["model"]).cuda()
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def seg_prob(seg, s01, mean, std):
    x = torch.from_numpy((s01 - mean) / std)[None, None].cuda()
    x, h, w = pad_to_multiple(x, 16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = seg(x)
    return torch.sigmoid(logit.float())[0, 0, :h, :w].cpu().numpy()


def _load_dm(structure, op):
    p = os.path.join(ROOT, "data", "defect_masks", f"operation_{op}", structure.replace(".png", ".npz"))
    return np.load(p)["mask"] if os.path.exists(p) else None


FEATHER_PX = 6


def make_mask(prob, thr, dil, min_size):
    m = prob > thr
    if min_size > 0:
        from scipy.ndimage import label as cc_label
        lab, n = cc_label(m)
        if n:
            sizes = np.bincount(lab.ravel())
            m = np.isin(lab, np.nonzero(sizes >= min_size)[0][1:] if sizes[0] >= min_size
                        else np.nonzero(sizes >= min_size)[0])
            m[lab == 0] = False
    return binary_dilation(m, iterations=dil)


def feather_alpha(mask, feather=FEATHER_PX):
    """Linear alpha ramp INSIDE the mask edge: exactly 0 outside (verbatim-source
    guarantee preserved), 1 in the mask interior. Kills paste-seam artifacts."""
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    return np.clip(distance_transform_edt(mask) / feather, 0.0, 1.0).astype(np.float32)


def one_metric_job(job):
    structure, op, lv, s, t, prob, pred, gt = job
    out = []
    dm = _load_dm(structure, op)
    dark_sites = ((dm == 1) | (dm == 4)) & ~gt if dm is not None else None
    for thr, dil, min_size in SETTINGS:
        mask = make_mask(prob, thr, dil, min_size)
        alpha = feather_alpha(mask)
        gated = alpha * pred + (1.0 - alpha) * s
        m = all_metrics(s, t, gated, with_columns=True, defect_mask=dm)
        tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
        m["seg_precision"] = tp / max(tp + fp, 1)
        m["seg_recall"] = tp / max(tp + fn, 1)
        # the guarantee metric: FP rate at dark defect sites
        m["seg_fp_at_defects"] = (float(mask[dark_sites].mean())
                                  if dark_sites is not None and dark_sites.sum() >= 10 else np.nan)
        missed = gt & ~mask
        m["psnr_missed_dmg"] = (10 * np.log10(1.0 / max(float(np.mean((gated[missed] - t[missed]) ** 2)), 1e-12))
                                if missed.sum() >= 10 else np.nan)
        m["frac_edited"] = float(mask.mean())
        m.update(structure=structure, family=material_family(structure), op=op, level=lv,
                 thr=thr, dil=dil, ms=min_size)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seg-run", default="damage_seg_v1", help="segmenter run dir name")
    ap.add_argument("--prefix", default="gated", help="cache/output name prefix")
    args = ap.parse_args()

    seg = load_net(os.path.join(ROOT, "runs", args.seg_run, "config.yaml"),
                   os.path.join(ROOT, "runs", args.seg_run, "latest.pth"))
    res = load_net(RES_CFG, RES_CKPT)
    pairs = build_index(load_split(args.split))
    subset = set(val_subset_structures())
    out_dir = os.path.join(ROOT, "results", "phase3")
    os.makedirs(out_dir, exist_ok=True)
    cache_dirs = {}
    for thr, dil, ms in SETTINGS:
        d = os.path.join(ROOT, "data", "cache_phase1_preds", f"{args.prefix}_t{thr}_d{dil}_s{ms}")
        os.makedirs(d, exist_ok=True)
        cache_dirs[(thr, dil, ms)] = d

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (structure, op, lv) in enumerate(pairs):
            s01, t01, mean, std = load_pair(structure, op, lv)
            prob = seg_prob(seg, s01, mean, std)
            pred = infer_full(res, s01, mean, std)
            gt = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2)
            if structure in subset and op == "A":
                for (thr, dil, ms), d in cache_dirs.items():
                    mask = make_mask(prob, thr, dil, ms)
                    a = feather_alpha(mask)
                    np.save(os.path.join(d, f"{structure.replace('.png', '')}_lvl{lv}.npy"),
                            (a * pred + (1.0 - a) * s01).astype(np.float32))
            futures.append(ex.submit(one_metric_job, (structure, op, lv, s01, t01, prob, pred, gt)))
            if (i + 1) % 200 == 0:
                print(f"inferred {i + 1}/{len(pairs)}", flush=True)
        for f in futures:
            rows.extend(f.result())

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{args.prefix}_sweep_{args.split}_per_image.csv"), index=False)
    summary = (df.groupby(["thr", "dil", "ms", "level"])
               .agg(psnr=("psnr", "median"), recall=("col_recall", "median"),
                    psnr_dmg=("psnr_dmg", "median"), def_pres=("defect_preserved_frac", "mean"),
                    vac_ph=("vacuum_phantom_frac", "mean"), seg_p=("seg_precision", "median"),
                    seg_r=("seg_recall", "median"), seg_fp_def=("seg_fp_at_defects", "mean"),
                    psnr_missed=("psnr_missed_dmg", "median"),
                    edited=("frac_edited", "median")).round(4))
    summary.to_csv(os.path.join(out_dir, f"{args.prefix}_sweep_summary.csv"))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
