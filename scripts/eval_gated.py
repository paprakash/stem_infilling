#!/usr/bin/env python3
"""Phase-3 gated-system evaluation.

Gate: predicted damage mask (segmenter, threshold + conservative dilation) —
source copied VERBATIM outside the mask, restorer prediction inside; optional
matched-noise injection inside edited regions only.

Evaluates the gated system on the full val split with the standard battery
(incl. defect preservation) and reports, per level:
  - all committed metrics of the gated output
  - segmenter pixel precision/recall vs ground-truth damage mask
  - miss-rate cost: PSNR of gated output on gt-damaged pixels the mask MISSED
Also caches gated val-subset predictions for the hallucination audit.

Usage: eval_gated.py [--thr 0.3] [--dilate 4] [--noise]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import binary_dilation, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import build_index, load_pair, load_split, material_family, val_subset_structures
from stem_metrics import all_metrics
from train_stem import build_model, infer_full, pad_to_multiple

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SEG_CFG = os.path.join(ROOT, "runs", "damage_seg_v1", "config.yaml")
SEG_CKPT = os.path.join(ROOT, "runs", "damage_seg_v1", "latest.pth")
RES_CFG = os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "config.yaml")
RES_CKPT = os.path.join(ROOT, "runs", "nafnet_w32_ft_evid_asym1", "iter_25000.pth")


def load_net(cfg_path, ckpt_path):
    cfg = yaml.safe_load(open(cfg_path))
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    m = build_model(cfg["model"]).cuda()
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def seg_mask(seg, s01, mean, std, thr, dil):
    x = torch.from_numpy((s01 - mean) / std)[None, None].cuda()
    x, h, w = pad_to_multiple(x, 16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = seg(x)
    p = torch.sigmoid(logit.float())[0, 0, :h, :w].cpu().numpy()
    return binary_dilation(p > thr, iterations=dil)


def synth_matched_noise(noise, rng):
    h, w = noise.shape
    amp = np.abs(np.fft.fft2(noise))
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(np.minimum(yy, h - yy), np.minimum(xx, w - xx)).astype(np.int32)
    prof = np.bincount(r.ravel(), weights=amp.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    out = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * prof[r]))
    return out * noise.std() / max(out.std(), 1e-12)


def one_metric_job(job):
    structure, op, lv, s, t, gated, mask_pred, gt = job
    m = all_metrics(s, t, gated, with_columns=True, defect_mask=_load_dm(structure, op))
    tp = int((mask_pred & gt).sum()); fp = int((mask_pred & ~gt).sum()); fn = int((~mask_pred & gt).sum())
    m["seg_precision"] = tp / max(tp + fp, 1)
    m["seg_recall"] = tp / max(tp + fn, 1)
    missed = gt & ~mask_pred
    if missed.sum() >= 10:
        mse = float(np.mean((gated[missed] - t[missed]) ** 2))
        m["psnr_missed_dmg"] = 10 * np.log10(1.0 / max(mse, 1e-12))
    else:
        m["psnr_missed_dmg"] = np.nan
    m["frac_edited"] = float(mask_pred.mean())
    m.update(structure=structure, family=material_family(structure), op=op, level=lv)
    return m


def _load_dm(structure, op):
    p = os.path.join(ROOT, "data", "defect_masks", f"operation_{op}", structure.replace(".png", ".npz"))
    return np.load(p)["mask"] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    seg = load_net(SEG_CFG, SEG_CKPT)
    res = load_net(RES_CFG, RES_CKPT)
    rng = np.random.default_rng(20260716)
    tag = f"gated_thr{args.thr}_d{args.dilate}" + ("_noise" if args.noise else "")

    pairs = build_index(load_split(args.split))
    subset = set((s, "A") for s in val_subset_structures())
    cache_dir = os.path.join(ROOT, "data", "cache_phase1_preds", tag)
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = os.path.join(ROOT, "results", "phase3")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (structure, op, lv) in enumerate(pairs):
            s01, t01, mean, std = load_pair(structure, op, lv)
            mask_pred = seg_mask(seg, s01, mean, std, args.thr, args.dilate)
            pred = infer_full(res, s01, mean, std)
            gated = np.where(mask_pred, pred, s01)
            if args.noise and mask_pred.any():
                n = synth_matched_noise(s01 - gaussian_filter(s01, 1.5), rng)
                gated = np.clip(np.where(mask_pred, gated + n, gated), 0, 1)
            gt = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2)
            if (structure, op) in subset:
                np.save(os.path.join(cache_dir, f"{structure.replace('.png', '')}_lvl{lv}.npy"),
                        gated.astype(np.float32))
            futures.append(ex.submit(one_metric_job, (structure, op, lv, s01, t01, gated, mask_pred, gt)))
            if (i + 1) % 200 == 0:
                print(f"inferred {i + 1}/{len(pairs)}", flush=True)
        for f in futures:
            rows.append(f.result())

    df = pd.DataFrame(rows)
    df["exp"] = tag
    out_csv = os.path.join(out_dir, f"{tag}_{args.split}_per_image.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}", flush=True)
    print(df.groupby("level")[["psnr", "col_recall", "psnr_dmg", "defect_preserved_frac",
                               "vacuum_phantom_frac", "seg_precision", "seg_recall",
                               "psnr_missed_dmg", "frac_edited"]]
          .agg({"psnr": "median", "col_recall": "median", "psnr_dmg": "median",
                "defect_preserved_frac": "mean", "vacuum_phantom_frac": "mean",
                "seg_precision": "median", "seg_recall": "median",
                "psnr_missed_dmg": "median", "frac_edited": "median"}).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
