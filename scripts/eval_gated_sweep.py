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
SETTINGS = [(0.3, 2), (0.3, 4), (0.5, 2), (0.5, 4), (0.7, 2), (0.7, 4)]


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


def one_metric_job(job):
    structure, op, lv, s, t, prob, pred, gt = job
    out = []
    dm = _load_dm(structure, op)
    for thr, dil in SETTINGS:
        mask = binary_dilation(prob > thr, iterations=dil)
        gated = np.where(mask, pred, s)
        m = all_metrics(s, t, gated, with_columns=True, defect_mask=dm)
        tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
        m["seg_precision"] = tp / max(tp + fp, 1)
        m["seg_recall"] = tp / max(tp + fn, 1)
        missed = gt & ~mask
        m["psnr_missed_dmg"] = (10 * np.log10(1.0 / max(float(np.mean((gated[missed] - t[missed]) ** 2)), 1e-12))
                                if missed.sum() >= 10 else np.nan)
        m["frac_edited"] = float(mask.mean())
        m.update(structure=structure, family=material_family(structure), op=op, level=lv,
                 thr=thr, dil=dil)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    seg = load_net(SEG_CFG, SEG_CKPT)
    res = load_net(RES_CFG, RES_CKPT)
    pairs = build_index(load_split(args.split))
    subset = set(val_subset_structures())
    out_dir = os.path.join(ROOT, "results", "phase3")
    os.makedirs(out_dir, exist_ok=True)
    cache_dirs = {}
    for thr, dil in SETTINGS:
        d = os.path.join(ROOT, "data", "cache_phase1_preds", f"gated_t{thr}_d{dil}")
        os.makedirs(d, exist_ok=True)
        cache_dirs[(thr, dil)] = d

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (structure, op, lv) in enumerate(pairs):
            s01, t01, mean, std = load_pair(structure, op, lv)
            prob = seg_prob(seg, s01, mean, std)
            pred = infer_full(res, s01, mean, std)
            gt = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2)
            if structure in subset and op == "A":
                for (thr, dil), d in cache_dirs.items():
                    mask = binary_dilation(prob > thr, iterations=dil)
                    np.save(os.path.join(d, f"{structure.replace('.png', '')}_lvl{lv}.npy"),
                            np.where(mask, pred, s01).astype(np.float32))
            futures.append(ex.submit(one_metric_job, (structure, op, lv, s01, t01, prob, pred, gt)))
            if (i + 1) % 200 == 0:
                print(f"inferred {i + 1}/{len(pairs)}", flush=True)
        for f in futures:
            rows.extend(f.result())

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"gate_sweep_{args.split}_per_image.csv"), index=False)
    summary = (df.groupby(["thr", "dil", "level"])
               .agg(psnr=("psnr", "median"), recall=("col_recall", "median"),
                    psnr_dmg=("psnr_dmg", "median"), def_pres=("defect_preserved_frac", "mean"),
                    vac_ph=("vacuum_phantom_frac", "mean"), seg_p=("seg_precision", "median"),
                    seg_r=("seg_recall", "median"), psnr_missed=("psnr_missed_dmg", "median"),
                    edited=("frac_edited", "median")).round(4))
    summary.to_csv(os.path.join(out_dir, "gate_sweep_summary.csv"))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
