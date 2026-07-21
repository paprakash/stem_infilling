#!/usr/bin/env python3
"""Production inference: STEM beam-damage restoration with blind hysteresis gating.

Pipeline (MODEL_CARD.md deployment section):
  1. segmenter v2 -> damage probability map (source only, no dose input)
  2. hysteresis mask: loose components (p > 0.5) containing a strict seed
     (p > 0.9, >= 400 px), dilated 2, feathered 6 px
  3. output = alpha * restorer(source) + (1 - alpha) * source
     (verbatim source outside the mask)
  4. optional --noise: source-spectrum matched noise inside edited regions
  5. review flag: largest UNSEEDED loose component > --flag-px (default 20000)
     -> "diffuse damage, review manually". Partial tail mitigation only —
     measured on val: 3.4% review rate, catches 1/5 known tail cases at the
     default; 10000 -> 9.9% rate, 2/5. Undetected tail failures are
     CONSERVATIVE (source kept, nothing invented).

Usage: stem_restore.py INPUT.png [INPUT2.png ...] --out DIR
       [--noise] [--flag-px 20000] [--models DIR]
Writes <name>_restored.png, <name>_mask.png, <name>_meta.json per input.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.ndimage import label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stem import build_model, pad_to_multiple

DEF_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models_release")
GROWTH_THR, STRICT_THR, SEED_MIN, DILATE, FEATHER = 0.5, 0.9, 400, 2, 6


def load_net(models_dir, name):
    cfg = yaml.safe_load(open(os.path.join(models_dir, f"{name}_config.yaml")))
    m = build_model(cfg["model"]).cuda()
    sd = torch.load(os.path.join(models_dir, f"{name}.pth"), map_location="cuda", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def forward(model, s01, mean, std, sigmoid=False):
    x = torch.from_numpy((s01 - mean) / std)[None, None].cuda()
    x, h, w = pad_to_multiple(x, 16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
    y = y.float()[0, 0, :h, :w].cpu().numpy()
    return 1 / (1 + np.exp(-y)) if sigmoid else np.clip(y * std + mean, 0.0, 1.0)


def hysteresis(prob):
    seeds = prob > STRICT_THR
    lab_s, _ = cc_label(seeds)
    sz = np.bincount(lab_s.ravel()); sz[0] = 0
    good = np.isin(lab_s, np.nonzero(sz >= SEED_MIN)[0])
    loose = prob > GROWTH_THR
    lab_l, n = cc_label(loose)
    mask = np.zeros(prob.shape, dtype=bool)
    unseeded_max = 0
    if n:
        keep = np.unique(lab_l[good & loose])
        keep = keep[keep > 0]
        mask = np.isin(lab_l, keep)
        for i in range(1, n + 1):
            if i not in keep:
                unseeded_max = max(unseeded_max, int((lab_l == i).sum()))
    return binary_dilation(mask, iterations=DILATE), unseeded_max, bool(good.any())


def matched_noise(s01, rng):
    n = s01 - gaussian_filter(s01, 1.5)
    h, w = n.shape
    amp = np.abs(np.fft.fft2(n))
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(np.minimum(yy, h - yy), np.minimum(xx, w - xx)).astype(np.int32)
    prof = np.bincount(r.ravel(), weights=amp.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    out = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * prof[r]))
    return out * n.std() / max(out.std(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=DEF_MODELS)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--flag-px", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seg = load_net(args.models, "segmenter_v2")
    res = load_net(args.models, "restorer_ft_evid_asym1")
    rng = np.random.default_rng(0)

    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        k = 2
        while os.path.exists(os.path.join(args.out, f"{name}_restored.png")):
            name = f"{os.path.splitext(os.path.basename(path))[0]}_{k}"
            k += 1
        s01 = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        mean, std = float(s01.mean()), max(float(s01.std()), 1e-6)
        prob = forward(seg, s01, mean, std, sigmoid=True)
        mask, unseeded_max, has_seed = hysteresis(prob)
        alpha = (np.clip(distance_transform_edt(mask) / FEATHER, 0, 1).astype(np.float32)
                 if mask.any() else np.zeros_like(s01))
        pred = forward(res, s01, mean, std)
        outp = alpha * pred + (1 - alpha) * s01
        if args.noise and mask.any():
            outp = np.clip(outp + alpha * matched_noise(s01, rng), 0, 1)
        flag = unseeded_max > args.flag_px
        Image.fromarray((outp * 255).astype(np.uint8)).save(os.path.join(args.out, f"{name}_restored.png"))
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, f"{name}_mask.png"))
        meta = dict(input=os.path.abspath(path), edited_frac=float(mask.mean()),
                    has_strict_seed=has_seed, largest_unseeded_loose_px=unseeded_max,
                    review_flag=bool(flag),
                    review_reason=("diffuse damage, review manually" if flag else None),
                    gate=dict(growth=GROWTH_THR, strict=STRICT_THR, seed_min=SEED_MIN,
                              dilate=DILATE, feather=FEATHER),
                    noise_injected=bool(args.noise))
        with open(os.path.join(args.out, f"{name}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"{name}: edited {mask.mean():.1%}"
              + (" [REVIEW FLAG: diffuse damage]" if flag else ""), flush=True)


if __name__ == "__main__":
    main()
