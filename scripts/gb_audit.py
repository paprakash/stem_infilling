#!/usr/bin/env python3
"""Grain-boundary healing audit (runs on cached val-subset predictions).

The operations are built on grain boundaries: most targets carry a vertical
registry-shift seam near the image center (a dark line with lattice mismatch
across it). This audit:
  1. locates the GB line per structure: minimum of the column-median intensity
     profile within the central 30-70% band (validated visually);
  2. renders cross-boundary strips source|pred...|target at levels 1/36 for a
     contact sheet — the eye test for "healing" (lattice made continuous);
  3. counts audited FP cases (borderline_shifted, invented_*) within 1.5*d of
     the GB line vs the area-expected fraction -> clustering ratio.

Outputs to results/phase2/: gb_strips.png, gb_clustering.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import load_pair, val_subset_structures

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
CACHE = os.path.join(ROOT, "data", "cache_phase1_preds")
OUT = os.path.join(ROOT, "results", "phase2")
MODELS = ["nafnet_base100k", "nafnet_ft"]


def locate_gb_x(t):
    """x of the vertical GB seam: min of column-median intensity in the
    central band. Returns (x, depth) where depth is the contrast of the dip."""
    prof = np.median(t, axis=0)
    lo, hi = int(0.30 * len(prof)), int(0.70 * len(prof))
    x = lo + int(np.argmin(prof[lo:hi]))
    depth = float(np.median(prof) - prof[x])
    return x, depth


def main():
    os.makedirs(OUT, exist_ok=True)
    structures = val_subset_structures()

    # --- strips for the contact sheet ---
    strips, labels = [], []
    strip_h, strip_w = 120, 300
    for s_name in structures[:8]:
        for lv in (1, 36):
            s, t, _, _ = load_pair(s_name, "A", lv)
            x_gb, depth = locate_gb_x(t)
            h, w = t.shape
            y0 = h // 2 - strip_h // 2
            x0 = min(max(x_gb - strip_w // 2, 0), w - strip_w)
            sl = np.s_[y0:y0 + strip_h, x0:x0 + strip_w]
            panels = [s[sl]]
            for m in MODELS:
                p = np.load(os.path.join(CACHE, m, f"{s_name.replace('.png', '')}_lvl{lv}.npy"))
                panels.append(p[sl])
            panels.append(t[sl])
            strips.append(np.concatenate(panels, axis=1))
            labels.append(f"{s_name[:26]} L{lv} gb_x={x_gb} depth={depth:.2f}")

    pad, cap = 6, 14
    W = strips[0].shape[1]
    sheet = Image.new("L", (W + 2 * pad, len(strips) * (strip_h + cap + pad) + pad), 40)
    draw = ImageDraw.Draw(sheet)
    for i, (st, lab) in enumerate(zip(strips, labels)):
        y0 = pad + i * (strip_h + cap + pad)
        sheet.paste(Image.fromarray((np.clip(st, 0, 1) * 255).astype(np.uint8)), (pad, y0 + cap))
        draw.text((pad, y0), lab + "   [src | base100k | ft | tgt]", fill=255)
    sheet.save(os.path.join(OUT, "gb_strips.png"))

    # --- FP clustering near the GB ---
    rows = []
    for tag, cases_csv in [("base100k", "results/phase2/diagnostics_placeholder")]:
        pass
    case_files = {
        "base100k": os.path.join(ROOT, "results", "phase1", "diagnostics", "hallucination_cases_base100k.csv"),
        "ft": os.path.join(ROOT, "results", "phase1", "diagnostics", "hallucination_cases_ft.csv"),
    }
    gb_x_of = {}
    d_of = {}
    for s_name in structures:
        _, t, _, _ = load_pair(s_name, "A", 1)
        gb_x_of[s_name], _ = locate_gb_x(t)
        d_of[s_name] = 40.0  # refined below from case tol if available
    for tag, f in case_files.items():
        if not os.path.exists(f):
            print(f"skip {tag}: {f} missing")
            continue
        df = pd.read_csv(f)
        for cls in ["borderline_shifted", "invented_undamaged", "invented_damaged"]:
            sub = df[df.cls == cls]
            if not len(sub):
                continue
            near = 0
            exp_frac = []
            for _, r in sub.iterrows():
                gx = gb_x_of.get(r.structure)
                if gx is None:
                    continue
                _, t, _, _ = load_pair(r.structure, "A", int(r.level))
                band = 60.0  # ~1.5 typical lattice spacings
                if abs(r.x - gx) <= band:
                    near += 1
                exp_frac.append(2 * band / t.shape[1])
            n = len(sub)
            expected = float(np.mean(exp_frac)) if exp_frac else np.nan
            rows.append(dict(model=tag, cls=cls, n=n, near_gb=near,
                             frac_near=near / n, expected_frac=round(expected, 3),
                             clustering_ratio=round((near / n) / expected, 2) if expected else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "gb_clustering.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
