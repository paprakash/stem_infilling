#!/usr/bin/env python3
"""Diagnostic 1: hallucination audit of false-positive column detections.

For each model, walk the cached val-subset predictions, collect unmatched
predicted columns (FPs), and classify each:
  invented_damaged    FP inside the damage mask, no true column within 1.5*tol
                      -> model invented an atom where damage erased/never had one
  invented_undamaged  FP in an undamaged region, no true column within 1.5*tol
                      -> most serious class: hallucination with no damage excuse
  borderline_shifted  a true column exists within 1.5*tol (detection shifted or
                      double-fired) -> metric artifact more than physics problem
  boundary_edge       FP center within 16 px of image border or 6 px of a
                      damage-mask boundary -> blob-detector edge artifact

Outputs to results/phase1/diagnostics/:
  hallucination_counts.csv           per model x level x class
  hallucination_cases.csv            every FP with class, position, blob stats
  contact_<model>.png                contact sheet: rows of src|pred|tgt crops
                                     (FP at crop center), grouped by class
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import LEVELS, load_pair, val_subset_structures
from stem_metrics import damage_mask, match_column_positions

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
CACHE = os.path.join(ROOT, "data", "cache_phase1_preds")
OUT = os.path.join(ROOT, "results", "phase1", "diagnostics")
MODELS = ["nafnet", "restormer"]
PATCH = 64
MAX_PER_LEVEL = 6  # spread cases across levels
TARGET_CASES = 30


def classify(fp_yx, ct, tol, mask, shape):
    y, x = fp_yx
    h, w = shape
    if y < 16 or x < 16 or y > h - 17 or x > w - 17:
        return "boundary_edge"
    if len(ct):
        d = cKDTree(ct).query([fp_yx], k=1)[0][0]
        if d <= 1.5 * tol:
            return "borderline_shifted"
    iy, ix = int(round(y)), int(round(x))
    on_boundary = mask[iy, ix] != binary_erosion(mask, iterations=6)[iy, ix] or \
        (not mask[iy, ix] and binary_dilation(mask, iterations=6)[iy, ix])
    if on_boundary:
        return "boundary_edge"
    return "invented_damaged" if mask[iy, ix] else "invented_undamaged"


def crop(a, y, x, half=PATCH // 2):
    h, w = a.shape
    y, x = int(round(y)), int(round(x))
    y = min(max(y, half), h - half)
    x = min(max(x, half), w - half)
    return a[y - half:y + half, x - half:x + half]


def contact_sheet(cases, path, per_row=5):
    """cases: list of (src, pred, tgt, label) crops. Triplet tiles + class label."""
    tile_w, tile_h, pad, cap = PATCH * 3 + 4, PATCH, 6, 14
    rows = (len(cases) + per_row - 1) // per_row
    sheet = Image.new("L", (per_row * (tile_w + pad) + pad, rows * (tile_h + cap + pad) + pad), 40)
    draw = ImageDraw.Draw(sheet)
    for i, (s, p, t, label) in enumerate(cases):
        r, c = divmod(i, per_row)
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + cap + pad)
        tri = np.concatenate([s, np.full((PATCH, 2), 1.0), p, np.full((PATCH, 2), 1.0), t], axis=1)
        sheet.paste(Image.fromarray((np.clip(tri, 0, 1) * 255).astype(np.uint8)), (x0, y0 + cap))
        draw.text((x0, y0), label, fill=255)
    sheet.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    structures = val_subset_structures()
    all_rows = []
    # per-class caps for the contact sheet; stats are ALWAYS collected over every
    # structure x level (a previous version capped stats too — biased to vo2)
    sheet_caps = {"invented_undamaged": 999, "invented_damaged": 20,
                  "borderline_shifted": 10, "boundary_edge": 15}
    for model in MODELS:
        cases = []
        n_sheet = {c: 0 for c in sheet_caps}
        n_struct_lv = {}
        for lv in LEVELS:
            for s_name in structures:
                pred_f = os.path.join(CACHE, model, f"{s_name.replace('.png', '')}_lvl{lv}.npy")
                s, t, _, _ = load_pair(s_name, "A", lv)
                pred = np.load(pred_f)
                mm = match_column_positions(t, pred)
                if not len(mm["fp_idx"]):
                    continue
                mask = damage_mask(s, t)
                for pi in mm["fp_idx"]:
                    yx = mm["cp"][pi]
                    cls = classify(yx, mm["ct"], mm["tol"], mask, t.shape)
                    all_rows.append(dict(model=model, structure=s_name, level=lv,
                                         y=yx[0], x=yx[1], cls=cls,
                                         n_true_cols=len(mm["ct"]),
                                         frac_damaged=float(mask.mean()),
                                         pred_val=float(pred[int(yx[0]), int(yx[1])]),
                                         tgt_val=float(t[int(yx[0]), int(yx[1])])))
                    key = (s_name, lv, cls)
                    if n_sheet[cls] < sheet_caps[cls] and n_struct_lv.get(key, 0) < 1:
                        cases.append((crop(s, *yx), crop(pred, *yx), crop(t, *yx),
                                      f"{cls[:14]} L{lv} {s_name[:14]}"))
                        n_sheet[cls] += 1
                        n_struct_lv[key] = 1
        order = {"invented_undamaged": 0, "invented_damaged": 1, "borderline_shifted": 2, "boundary_edge": 3}
        cases.sort(key=lambda c: order.get(c[3].split(" ")[0][:14], 9))
        contact_sheet(cases, os.path.join(OUT, f"contact_{model}.png"))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "hallucination_cases.csv"), index=False)
    counts = df.pivot_table(index=["model", "level"], columns="cls", aggfunc="size", fill_value=0)
    counts.to_csv(os.path.join(OUT, "hallucination_counts.csv"))
    print(counts.to_string())
    print()
    print(df.groupby(["model", "cls"]).size().unstack(fill_value=0).to_string())
    print(f"\ntotal FP instances: {len(df)} over {len(structures)}x{len(LEVELS)} images/model")


if __name__ == "__main__":
    main()
