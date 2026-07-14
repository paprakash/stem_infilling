#!/usr/bin/env python3
"""Diagnostic 2: column precision/recall/RMSE restricted to UNDAMAGED regions.

If these come out ~1.0/~1.0/sub-px for both models at every level, the KL gap,
the do-no-harm PSNR gap, and the FFT-spectrum gap are all a single smoothing
phenomenon (texture, not structure). Writes
results/phase1/diagnostics/undamaged_column_metrics.csv and prints the table.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import LEVELS, load_pair, val_subset_structures
from stem_metrics import damage_mask, match_column_positions

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
CACHE = os.path.join(ROOT, "data", "cache_phase1_preds")
OUT = os.path.join(ROOT, "results", "phase1", "diagnostics")
MODELS = ["nafnet", "restormer"]


def restricted(target, pred, mask):
    """P/R/RMSE using only columns whose center lies in an undamaged pixel."""
    mm = match_column_positions(target, pred)
    ct, cp = mm["ct"], mm["cp"]

    def undamaged(pts):
        if not len(pts):
            return np.zeros(0, dtype=bool)
        return ~mask[np.clip(pts[:, 0].astype(int), 0, mask.shape[0] - 1),
                     np.clip(pts[:, 1].astype(int), 0, mask.shape[1] - 1)]

    keep_t, keep_p = undamaged(ct), undamaged(cp)
    n_t, n_p = int(keep_t.sum()), int(keep_p.sum())
    if n_t == 0 or n_p == 0:
        return dict(u_precision=np.nan, u_recall=np.nan, u_rmse=np.nan, n_t=n_t, n_p=n_p)
    # precision counts a match if the PREDICTED column is undamaged; recall if the
    # TARGET column is (a pair straddling the mask edge must not deflate either)
    m_p = sum(1 for _, pi, _ in mm["matches"] if keep_p[pi])
    m_t = sum(1 for ti, _, _ in mm["matches"] if keep_t[ti])
    errs = [d for ti, pi, d in mm["matches"] if keep_t[ti] and keep_p[pi]]
    return dict(u_precision=m_p / n_p, u_recall=m_t / n_t,
                u_rmse=float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan,
                n_t=n_t, n_p=n_p)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for model in MODELS:
        for s_name in val_subset_structures():
            for lv in LEVELS:
                s, t, _, _ = load_pair(s_name, "A", lv)
                pred = np.load(os.path.join(CACHE, model, f"{s_name.replace('.png', '')}_lvl{lv}.npy"))
                r = restricted(t, pred, damage_mask(s, t))
                r.update(model=model, structure=s_name, level=lv)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "undamaged_column_metrics.csv"), index=False)
    agg = df.groupby(["model", "level"])[["u_precision", "u_recall", "u_rmse"]].median().round(4)
    print(agg.to_string())


if __name__ == "__main__":
    main()
