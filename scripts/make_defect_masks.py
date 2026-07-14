#!/usr/bin/env python3
"""Precompute per-target defect-site + vacuum masks (TARGET-only, offline).

Method:
  columns   LoG blob detection (same detector as metrics), lattice spacing d =
            median NN distance.
  vacuum    gaussian(target,3) < 0.04, i.e. near-black field-of-view margins.
  vacancy   pixels farther than 0.8*d from every detected column, excluding
            vacuum and a border band of width d (detector unreliable there);
            connected components sized like a single missing column
            (0.15*d^2 .. 4*d^2) become disks of radius 0.35*d at the centroid.
  anomaly   columns whose intensity sits far from BOTH intensity clusters
            (2-means over column intensities separates metal/chalcogen
            sublattices; |I - center| > max(3.5*sigma_cluster, 0.10) for the
            nearer cluster) OR with an NN closer than 0.6*d (adatom/interstitial
            spacing anomaly) become disks of radius 0.35*d.

Output: data/defect_masks/operation_<op>/<structure>.npz with
  mask uint8 HxW: 0 none, 1 vacancy, 2 anomalous column, 3 vacuum
  meta: d, n_cols, n_vacancy_sites, n_anomalous_cols, vacuum_frac

Usage: make_defect_masks.py [--ops A B C] [--workers 32]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_metrics import detect_columns

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
TGT = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Target_BD")
OUT = os.path.join(ROOT, "data", "defect_masks")


def disk(mask, cy, cx, r, val):
    h, w = mask.shape
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r + 1))
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r + 1))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    sel = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    region = mask[y0:y1, x0:x1]
    region[sel & (region == 0)] = val


def column_intensities(t, cols, rad=2):
    vals = []
    sm = gaussian_filter(t, 1.0)
    for y, x in cols:
        vals.append(float(sm[int(round(y)), int(round(x))]))
    return np.asarray(vals)


def two_means(vals, iters=25):
    c = np.array([np.percentile(vals, 25), np.percentile(vals, 75)], dtype=np.float64)
    for _ in range(iters):
        assign = np.abs(vals[:, None] - c[None, :]).argmin(1)
        for k in (0, 1):
            if (assign == k).any():
                c[k] = vals[assign == k].mean()
    sig = np.array([vals[assign == k].std() if (assign == k).sum() > 2 else 0.05 for k in (0, 1)])
    return c, np.maximum(sig, 0.015), assign


def build_mask(t):
    h, w = t.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cols = detect_columns(t)
    meta = dict(d=np.nan, n_cols=len(cols), n_vacancy_sites=0, n_anomalous_cols=0,
                vacuum_frac=0.0)
    if len(cols) < 10:
        vac0 = gaussian_filter(t, 3.0) < 0.04
        mask[vac0] = 3
        meta["vacuum_frac"] = float(vac0.mean())
        return mask, meta
    dnn, _ = cKDTree(cols).query(cols, k=2)
    d = float(np.median(dnn[:, 1]))
    meta["d"] = d

    # --- vacuum: near-black, LARGE, and touching the image border (real
    # field-of-view margins; kills dark inter-column puddles) ---
    dark = gaussian_filter(t, 3.0) < 0.04
    lab_v, nlab_v = label(dark)
    vacuum = np.zeros((h, w), dtype=bool)
    for i in range(1, nlab_v + 1):
        comp = lab_v == i
        if comp.sum() < 4.0 * d * d:
            continue
        ys, xs = np.nonzero(comp)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            vacuum |= comp
    meta["vacuum_frac"] = float(vacuum.mean())

    # --- scan-line correction (additive column-median profile). Known
    # limitation: partial — residual stripe contrast still produces anomaly
    # false positives on a minority of structures (~2/30 in QC); benign for
    # loss weighting (extra protected real columns), mildly dilutive for the
    # preservation metric. Multiplicative and local-reference variants were
    # tried and regressed Defect-Free structures. ---
    interior = ~vacuum
    col_med = np.nanmedian(np.where(interior, t, np.nan), axis=0)
    col_med = np.nan_to_num(col_med, nan=float(np.nanmedian(col_med)))
    stripe = col_med - float(np.median(col_med))
    t_flat = t - stripe[None, :]  # stripe-corrected image for intensity reads
    dark_floor = float(np.percentile(t_flat[interior], 10))

    sm = gaussian_filter(t_flat, 1.0)
    ivals = np.array([float(sm[int(round(y)), int(round(x))]) for y, x in cols])
    c, sig, _ = two_means(ivals)
    c_dim = float(c.min())

    # --- vacancies: far-from-any-column interior pixels whose site is DARK
    # (a missed dim column reads ~c_dim, a true vacancy reads ~dark_floor) ---
    colmap = np.zeros((h, w), dtype=bool)
    colmap[np.clip(cols[:, 0].astype(int), 0, h - 1), np.clip(cols[:, 1].astype(int), 0, w - 1)] = True
    dist = distance_transform_edt(~colmap)
    border = np.zeros((h, w), dtype=bool)
    b = int(round(d))
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    cand = (dist > 0.8 * d) & ~vacuum & ~binary_dilation(vacuum, iterations=b) & ~border
    lab, nlab = label(cand)
    n_vac = 0
    vac_gate = dark_floor + 0.35 * max(c_dim - dark_floor, 0.02)
    for i in range(1, nlab + 1):
        ys, xs = np.nonzero(lab == i)
        area = len(ys)
        if not (0.15 * d * d <= area <= 4.0 * d * d):
            continue
        cy, cx = ys.mean(), xs.mean()
        r = max(2, int(round(0.25 * d)))
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r + 1))
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r + 1))
        if float(t_flat[y0:y1, x0:x1].mean()) > vac_gate:
            continue  # bright-ish: a missed dim column, not a vacancy
        disk(mask, cy, cx, 0.35 * d, 1)
        n_vac += 1
    meta["n_vacancy_sites"] = n_vac

    # --- anomalous columns: intensity outlier vs both sublattice clusters
    # (global 2-means), or spacing anomaly (too-close neighbour) ---
    n_anom = 0
    for i, (y, x) in enumerate(cols):
        k = int(np.abs(ivals[i] - c).argmin())
        intensity_anom = abs(ivals[i] - c[k]) > max(3.5 * sig[k], 0.10)
        spacing_anom = dnn[i, 1] < 0.6 * d
        if intensity_anom or spacing_anom:
            disk(mask, y, x, 0.35 * d, 2)
            n_anom += 1
    meta["n_anomalous_cols"] = n_anom

    mask[vacuum & (mask == 0)] = 3

    # --- class 4 "protect_dark" (TRAINING ONLY, not used by the preservation
    # metric): permissive coverage of every site where a column could be
    # invented — dark pixels far from any detected column, INCLUDING the
    # border band. The strict vacancy gates above keep the metric clean but
    # cover only ~half of audited invention sites; the loss weight needs the
    # rest. ---
    protect = (dist > 0.7 * d) & (t_flat < vac_gate) & ~vacuum & (mask == 0)
    mask[protect] = 4
    meta["protect_dark_frac"] = float(protect.mean())

    # --- class 5 "gb_line" (TRAINING ONLY): grain-boundary seam. Operations
    # are built on GBs — audited borderline-shifted detections cluster 2.5-2.9x
    # near the seam (displacement pressure, not erasure). Located as the
    # deepest central-band dip of the row/column median intensity profile,
    # gated on dip depth; band of ±0.4*d marked. ---
    n_gb = 0
    for axis in (0, 1):  # 0: column profile (vertical seam), 1: row profile
        prof = np.nanmedian(np.where(interior, t, np.nan), axis=axis)
        prof = np.nan_to_num(prof, nan=float(np.nanmedian(prof)))
        L = len(prof)
        lo, hi = int(0.30 * L), int(0.70 * L)
        pos = lo + int(np.argmin(prof[lo:hi]))
        depth = float(np.median(prof) - prof[pos])
        if depth < 0.06:
            continue
        half = max(2, int(round(0.4 * d)))
        sel = np.zeros((h, w), dtype=bool)
        if axis == 0:
            sel[:, max(0, pos - half):pos + half] = True
        else:
            sel[max(0, pos - half):pos + half, :] = True
        sel &= (mask == 0) | (mask == 4)
        mask[sel] = 5
        n_gb += 1
    meta["n_gb_lines"] = n_gb
    return mask, meta


def one(job):
    op, name = job
    t = np.asarray(Image.open(os.path.join(TGT, f"operation_{op}", name)), dtype=np.float32) / 255.0
    mask, meta = build_mask(t)
    out_dir = os.path.join(OUT, f"operation_{op}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, name.replace(".png", ".npz")), mask=mask, **meta)
    return dict(op=op, structure=name, **meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    jobs = []
    for op in args.ops:
        for name in sorted(os.listdir(os.path.join(TGT, f"operation_{op}"))):
            jobs.append((op, name))
    print(f"{len(jobs)} targets", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=8))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "defect_mask_inventory.csv"), index=False)
    print(df[["n_cols", "n_vacancy_sites", "n_anomalous_cols", "vacuum_frac"]].describe().round(3).to_string())
    # sanity: Vacancy-Anion-NN names encode expected defect counts
    va = df[df.structure.str.contains("Vacancy-Anion")].copy()
    va["nominal"] = va.structure.str.extract(r"Vacancy-Anion-(\d+)").astype(float)
    print("\nVacancy-Anion sanity (detected vacancy sites vs nominal count in name):")
    print(va.groupby("nominal")[["n_vacancy_sites"]].median().to_string())


if __name__ == "__main__":
    main()
