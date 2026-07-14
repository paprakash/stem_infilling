"""Evaluation metrics. All operate on float arrays in [0,1] at native resolution.

Committed metric set (NOTES.md 2026-07-13), emitted per damage level and family:
  psnr, ssim              standard fidelity
  kl                      KL(hist_target || hist_pred), 256 bins (group's CycleGAN metric)
  fft_err                 L1 of log10 radially-averaged power spectra
  col_precision/recall    atom-column detection (LoG blobs), pred vs target
  col_rmse                position RMSE of matched columns (px)
  psnr_dmg / psnr_undmg   PSNR restricted to damaged / undamaged pixels
                          (mask = |source-target| > 0.06, dilated 5x5; do-no-harm check)
"""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from skimage.feature import blob_log
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EPS = 1e-8
DAMAGE_THR = 0.06


def hist_kl(target, pred):
    p, _ = np.histogram(target, bins=256, range=(0.0, 1.0))
    q, _ = np.histogram(pred, bins=256, range=(0.0, 1.0))
    p = p.astype(np.float64) + EPS
    q = q.astype(np.float64) + EPS
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def radial_power_spectrum(img):
    f = np.fft.fftshift(np.abs(np.fft.fft2(img)) ** 2)
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2).astype(np.int32)
    nbins = min(h, w) // 2
    sums = np.bincount(r.ravel(), weights=f.ravel(), minlength=nbins)[:nbins]
    cnts = np.bincount(r.ravel(), minlength=nbins)[:nbins]
    return sums / np.maximum(cnts, 1)


def fft_radial_error(target, pred):
    pt = radial_power_spectrum(target)
    pp = radial_power_spectrum(pred)
    return float(np.mean(np.abs(np.log10(pp + EPS) - np.log10(pt + EPS))))


def detect_columns(img):
    """Bright atom-column centers via Laplacian-of-Gaussian. Returns (N,2) y,x."""
    blobs = blob_log(img, min_sigma=2, max_sigma=12, num_sigma=6, threshold=0.05)
    return blobs[:, :2] if len(blobs) else np.zeros((0, 2))


def match_column_positions(target, pred):
    """Full matching detail for diagnostics: detected positions, matches, and
    unmatched predicted columns (false positives). Returns dict with
    ct (N,2), cp (M,2), tol, matches [(ti, pi, dist)], fp_idx (unmatched pred)."""
    ct = detect_columns(target)
    cp = detect_columns(pred)
    out = {"ct": ct, "cp": cp, "tol": np.nan, "matches": [], "fp_idx": []}
    if len(ct) == 0 or len(cp) == 0:
        out["fp_idx"] = list(range(len(cp)))
        return out
    dnn, _ = cKDTree(ct).query(ct, k=2)
    tol = max(2.0, 0.5 * float(np.median(dnn[:, 1])))
    out["tol"] = tol
    d, j = cKDTree(cp).query(ct, k=1)
    order = np.argsort(d)
    used_p, used_t = set(), set()
    for ti in order:
        if d[ti] > tol:
            break
        pi = int(j[ti])
        if pi in used_p or int(ti) in used_t:
            continue
        used_p.add(pi)
        used_t.add(int(ti))
        out["matches"].append((int(ti), pi, float(d[ti])))
    out["fp_idx"] = [i for i in range(len(cp)) if i not in used_p]
    return out


def atom_column_metrics(target, pred):
    ct = detect_columns(target)
    cp = detect_columns(pred)
    out = {"col_precision": np.nan, "col_recall": np.nan, "col_rmse": np.nan,
           "n_cols_target": len(ct), "n_cols_pred": len(cp)}
    if len(ct) == 0 or len(cp) == 0:
        if len(ct) == 0 and len(cp) == 0:
            out.update(col_precision=1.0, col_recall=1.0)
        else:
            out.update(col_precision=0.0 if len(cp) else np.nan,
                       col_recall=0.0 if len(ct) else np.nan)
        return out
    # tolerance: half the median nearest-neighbour spacing of true columns
    dnn, _ = cKDTree(ct).query(ct, k=2)
    tol = max(2.0, 0.5 * float(np.median(dnn[:, 1])))
    # greedy one-to-one matching, nearest first
    d, j = cKDTree(cp).query(ct, k=1)
    order = np.argsort(d)
    used_p, used_t, errs = set(), set(), []
    for ti in order:
        if d[ti] > tol:
            break
        pi = int(j[ti])
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(int(ti))
        errs.append(d[ti])
    n_match = len(errs)
    out["col_precision"] = n_match / len(cp)
    out["col_recall"] = n_match / len(ct)
    out["col_rmse"] = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
    return out


def damage_mask(source, target, thr=DAMAGE_THR):
    return binary_dilation(np.abs(source - target) > thr, iterations=2)


def region_metrics(source, target, pred):
    m = damage_mask(source, target)
    out = {"frac_damaged": float(m.mean())}
    for name, sel in (("dmg", m), ("undmg", ~m)):
        if sel.sum() < 10:
            out[f"psnr_{name}"] = np.nan
            out[f"mae_{name}"] = np.nan
            continue
        err = (pred[sel] - target[sel]).astype(np.float64)
        mse = float(np.mean(err ** 2))
        out[f"psnr_{name}"] = float(10 * np.log10(1.0 / max(mse, 1e-12)))
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
    return out


def defect_preservation(target, pred, defect_mask, tol=0.08):
    """Fraction of defect sites (mask classes 1=vacancy, 2=anomalous column)
    where the prediction stays within `tol` mean-abs of the target inside the
    site disk — dark stays dark, bright stays bright. Also the vacuum phantom
    rate: fraction of vacuum pixels (class 3) where pred exceeds target by 0.1.
    """
    from scipy.ndimage import label as cc_label
    out = {"defect_sites": 0, "defect_preserved_frac": np.nan,
           "vacancy_preserved_frac": np.nan, "anom_preserved_frac": np.nan,
           "vacuum_phantom_frac": np.nan}
    per_class = {}
    for cls, key in ((1, "vacancy"), (2, "anom")):
        sel = defect_mask == cls
        if not sel.any():
            continue
        lab, n = cc_label(sel)
        kept = 0
        for i in range(1, n + 1):
            site = lab == i
            if np.abs(pred[site] - target[site]).mean() <= tol:
                kept += 1
        per_class[key] = (kept, n)
    n_tot = sum(n for _, n in per_class.values())
    out["defect_sites"] = n_tot
    if n_tot:
        out["defect_preserved_frac"] = sum(k for k, _ in per_class.values()) / n_tot
    for key in ("vacancy", "anom"):
        if key in per_class:
            k, n = per_class[key]
            out[f"{key}_preserved_frac"] = k / n
    vac = defect_mask == 3
    if vac.sum() >= 10:
        out["vacuum_phantom_frac"] = float(((pred - target)[vac] > 0.10).mean())
    return out


def core_metrics(target, pred):
    return {
        "psnr": peak_signal_noise_ratio(target, pred, data_range=1.0),
        "ssim": structural_similarity(target, pred, data_range=1.0),
        "kl": hist_kl(target, pred),
        "fft_err": fft_radial_error(target, pred),
    }


def all_metrics(source, target, pred, with_columns=True, defect_mask=None):
    out = core_metrics(target, pred)
    out.update(region_metrics(source, target, pred))
    if with_columns:
        out.update(atom_column_metrics(target, pred))
    if defect_mask is not None:
        out.update(defect_preservation(target, pred, defect_mask))
    return out
