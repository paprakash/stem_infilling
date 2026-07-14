#!/usr/bin/env python3
"""Diagnostic 3: matched-noise injection probe.

Estimate the high-frequency noise of each SOURCE image (deployment-legal: the
source is available at inference), synthesize a noise field with the same radial
amplitude spectrum and variance, add it to the model prediction, and recompute
intensity-KL and FFT radial error. If KL drops to ~identity levels, the texture
gap is closable with a zero-hallucination post-process.

Noise estimate: n = src - gaussian(src, sigma=1.5). Synthesis: white noise
shaped in Fourier space by n's radially-averaged amplitude profile, rescaled to
std(n). Fixed seed. Writes results/phase1/diagnostics/noise_probe.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import LEVELS, load_pair, val_subset_structures
from stem_metrics import fft_radial_error, hist_kl

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
CACHE = os.path.join(ROOT, "data", "cache_phase1_preds")
OUT = os.path.join(ROOT, "results", "phase1", "diagnostics")
MODELS = ["nafnet", "restormer"]
SEED = 20260714


def synth_matched_noise(noise, rng):
    """White noise shaped by `noise`'s radial amplitude spectrum, same std."""
    h, w = noise.shape
    amp = np.abs(np.fft.fft2(noise))
    yy, xx = np.ogrid[:h, :w]
    fy = np.minimum(yy, h - yy)
    fx = np.minimum(xx, w - xx)
    r = np.hypot(fy, fx).astype(np.int32)
    prof = np.bincount(r.ravel(), weights=amp.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    shaped = np.fft.fft2(rng.standard_normal((h, w))) * prof[r]
    out = np.real(np.fft.ifft2(shaped))
    out *= noise.std() / max(out.std(), 1e-12)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for model in MODELS:
        for s_name in val_subset_structures():
            for lv in LEVELS:
                s, t, _, _ = load_pair(s_name, "A", lv)
                pred = np.load(os.path.join(CACHE, model, f"{s_name.replace('.png', '')}_lvl{lv}.npy"))
                noise_est = s - gaussian_filter(s, 1.5)
                pred_n = np.clip(pred + synth_matched_noise(noise_est, rng), 0.0, 1.0)
                rows.append(dict(model=model, structure=s_name, level=lv,
                                 noise_std=float(noise_est.std()),
                                 kl_before=hist_kl(t, pred), kl_after=hist_kl(t, pred_n),
                                 fft_before=fft_radial_error(t, pred),
                                 fft_after=fft_radial_error(t, pred_n)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "noise_probe.csv"), index=False)
    agg = df.groupby(["model", "level"])[["kl_before", "kl_after", "fft_before", "fft_after"]].median().round(4)
    print(agg.to_string())


if __name__ == "__main__":
    main()
