#!/usr/bin/env python3
"""Identity baselines: metrics of the RAW damaged source against its target.

This is the "do nothing" model — any trained model must beat these numbers.
Computed for every source-target pair, then aggregated per damage level and per
material family as median + IQR (damage is non-monotonic per structure, so
distributions matter more than means).

Metrics per pair (uint8 images scaled to [0,1], native resolution, whole image):
  psnr        peak signal-to-noise ratio, data_range=1
  ssim        structural similarity, data_range=1
  kl          KL(hist_target || hist_source), 256 bins, eps-smoothed
              (same intensity-histogram KL family the group uses for CycleGAN)

Outputs (results/phase0/):
  identity_per_image.csv     one row per source-target pair
  identity_by_level.csv      median/IQR per damage level (+ per split)
  identity_by_family.csv     median/IQR per family x level
  identity_summary.md        headline tables
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SRC = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Source_BD")
TGT = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Target_BD")
OUT = os.path.join(ROOT, "results", "phase0")
SPLITS = os.path.join(ROOT, "data", "splits")
OPS = ["A", "B", "C"]
LEVELS = [1, 2, 4, 8, 12, 16, 20, 28, 36]
EPS = 1e-8


def material_family(name: str) -> str:
    base = name.replace(".png", "").split("_")[0]
    if base.endswith(("-A", "-B")):
        base = base[:-2]
    return base


def hist_kl(target: np.ndarray, source: np.ndarray) -> float:
    """KL(P_target || Q_source) over 256-bin intensity histograms."""
    p, _ = np.histogram(target, bins=256, range=(0.0, 1.0), density=False)
    q, _ = np.histogram(source, bins=256, range=(0.0, 1.0), density=False)
    p = p.astype(np.float64) + EPS
    q = q.astype(np.float64) + EPS
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def one_pair(job):
    structure, op, level = job
    t = np.asarray(Image.open(os.path.join(TGT, f"operation_{op}", structure)), dtype=np.float32) / 255.0
    s = np.asarray(Image.open(os.path.join(SRC, f"operation_{op}", f"beam_damage_{level}", structure)),
                   dtype=np.float32) / 255.0
    return dict(
        structure=structure, family=material_family(structure), op=op, level=level,
        psnr=peak_signal_noise_ratio(t, s, data_range=1.0),
        ssim=structural_similarity(t, s, data_range=1.0),
        kl=hist_kl(t, s),
    )


def agg(df, by):
    g = df.groupby(by)[["psnr", "ssim", "kl"]]
    out = pd.concat([
        g.median().add_suffix("_med"),
        g.quantile(0.25).add_suffix("_q25"),
        g.quantile(0.75).add_suffix("_q75"),
        g.size().rename("n"),
    ], axis=1).reset_index()
    return out.round(4)


def main():
    split_of = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(SPLITS, f"{split}.txt")) as fh:
            for line in fh:
                split_of[line.strip()] = split

    jobs = []
    for op in OPS:
        for lv in LEVELS:
            d = os.path.join(SRC, f"operation_{op}", f"beam_damage_{lv}")
            for f in sorted(os.listdir(d)):
                jobs.append((f, op, lv))
    print(f"{len(jobs)} pairs")

    with ProcessPoolExecutor(max_workers=32) as ex:
        recs = list(ex.map(one_pair, jobs, chunksize=32))
    df = pd.DataFrame(recs)
    df["split"] = df.structure.map(split_of)
    df.to_csv(os.path.join(OUT, "identity_per_image.csv"), index=False)

    by_level = agg(df, ["level"])
    by_level_split = agg(df, ["split", "level"])
    by_family = agg(df, ["family", "level"])
    by_level.to_csv(os.path.join(OUT, "identity_by_level.csv"), index=False)
    by_level_split.to_csv(os.path.join(OUT, "identity_by_level_split.csv"), index=False)
    by_family.to_csv(os.path.join(OUT, "identity_by_family.csv"), index=False)

    fam_l36 = by_family[by_family.level == 36].sort_values("psnr_med")
    lines = ["# Identity baselines (raw source vs target)", "",
             "## By damage level (all splits pooled) — median [q25, q75]", "",
             by_level.to_markdown(index=False), "",
             "## Test split only", "",
             by_level_split[by_level_split.split == "test"].to_markdown(index=False), "",
             "## Hardest families at level 36 (by median PSNR)", "",
             fam_l36.head(10).to_markdown(index=False), ""]
    with open(os.path.join(OUT, "identity_summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
