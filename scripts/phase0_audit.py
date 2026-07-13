#!/usr/bin/env python3
"""Phase 0 data audit for the STEM beam-damage dataset.

Outputs (results/phase0/):
  audit_counts.csv        images per operation x damage level (+ target counts)
  audit_pairing.csv       pairing integrity: sources without target, targets without full level set
  audit_sizes.csv         image-size inventory (per unique WxH: n structures)
  audit_intensity.csv     per-image intensity stats for ALL images (structure, op, level, mean, std, min, max, frac_saturated)
  audit_summary.md        human-readable summary of all of the above
"""
import os
import sys
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SRC = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Source_BD")
TGT = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Target_BD")
OUT = os.path.join(ROOT, "results", "phase0")
OPS = ["A", "B", "C"]
LEVELS = [1, 2, 4, 8, 12, 16, 20, 28, 36]


def material_family(name: str) -> str:
    """cose2_Fe_adatom.png -> cose2 ; mos2-A_Vacancy-Anion-06A.png -> mos2 (polytype suffix stripped)."""
    base = name.replace(".png", "").split("_")[0]
    if base.endswith(("-A", "-B")):
        base = base[:-2]
    return base


def img_stats(path):
    a = np.asarray(Image.open(path), dtype=np.float32)
    return dict(
        h=a.shape[0], w=a.shape[1],
        mean=float(a.mean()), std=float(a.std()),
        min=float(a.min()), max=float(a.max()),
        frac_sat=float((a >= 254).mean()),
    )


def one_image(job):
    structure, op, level, path = job
    s = img_stats(path)
    return dict(structure=structure, op=op, level=level, family=material_family(structure), **s)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------- inventory ----------
    targets = {op: sorted(os.listdir(os.path.join(TGT, f"operation_{op}"))) for op in OPS}
    sources = {}
    for op in OPS:
        for lv in LEVELS:
            d = os.path.join(SRC, f"operation_{op}", f"beam_damage_{lv}")
            sources[(op, lv)] = sorted(os.listdir(d)) if os.path.isdir(d) else []

    rows = []
    for op in OPS:
        row = {"operation": op, "targets": len(targets[op])}
        for lv in LEVELS:
            row[f"lvl_{lv}"] = len(sources[(op, lv)])
        row["source_total"] = sum(row[f"lvl_{lv}"] for lv in LEVELS)
        rows.append(row)
    counts = pd.DataFrame(rows)
    counts.to_csv(os.path.join(OUT, "audit_counts.csv"), index=False)

    # ---------- pairing integrity ----------
    problems = []
    for op in OPS:
        tset = set(targets[op])
        for lv in LEVELS:
            for f in sources[(op, lv)]:
                if f not in tset:
                    problems.append(dict(kind="source_without_target", op=op, level=lv, file=f))
        # targets missing some damage level
        for f in targets[op]:
            missing = [lv for lv in LEVELS if f not in set(sources[(op, lv)])]
            if missing:
                problems.append(dict(kind="target_missing_levels", op=op, level=",".join(map(str, missing)), file=f))
    # structures not present in all operations
    all_names = sorted(set().union(*[set(t) for t in targets.values()]))
    for f in all_names:
        ops_missing = [op for op in OPS if f not in set(targets[op])]
        if ops_missing:
            problems.append(dict(kind="structure_missing_operation", op=",".join(ops_missing), level="", file=f))
    pairing = pd.DataFrame(problems, columns=["kind", "op", "level", "file"])
    pairing.to_csv(os.path.join(OUT, "audit_pairing.csv"), index=False)

    # ---------- per-image intensity stats (all images, parallel) ----------
    jobs = []
    for op in OPS:
        for f in targets[op]:
            jobs.append((f, op, 0, os.path.join(TGT, f"operation_{op}", f)))  # level 0 = target
        for lv in LEVELS:
            for f in sources[(op, lv)]:
                jobs.append((f, op, lv, os.path.join(SRC, f"operation_{op}", f"beam_damage_{lv}", f)))
    with ProcessPoolExecutor(max_workers=16) as ex:
        recs = list(ex.map(one_image, jobs, chunksize=64))
    inten = pd.DataFrame(recs)
    inten.to_csv(os.path.join(OUT, "audit_intensity.csv"), index=False)

    # ---------- size inventory (from target images; one row per unique size) ----------
    tgt_only = inten[inten.level == 0].drop_duplicates(subset=["structure"])
    size_inv = (tgt_only.groupby(["h", "w"]).size().reset_index(name="n_structures")
                .sort_values("n_structures", ascending=False))
    size_inv.to_csv(os.path.join(OUT, "audit_sizes.csv"), index=False)

    # ---------- family inventory ----------
    fam = tgt_only.groupby("family").size().reset_index(name="n_structures").sort_values("n_structures", ascending=False)
    fam.to_csv(os.path.join(OUT, "audit_families.csv"), index=False)

    # ---------- summary ----------
    lines = ["# Phase 0 data audit", ""]
    lines += ["## Counts per operation x damage level", "", counts.to_markdown(index=False), ""]
    lines += [f"## Pairing integrity: {len(pairing)} problems", ""]
    if len(pairing):
        lines += [pairing.to_markdown(index=False), ""]
    else:
        lines += ["All sources have a target; all targets have all 9 levels; all structures in all 3 operations.", ""]
    lines += ["## Image sizes (unique HxW over structures)", "", size_inv.to_markdown(index=False), ""]
    mn, mx = size_inv[["h", "w"]].min().min(), size_inv[["h", "w"]].max().max()
    small = size_inv[(size_inv.h < 384) | (size_inv.w < 384)]
    lines += [f"Smallest dim {mn}, largest {mx}. Structures with any dim < 384 px: {small.n_structures.sum()}", ""]
    lines += ["## Material families", "", fam.to_markdown(index=False), ""]
    g = inten.groupby("level")[["mean", "std", "frac_sat"]].median().round(2)
    lines += ["## Median intensity stats by level (0 = target)", "", g.to_markdown(), ""]
    with open(os.path.join(OUT, "audit_summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
