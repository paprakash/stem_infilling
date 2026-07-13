#!/usr/bin/env python3
"""Structure-level 80/10/10 splits, stratified by material family.

A "structure" is a target filename (e.g. cose2_Fe_adatom.png). All damage levels
and all operations (A/B/C) of a structure land in the same split. Stratification:
within each material family (cose2, mos2, ... — polytype -A/-B suffixes merged),
structures are shuffled with a fixed seed and dealt 80/10/10; small families are
still guaranteed >=1 val and >=1 test structure when they have >=3 members.

Outputs (data/splits/):
  train.txt / val.txt / test.txt   one structure filename per line
  split_manifest.csv               structure, family, split, ops present, levels present
  split_report.md                  family x split table for verification
"""
import os
import random
from collections import defaultdict

import pandas as pd

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SRC = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Source_BD")
TGT = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Target_BD")
OUT = os.path.join(ROOT, "data", "splits")
OPS = ["A", "B", "C"]
LEVELS = [1, 2, 4, 8, 12, 16, 20, 28, 36]
SEED = 20260713
FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def material_family(name: str) -> str:
    base = name.replace(".png", "").split("_")[0]
    if base.endswith(("-A", "-B")):
        base = base[:-2]
    return base


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = {op: set(os.listdir(os.path.join(TGT, f"operation_{op}"))) for op in OPS}
    all_structs = sorted(set().union(*targets.values()))

    by_family = defaultdict(list)
    for s in all_structs:
        by_family[material_family(s)].append(s)

    rng = random.Random(SEED)
    assign = {}
    for fam in sorted(by_family):
        members = sorted(by_family[fam])
        rng.shuffle(members)
        n = len(members)
        n_val = max(1, round(FRACS["val"] * n)) if n >= 3 else 0
        n_test = max(1, round(FRACS["test"] * n)) if n >= 3 else 0
        for i, s in enumerate(members):
            if i < n_test:
                assign[s] = "test"
            elif i < n_test + n_val:
                assign[s] = "val"
            else:
                assign[s] = "train"

    # manifest with per-structure availability
    rows = []
    for s in all_structs:
        ops_present = [op for op in OPS if s in targets[op]]
        lvls = set(LEVELS)
        for op in ops_present:
            lvls &= {lv for lv in LEVELS
                     if os.path.exists(os.path.join(SRC, f"operation_{op}", f"beam_damage_{lv}", s))}
        rows.append(dict(structure=s, family=material_family(s), split=assign[s],
                         ops=",".join(ops_present), n_ops=len(ops_present),
                         levels=",".join(map(str, sorted(lvls))), n_levels=len(lvls)))
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(OUT, "split_manifest.csv"), index=False)

    for split in ["train", "val", "test"]:
        names = man[man.split == split].structure.tolist()
        with open(os.path.join(OUT, f"{split}.txt"), "w") as fh:
            fh.write("\n".join(names) + "\n")

    # report
    tab = man.pivot_table(index="family", columns="split", values="structure",
                          aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)
    totals = man.split.value_counts()
    n_pairs = man.assign(pairs=man.n_ops * man.n_levels).groupby("split").pairs.sum()
    lines = ["# Split report", "",
             f"Seed {SEED}; structure-level 80/10/10 stratified by family.", "",
             f"Structures: train={totals.get('train', 0)} val={totals.get('val', 0)} test={totals.get('test', 0)}",
             f"Source-target pairs: train={n_pairs.get('train', 0)} val={n_pairs.get('val', 0)} test={n_pairs.get('test', 0)}", "",
             tab.to_markdown(), ""]
    with open(os.path.join(OUT, "split_report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
