"""Paired STEM dataset: beam-damaged source -> pristine target.

Normalization contract (used everywhere, incl. inference where only the source
exists): both images are scaled to [0,1] then standardized with the SOURCE's
mean/std. Predictions are de-normalized with the same source stats and clipped
to [0,1] before any metric is computed.

Train pipeline: reflect-pad if smaller than crop (audit: never happens, guard
anyway) -> random aligned crop -> random dihedral (flips/90-deg rotations only).
Level balance: the dataset has 617-618 pairs per damage level per operation, so
uniform sampling over pairs is balanced across the 9 levels by construction.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
SRC = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Source_BD")
TGT = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Target_BD")
SPLITS = os.path.join(ROOT, "data", "splits")
OPS = ("A", "B", "C")
LEVELS = (1, 2, 4, 8, 12, 16, 20, 28, 36)


def material_family(name: str) -> str:
    base = name.replace(".png", "").split("_")[0]
    if base.endswith(("-A", "-B")):
        base = base[:-2]
    return base


def load_split(split: str) -> list:
    with open(os.path.join(SPLITS, f"{split}.txt")) as fh:
        return [l.strip() for l in fh if l.strip()]


def build_index(structures, ops=OPS, levels=LEVELS):
    """[(structure, op, level)] for every pair that exists on disk."""
    idx = []
    for op in ops:
        tdir = os.path.join(TGT, f"operation_{op}")
        for lv in levels:
            sdir = os.path.join(SRC, f"operation_{op}", f"beam_damage_{lv}")
            for s in structures:
                if os.path.exists(os.path.join(sdir, s)) and os.path.exists(os.path.join(tdir, s)):
                    idx.append((s, op, lv))
    return idx


def load_pair(structure: str, op: str, level: int):
    """Returns (source, target) float32 arrays in [0,1], plus (mean, std) of source."""
    s = np.asarray(Image.open(os.path.join(SRC, f"operation_{op}", f"beam_damage_{level}", structure)),
                   dtype=np.float32) / 255.0
    t = np.asarray(Image.open(os.path.join(TGT, f"operation_{op}", structure)),
                   dtype=np.float32) / 255.0
    mean, std = float(s.mean()), max(float(s.std()), 1e-6)
    return s, t, mean, std


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rot90 x (k%4), then horizontal flip if k>=4."""
    a = np.rot90(a, k % 4)
    if k >= 4:
        a = np.fliplr(a)
    return a


class PairedSTEMTrain(Dataset):
    def __init__(self, split="train", crop=384, seed=0, defect_weight=1.0):
        """defect_weight > 1 returns a per-pixel loss-weight map as third tensor:
        `defect_weight` inside defect+vacuum mask disks (data/defect_masks/),
        1 elsewhere. Weight maps go through the SAME crop/dihedral as images."""
        self.index = build_index(load_split(split))
        self.crop = crop
        self.rng = random.Random(seed)
        self.defect_weight = defect_weight

    def __len__(self):
        return len(self.index)

    def _load_weight(self, structure, op, shape):
        p = os.path.join(ROOT, "data", "defect_masks", f"operation_{op}",
                         structure.replace(".png", ".npz"))
        w = np.ones(shape, dtype=np.float32)
        if self.defect_weight != 1.0 and os.path.exists(p):
            m = np.load(p)["mask"]
            w[m > 0] = self.defect_weight
        return w

    def __getitem__(self, i):
        structure, op, lv = self.index[i]
        s, t, mean, std = load_pair(structure, op, lv)
        s = (s - mean) / std
        t = (t - mean) / std
        wmap = self._load_weight(structure, op, s.shape)

        c = self.crop
        h, w = s.shape
        if h < c or w < c:  # guard: reflect-pad up to crop size
            ph, pw = max(0, c - h), max(0, c - w)
            s = np.pad(s, ((0, ph), (0, pw)), mode="reflect")
            t = np.pad(t, ((0, ph), (0, pw)), mode="reflect")
            wmap = np.pad(wmap, ((0, ph), (0, pw)), mode="reflect")
            h, w = s.shape
        y = torch.randint(0, h - c + 1, (1,)).item()
        x = torch.randint(0, w - c + 1, (1,)).item()
        s = s[y:y + c, x:x + c]
        t = t[y:y + c, x:x + c]
        wmap = wmap[y:y + c, x:x + c]

        k = torch.randint(0, 8, (1,)).item()
        s = dihedral(s, k).copy()
        t = dihedral(t, k).copy()
        wmap = dihedral(wmap, k).copy()

        return torch.from_numpy(s)[None], torch.from_numpy(t)[None], torch.from_numpy(wmap)[None]


def val_subset_structures():
    p = os.path.join(SPLITS, "val_subset.txt")
    with open(p) as fh:
        return [l.strip() for l in fh if l.strip()]
