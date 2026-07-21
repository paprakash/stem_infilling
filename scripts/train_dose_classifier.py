#!/usr/bin/env python3
"""Tiny dose classifier: source image -> low (levels 1-4) vs high (8+).

Labels come from the beam_damage_<level> directory structure. Fully-conv
4-block CNN + GAP + linear so variable image sizes work at inference.
Train split only; reports val accuracy overall, per level, and the 4-vs-8
confusion specifically (the adjacent boundary).

Usage: train_dose_classifier.py            # train + validate
       train_dose_classifier.py --eval-only
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import LEVELS, build_index, dihedral, load_pair, load_split

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
RUN = os.path.join(ROOT, "runs", "dose_classifier_v1")
LOW = {1, 2, 4}


class DoseNet(nn.Module):
    def __init__(self, ch=24):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in (ch, ch * 2, ch * 4, ch * 4):
            blocks += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.GELU(),
                       nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU()]
            c_in = c_out
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):
        f = self.body(x).mean(dim=(2, 3))
        return self.head(f)[:, 0]  # logit: >0 = high


class DoseDataset(Dataset):
    def __init__(self, split="train", crop=256):
        self.index = build_index(load_split(split))
        self.crop = crop

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        structure, op, lv = self.index[i]
        s, _, mean, std = load_pair(structure, op, lv)
        s = (s - mean) / std
        c = self.crop
        h, w = s.shape
        y = torch.randint(0, h - c + 1, (1,)).item()
        x = torch.randint(0, w - c + 1, (1,)).item()
        k = torch.randint(0, 8, (1,)).item()
        s = dihedral(s[y:y + c, x:x + c], k).copy()
        return torch.from_numpy(s)[None], torch.tensor(0.0 if lv in LOW else 1.0)


@torch.no_grad()
def evaluate(model, pairs):
    model.eval()
    rows = []
    for structure, op, lv in pairs:
        s, _, mean, std = load_pair(structure, op, lv)
        x = torch.from_numpy((s - mean) / std)[None, None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x).float().item()
        rows.append((lv, logit))
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--iters", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    torch.manual_seed(20260721)

    model = DoseNet().cuda()
    ckpt = os.path.join(RUN, "latest.pth")
    if not args.eval_only:
        ds = DoseDataset("train")
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=8,
                                             pin_memory=True, drop_last=True, persistent_workers=True)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)
        bce = nn.BCEWithLogitsLoss()
        it, t0 = 0, time.time()
        model.train()
        while it < args.iters:
            for xb, yb in loader:
                if it >= args.iters:
                    break
                xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(xb)
                loss = bce(logit.float(), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step(); it += 1
                if it % 500 == 0:
                    print(f"it {it}/{args.iters} bce {loss.item():.4f} {it/(time.time()-t0):.1f} it/s", flush=True)
        torch.save({"model": model.state_dict()}, ckpt)
    else:
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"])

    # full-val evaluation (whole images, blind)
    pairs = build_index(load_split("val"))
    model.eval()
    rows = []
    with torch.no_grad():
        for structure, op, lv in pairs:
            s, _, mean, std = load_pair(structure, op, lv)
            x = torch.from_numpy((s - mean) / std)[None, None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x).float().item()
            rows.append((structure, op, lv, logit))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["structure", "op", "level", "logit"])
    df["pred_high"] = df.logit > 0
    df["true_high"] = ~df.level.isin(sorted(LOW))
    df.to_csv(os.path.join(RUN, "val_logits.csv"), index=False)
    acc = (df.pred_high == df.true_high).mean()
    per = df.groupby("level").apply(lambda g: (g.pred_high == g.true_high).mean(), include_groups=False)
    print(f"overall acc {acc:.4f}")
    print("per-level acc:", per.round(4).to_dict())
    c4 = df[df.level == 4]; c8 = df[df.level == 8]
    print(f"4-vs-8 confusion: lvl4 misread as high {(~(c4.pred_high == False)).mean():.4f}, "
          f"lvl8 misread as low {(c8.pred_high == False).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
