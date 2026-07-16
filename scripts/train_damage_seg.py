#!/usr/bin/env python3
"""Train the Phase-3 damage-mask segmenter: source -> damage probability.

Ground truth: dilate(|source-target| > 0.06, 2 iters) — same definition as the
eval damage mask. Train split only (structure-disjoint). Small NAFNet (width 16)
with a single-logit output; BCE with pos_weight for the minority damage class.

Usage: train_damage_seg.py --config runs/damage_seg_v1/config.yaml [--resume]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import LEVELS, build_index, dihedral, load_pair, load_split, val_subset_structures
from train_stem import build_model, pad_to_multiple

ROOT = "/blue/hennig/pawanprakash/ornl_stem"


class DamageSegDataset(Dataset):
    def __init__(self, split="train", crop=384):
        self.index = build_index(load_split(split))
        self.crop = crop

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        structure, op, lv = self.index[i]
        s01, t01, mean, std = load_pair(structure, op, lv)
        m = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2).astype(np.float32)
        s = (s01 - mean) / std
        c = self.crop
        h, w = s.shape
        if h < c or w < c:
            ph, pw = max(0, c - h), max(0, c - w)
            s = np.pad(s, ((0, ph), (0, pw)), mode="reflect")
            m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
            h, w = s.shape
        y = torch.randint(0, h - c + 1, (1,)).item()
        x = torch.randint(0, w - c + 1, (1,)).item()
        s, m = s[y:y + c, x:x + c], m[y:y + c, x:x + c]
        k = torch.randint(0, 8, (1,)).item()
        return (torch.from_numpy(dihedral(s, k).copy())[None],
                torch.from_numpy(dihedral(m, k).copy())[None])


@torch.no_grad()
def seg_validate(model, pairs, thr=0.5):
    """Pixel precision/recall of thresholded logits vs ground-truth mask, per level."""
    model.eval()
    stats = {lv: [0, 0, 0] for lv in LEVELS}  # tp, fp, fn
    for structure, op, lv in pairs:
        s01, t01, mean, std = load_pair(structure, op, lv)
        gt = binary_dilation(np.abs(s01 - t01) > 0.06, iterations=2)
        x = torch.from_numpy((s01 - mean) / std)[None, None].cuda()
        x, h, w = pad_to_multiple(x, 16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x)
        pred = (torch.sigmoid(logit.float())[0, 0, :h, :w].cpu().numpy() > thr)
        stats[lv][0] += int((pred & gt).sum())
        stats[lv][1] += int((pred & ~gt).sum())
        stats[lv][2] += int((~pred & gt).sum())
    model.train()
    out = {}
    for lv, (tp, fp, fn) in stats.items():
        out[lv] = (tp / max(tp + fp, 1), tp / max(tp + fn, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run_dir = os.path.join(ROOT, "runs", cfg["exp"])
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(cfg["train"]["seed"])

    model = build_model(cfg["model"]).cuda()
    print(f"[{cfg['exp']}] {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
    total = cfg["train"]["total_iters"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=cfg["optim"]["min_lr"])
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["loss"]["pos_weight"]).cuda())

    start = 0
    latest = os.path.join(run_dir, "latest.pth")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start = ck["iter"]
        print(f"resumed at {start}", flush=True)

    ds = DamageSegDataset("train", cfg["data"]["crop"])
    print(f"train pairs: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["data"]["batch"], shuffle=True,
                                         num_workers=cfg["data"]["workers"], pin_memory=True,
                                         drop_last=True, persistent_workers=True)
    val_pairs = build_index(val_subset_structures(), ops=("A",))

    it, t0 = start, time.time()
    model.train()
    while it < total:
        for sb, mb in loader:
            if it >= total:
                break
            sb, mb = sb.cuda(non_blocking=True), mb.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(sb)
            # NAFNet adds a global residual (out = f(x) + x); harmless for a
            # logit head — BCE learns around it
            loss = bce(logit.float(), mb.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); it += 1
            if it % cfg["train"]["log_every"] == 0:
                print(f"it {it}/{total} bce {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(it-start)/(time.time()-t0):.2f} it/s", flush=True)
            if it % cfg["train"]["val_every"] == 0 or it == total:
                pr = seg_validate(model, val_pairs)
                print("--- seg val P/R by level:", {lv: (round(p, 3), round(r, 3)) for lv, (p, r) in pr.items()}, flush=True)
            if it % cfg["train"]["ckpt_every"] == 0 or it == total:
                torch.save({"iter": it, "model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "cfg": cfg}, latest + ".tmp")
                os.replace(latest + ".tmp", latest)
    print(f"DONE at iter {it}", flush=True)


if __name__ == "__main__":
    main()
