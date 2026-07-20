#!/usr/bin/env python3
"""Meeting figures (2026-07-17). Jobs 1-2 are CURATED (selection criteria in
code, sourced from cached predictions + audit CSVs); job 3 is RANDOM (fixed
seed, seed in filenames, drawn over the FULL val split with fresh inference —
typical behavior, successes and failures both).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import load_pair, load_split, val_subset_structures
from stem_metrics import damage_mask
from train_stem import build_model, infer_full

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
CACHE = os.path.join(ROOT, "data", "cache_phase1_preds")
OUT = os.path.join(ROOT, "results", "phase3")
VOUT = os.path.join(OUT, "visual_verification")
SEED = 20260717
PANEL = 384


def cached(model, structure, lv):
    return np.load(os.path.join(CACHE, model, f"{structure.replace('.png', '')}_lvl{lv}.npy"))


def crop_at(a, cy, cx, size):
    h, w = a.shape
    y0 = int(min(max(cy - size // 2, 0), h - size))
    x0 = int(min(max(cx - size // 2, 0), w - size))
    return a[y0:y0 + size, x0:x0 + size]


def to_img(a, size=PANEL):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return im


def strip(panels, labels, size=PANEL, cap=18, pad=4):
    W = len(panels) * (size + pad) + pad
    img = Image.new("L", (W, size + cap + pad), 30)
    d = ImageDraw.Draw(img)
    for i, (p, lab) in enumerate(zip(panels, labels)):
        x0 = pad + i * (size + pad)
        img.paste(to_img(p, size), (x0, cap))
        d.text((x0 + 2, 2), lab, fill=255)
    return img


def vstack(imgs, pad=6):
    W = max(i.size[0] for i in imgs)
    H = sum(i.size[1] for i in imgs) + pad * (len(imgs) - 1)
    out = Image.new("L", (W, H), 30)
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.size[1] + pad
    return out


def most_damaged_center(s, t):
    m = damage_mask(s, t)
    lab, n = cc_label(m)
    if n == 0:
        return s.shape[0] // 2, s.shape[1] // 2
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    ys, xs = np.nonzero(lab == sizes.argmax())
    return int(ys.mean()), int(xs.mean())


def psnr_region(pred, t, sel):
    if sel.sum() < 10:
        return np.nan
    return 10 * np.log10(1.0 / max(float(np.mean((pred[sel] - t[sel]) ** 2)), 1e-12))


# ---------------- job 1 ----------------
def job1():
    subset = val_subset_structures()
    # row 1: level-36 case, large damage, max ft-over-scratch damaged-PSNR gap
    best, pick = -1e9, None
    for s_name in subset:
        s, t, _, _ = load_pair(s_name, "A", 36)
        m = damage_mask(s, t)
        if m.mean() < 0.05:
            continue
        gap = (psnr_region(cached("nafnet_asym1", s_name, 36), t, m)
               - psnr_region(cached("nafnet_scratch", s_name, 36), t, m))
        score = gap + 2.0 * m.mean()
        if score > best:
            best, pick = score, s_name
    s, t, _, _ = load_pair(pick, "A", 36)
    cy, cx = most_damaged_center(s, t)
    row1 = strip([crop_at(a, cy, cx, PANEL) for a in
                  (s, cached("nafnet_base100k", pick, 36), cached("nafnet_asym1", pick, 36),
                   cached("nafnet_scratch", pick, 36), t)],
                 [f"input L36 {pick[:22]}", "base 100k", "+defect FT (asym1)", "from-scratch", "target"])
    # row 2: level 1-2 invented-atom site (base audit), vacancy-rich preferred
    cases = pd.read_csv(f"{ROOT}/results/phase1/diagnostics/hallucination_cases_base100k.csv")
    inv = cases[(cases.cls == "invented_undamaged") & (cases.level.isin([1, 2]))
                & (cases.tgt_val < 0.15)].copy()
    inv["vac_rich"] = inv.structure.str.contains("Vacancy|adatom|Adatom")
    inv = inv.sort_values(["vac_rich", "pred_val"], ascending=[False, False])
    r = inv.iloc[0]
    s, t, _, _ = load_pair(r.structure, "A", int(r.level))
    row2 = strip([crop_at(a, int(r.y), int(r.x), 192) for a in
                  (s, cached("nafnet_base100k", r.structure, int(r.level)),
                   cached("nafnet_asym1", r.structure, int(r.level)),
                   cached("nafnet_scratch", r.structure, int(r.level)), t)],
                 [f"input L{int(r.level)} {r.structure[:22]}", "base 100k (invents)",
                  "+defect FT (asym1)", "from-scratch (keeps dark)", "target"])
    vstack([row1, row2]).save(os.path.join(OUT, "nafnet_threeway_comparison.png"))
    print(f"job1: row1={pick}, row2={r.structure} L{int(r.level)} @({int(r.y)},{int(r.x)})")


# ---------------- job 2 ----------------
def job2():
    cases = pd.read_csv(f"{ROOT}/results/phase1/diagnostics/hallucination_cases_asym1.csv")
    inv = cases[(cases.cls == "invented_undamaged") & (cases.level.isin([1, 2]))
                & (cases.tgt_val < 0.15)].copy()
    inv["vac_rich"] = inv.structure.str.contains("Vacancy|adatom|Adatom")
    inv = inv.sort_values(["vac_rich", "pred_val"], ascending=[False, False])
    for _, r in inv.iterrows():  # need a case the gate actually suppresses
        g = cached("gatedv2_t0.9_d2_s400", r.structure, int(r.level))
        if g[int(r.y), int(r.x)] < 0.15:
            s, t, _, _ = load_pair(r.structure, "A", int(r.level))
            strip([crop_at(a, int(r.y), int(r.x), 192) for a in
                   (s, cached("nafnet_asym1", r.structure, int(r.level)), g, t)],
                  [f"input L{int(r.level)} {r.structure[:22]}", "ungated asym1 (invents)",
                   "gated thr0.9 (source kept)", "target"]).save(
                os.path.join(OUT, "gated_lowdose_demo.png"))
            print(f"job2: {r.structure} L{int(r.level)} @({int(r.y)},{int(r.x)})")
            return
    print("job2: NO case found where gate suppresses an ungated invention")


# ---------------- job 3 ----------------
def job3():
    os.makedirs(VOUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    val = sorted(load_split("val"))
    structs = list(rng.choice(val, size=12, replace=False))

    def net(cfg, ck):
        c = yaml.safe_load(open(cfg))
        k = torch.load(ck, map_location="cuda", weights_only=False)
        m = build_model(c["model"]).cuda()
        m.load_state_dict(k["model"])
        m.eval()
        return m

    ft = net(f"{ROOT}/runs/nafnet_w32_ft_evid_asym1/config.yaml",
             f"{ROOT}/runs/nafnet_w32_ft_evid_asym1/iter_25000.pth")
    sc = net(f"{ROOT}/runs/nafnet_w32_scratch_asym1/config.yaml",
             f"{ROOT}/runs/nafnet_w32_scratch_asym1/iter_100000.pth")

    for lv, name in [(36, f"repair_L36_sample_seed{SEED}.png"),
                     (12, f"midlevel_L12_sample_seed{SEED}.png")]:
        cells = []
        for s_name in structs:
            s, t, _, _ = load_pair(s_name, "A", lv)
            pf = infer_full(ft, s, s.mean(), max(s.std(), 1e-6))
            ps = infer_full(sc, s, s.mean(), max(s.std(), 1e-6))
            cy, cx = most_damaged_center(s, t)
            cells.append(strip([crop_at(a, cy, cx, PANEL) for a in (s, pf, ps, t)],
                               [f"{s_name[:20]} L{lv}", "ft_asym1", "scratch", "target"],
                               size=224))
        grid = vstack([hstack_imgs([cells[r * 4 + c] for c in range(4)]) for r in range(3)])
        grid.save(os.path.join(VOUT, name))
        print(f"job3: {name}")

    # defect sites: 12 random TRUE sites from defect masks, levels 1-2
    sites = []
    attempts = 0
    while len(sites) < 12 and attempts < 200:
        attempts += 1
        s_name = str(rng.choice(val))
        lv = int(rng.choice([1, 2]))
        p = os.path.join(ROOT, "data", "defect_masks", "operation_A",
                         s_name.replace(".png", ".npz"))
        if not os.path.exists(p):
            continue
        dm = np.load(p)["mask"]
        lab, n = cc_label((dm == 1) | (dm == 2))
        if n == 0:
            continue
        i = int(rng.integers(1, n + 1))
        ys, xs = np.nonzero(lab == i)
        sites.append((s_name, lv, int(ys.mean()), int(xs.mean())))
    cells = []
    for s_name, lv, cy, cx in sites:
        s, t, _, _ = load_pair(s_name, "A", lv)
        pf = infer_full(ft, s, s.mean(), max(s.std(), 1e-6))
        ps = infer_full(sc, s, s.mean(), max(s.std(), 1e-6))
        cells.append(strip([crop_at(a, cy, cx, 96) for a in (s, pf, ps, t)],
                           [f"{s_name[:16]} L{lv}", "ft_asym1", "scratch", "target"], size=192))
    grid = vstack([hstack_imgs([cells[r * 4 + c] for c in range(4)]) for r in range(3)])
    grid.save(os.path.join(VOUT, f"defect_sample_seed{SEED}.png"))
    print("job3: defect_sample")


def hstack_imgs(imgs, pad=6):
    H = max(i.size[1] for i in imgs)
    W = sum(i.size[0] for i in imgs) + pad * (len(imgs) - 1)
    out = Image.new("L", (W, H), 30)
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.size[0] + pad
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    job1()
    job2()
    job3()
