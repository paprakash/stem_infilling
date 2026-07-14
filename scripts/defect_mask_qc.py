#!/usr/bin/env python3
"""QC contact sheet for defect masks: target crops with mask overlays.

Overlay colors: vacancy = red disks, anomalous column = yellow, vacuum = blue
tint. Samples ~30 op-A structures across defect types (Defect-Free should show
~no sites; Vacancy-Anion-NN should show ~NN red disks). Writes
results/phase2/defect_mask_qc.png and a per-structure caption CSV.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
TGT = os.path.join(ROOT, "dropbox_data", "STEM_Simulations_Target_BD", "operation_A")
MASKS = os.path.join(ROOT, "data", "defect_masks", "operation_A")
OUT = os.path.join(ROOT, "results", "phase2")
TILE = 300  # center crop, native resolution


def pick_structures():
    inv = pd.read_csv(os.path.join(ROOT, "data", "defect_masks", "defect_mask_inventory.csv"))
    inv = inv[inv.op == "A"]
    picks = []
    for pat in ["Defect-Free", "Vacancy-Anion-03", "Vacancy-Anion-06", "Vacancy-Anion-09",
                "Vacancy-Anion-12", "adatom", "Adatom", "doped", "Doped", "Anti"]:
        sub = inv[inv.structure.str.contains(pat)]
        picks += sub.structure.sample(min(3, len(sub)), random_state=0).tolist()
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:30]


def overlay(t, mask):
    rgb = np.stack([t, t, t], axis=-1)
    v = mask == 1
    rgb[v] = 0.65 * rgb[v] + 0.35 * np.array([1.0, 0.1, 0.1])
    a = mask == 2
    rgb[a] = 0.65 * rgb[a] + 0.35 * np.array([1.0, 0.9, 0.1])
    vac = mask == 3
    rgb[vac] = 0.75 * rgb[vac] + 0.25 * np.array([0.2, 0.4, 1.0])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = pick_structures()
    per_row, pad, cap = 5, 8, 16
    rows = (len(names) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * (TILE + pad) + pad, rows * (TILE + cap + pad) + pad), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    caps = []
    for i, name in enumerate(names):
        t = np.asarray(Image.open(os.path.join(TGT, name)), dtype=np.float32) / 255.0
        z = np.load(os.path.join(MASKS, name.replace(".png", ".npz")))
        mask = z["mask"]
        h, w = t.shape
        cy, cx = h // 2, w // 2
        sl = np.s_[cy - TILE // 2:cy + TILE // 2, cx - TILE // 2:cx + TILE // 2]
        tile = overlay(t[sl], mask[sl])
        r, c = divmod(i, per_row)
        x0, y0 = pad + c * (TILE + pad), pad + r * (TILE + cap + pad)
        sheet.paste(Image.fromarray(tile), (x0, y0 + cap))
        label = f"{name[:34]} vac:{int(z['n_vacancy_sites'])} anom:{int(z['n_anomalous_cols'])}"
        draw.text((x0, y0 + 2), label, fill=(255, 255, 255))
        caps.append(dict(structure=name, n_vacancy=int(z["n_vacancy_sites"]),
                         n_anomalous=int(z["n_anomalous_cols"]), vacuum_frac=float(z["vacuum_frac"])))
    sheet.save(os.path.join(OUT, "defect_mask_qc.png"))
    pd.DataFrame(caps).to_csv(os.path.join(OUT, "defect_mask_qc.csv"), index=False)
    print(f"wrote {OUT}/defect_mask_qc.png ({len(names)} structures)")


if __name__ == "__main__":
    main()
