#!/usr/bin/env python3
"""Cache val-subset predictions (op A x 9 levels x 20 structures) for both 25k
checkpoints as .npy under data/cache_phase1_preds/<model>/ — shared by the
Phase-1 diagnostics so inference runs once."""
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import LEVELS, load_pair, val_subset_structures
from train_stem import build_model, infer_full

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
CACHE = os.path.join(ROOT, "data", "cache_phase1_preds")

CKPTS = {
    "nafnet": (f"{ROOT}/runs/nafnet_w32_v1/config.yaml", f"{ROOT}/runs/nafnet_w32_v1/iter_25000.pth"),
    "restormer": (f"{ROOT}/runs/restormer_v1/config.yaml", f"{ROOT}/runs/restormer_v1/iter_25000.pth"),
}

if __name__ == "__main__":
    structures = val_subset_structures()
    for name, (cfg_path, ckpt_path) in CKPTS.items():
        out_dir = os.path.join(CACHE, name)
        os.makedirs(out_dir, exist_ok=True)
        cfg = yaml.safe_load(open(cfg_path))
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model = build_model(cfg["model"]).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        n = 0
        for s_name in structures:
            for lv in LEVELS:
                out = os.path.join(out_dir, f"{s_name.replace('.png', '')}_lvl{lv}.npy")
                if os.path.exists(out):
                    continue
                s, t, mean, std = load_pair(s_name, "A", lv)
                np.save(out, infer_full(model, s, mean, std).astype(np.float32))
                n += 1
        del model
        torch.cuda.empty_cache()
        print(f"{name}: cached {n} preds @ iter {ck['iter']}", flush=True)
