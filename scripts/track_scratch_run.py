#!/usr/bin/env python3
"""Milestone tracker for the from-scratch asym1 retrain.

At each new iter_{N}k milestone (N multiple of 5k), evaluates levels {1,36} of
the FULL val split and appends def_pres@1 / recall@36 / psnr_dmg@36 to
runs/<exp>/scratch_tracking.csv, comparing against the ft_evid_asym1 endpoint
(0.9747 / 0.969 / 23.22, full-val).

Pause rule (user, 2026-07-15): if at any milestone >= 60k the run is materially
below the FT endpoint on ALL THREE (def_pres@1 < 0.965, recall@36 < 0.960,
psnr_dmg@36 < 22.7), kill the training, write PAUSED.flag, and exit — report
rather than burn to 100k. Exits normally when train.log shows DONE.
"""
import glob
import os
import re
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
EXP = "nafnet_w32_scratch_asym1"
RUN = os.path.join(ROOT, "runs", EXP)
PY = "/home/pawanprakash/miniconda3/envs/stem/bin/python"
FT_ENDPOINT = {"def_pres_1": 0.9747, "recall_36": 0.969, "psnr_dmg_36": 23.22}
PAUSE_GATES = {"def_pres_1": 0.965, "recall_36": 0.960, "psnr_dmg_36": 22.7}
TRACK = os.path.join(RUN, "scratch_tracking.csv")


def eval_milestone(ckpt):
    out_dir = os.path.join(ROOT, "results", "phase3_tracking")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([PY, os.path.join(ROOT, "scripts", "eval_checkpoint.py"),
                    "--config", os.path.join(RUN, "config.yaml"), "--ckpt", ckpt,
                    "--split", "val", "--levels", "1", "36", "--workers", "10",
                    "--out", out_dir], check=True, capture_output=True)
    it = re.search(r"iter_(\d+)", ckpt).group(1)
    df = pd.read_csv(os.path.join(out_dir, f"{EXP}_iter{it}_val_per_image.csv"))
    g1, g36 = df[df.level == 1], df[df.level == 36]
    return dict(iter=int(it),
                def_pres_1=float(g1.defect_preserved_frac.mean()),
                recall_36=float(g36.col_recall.median()),
                psnr_dmg_36=float(g36.psnr_dmg.median()))


def main():
    seen = set()
    if os.path.exists(TRACK):
        seen = set(pd.read_csv(TRACK)["iter"].tolist())
    while True:
        done = False
        log = os.path.join(RUN, "train.log")
        if os.path.exists(log):
            with open(log) as fh:
                done = any(l.startswith("DONE at iter") for l in fh)
        for ckpt in sorted(glob.glob(os.path.join(RUN, "iter_*.pth"))):
            it = int(re.search(r"iter_(\d+)", ckpt).group(1))
            if it in seen:
                continue
            row = eval_milestone(ckpt)
            seen.add(it)
            row["ft_gap_def"] = round(row["def_pres_1"] - FT_ENDPOINT["def_pres_1"], 4)
            row["ft_gap_recall"] = round(row["recall_36"] - FT_ENDPOINT["recall_36"], 4)
            row["ft_gap_psnr"] = round(row["psnr_dmg_36"] - FT_ENDPOINT["psnr_dmg_36"], 3)
            pd.DataFrame([row]).to_csv(TRACK, mode="a", header=not os.path.exists(TRACK), index=False)
            print(f"[track] {row}", flush=True)
            if it >= 60000 and all(row[k] < PAUSE_GATES[k] for k in PAUSE_GATES):
                subprocess.run(["pkill", "-f", f"train_stem.py --config runs/{EXP}"], check=False)
                with open(os.path.join(RUN, "PAUSED.flag"), "w") as fh:
                    fh.write(f"paused at iter {it}: below FT endpoint on all three gates\n{row}\n")
                print(f"[track] PAUSED at {it}: below all three gates", flush=True)
                return
        if done:
            print("[track] training DONE; tracker exiting", flush=True)
            return
        time.sleep(600)


if __name__ == "__main__":
    main()
