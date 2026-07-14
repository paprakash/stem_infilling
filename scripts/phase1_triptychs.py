#!/usr/bin/env python3
"""Render Phase-1 report triptychs: source | NAFNet | Restormer | target,
3 val structures (incl. hard-family vo2) x levels 1, 8, 36, operation A."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_phase1_report import triptychs

ROOT = "/blue/hennig/pawanprakash/ornl_stem"

if __name__ == "__main__":
    triptychs(
        ckpts={
            "nafnet": (f"{ROOT}/runs/nafnet_w32_v1/config.yaml",
                       f"{ROOT}/runs/nafnet_w32_v1/iter_25000.pth"),
            "restormer": (f"{ROOT}/runs/restormer_v1/config.yaml",
                          f"{ROOT}/runs/restormer_v1/iter_25000.pth"),
        },
        structures=["vo2_Vacancy-Anion-03.png", "cose2_Fe_doped.png", "mos2-A_Adatom-Li.png"],
        levels=[1, 8, 36],
        out_dir=f"{ROOT}/results/phase1/triptychs",
    )
