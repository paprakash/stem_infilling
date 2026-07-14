#!/usr/bin/env python3
"""Phase 1 report: per-level and per-family tables (median+IQR) for both models
side by side with the identity baseline, breakdown-curve figures, and triptychs.

Inputs: per-image CSVs produced by eval_checkpoint.py and eval_identity.py.
Outputs under results/phase1/: report tables (csv+md), breakdown_curves.png,
triptychs/.

Usage: make_phase1_report.py --nafnet <csv> --restormer <csv> --identity <csv>
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/blue/hennig/pawanprakash/ornl_stem"
OUT = os.path.join(ROOT, "results", "phase1")
LEVELS = [1, 2, 4, 8, 12, 16, 20, 28, 36]

METRICS = ["psnr", "ssim", "kl", "fft_err", "col_precision", "col_recall",
           "col_rmse", "psnr_dmg", "psnr_undmg"]
METRIC_LABELS = {
    "psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "kl": "intensity-KL ↓",
    "fft_err": "FFT radial log-error ↓", "col_precision": "column precision ↑",
    "col_recall": "column recall ↑", "col_rmse": "column RMSE (px) ↓",
    "psnr_dmg": "PSNR damaged px (dB) ↑", "psnr_undmg": "PSNR undamaged px (dB) ↑",
}
# dataviz reference palette: categorical slots 1-2; identity = muted reference
COLORS = {"nafnet": "#2a78d6", "restormer": "#1baf7a", "identity": "#898781"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"


def med_iqr(df, by):
    g = df.groupby(by)[METRICS]
    return (g.median().add_suffix("_med")
            .join(g.quantile(0.25).add_suffix("_q25"))
            .join(g.quantile(0.75).add_suffix("_q75"))
            .join(g.size().rename("n")))


def side_by_side(tables, by):
    """tables: {name: per-image df} -> wide table indexed by `by`."""
    parts = []
    for name, df in tables.items():
        t = med_iqr(df, by)
        t.columns = [f"{name}.{c}" for c in t.columns]
        parts.append(t)
    return pd.concat(parts, axis=1)


def compact_md(tables, by, metrics=("psnr", "ssim", "kl", "col_recall", "psnr_undmg")):
    """Readable md: median [q25-q75] strings, models as column groups."""
    rows = []
    idx = None
    for name, df in tables.items():
        t = med_iqr(df, by)
        idx = t.index if idx is None else idx
        for m in metrics:
            fmt = "{:.3f}" if t[f"{m}_med"].abs().max() < 10 else "{:.1f}"
            rows.append(pd.Series(
                [f"{fmt.format(v)} [{fmt.format(a)}–{fmt.format(b)}]"
                 for v, a, b in zip(t[f"{m}_med"], t[f"{m}_q25"], t[f"{m}_q75"])],
                index=t.index, name=f"{m} | {name}"))
    out = pd.concat(rows, axis=1)
    return out[[c for m in metrics for c in out.columns if c.startswith(f"{m} |")]]


def breakdown_figure(tables, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), facecolor=SURFACE)
    for ax, m in zip(axes.ravel(), METRICS):
        ax.set_facecolor(SURFACE)
        for name, df in tables.items():
            t = med_iqr(df, "level").reindex(LEVELS)
            c = COLORS[name]
            ls = "--" if name == "identity" else "-"
            ax.plot(LEVELS, t[f"{m}_med"], ls, color=c, lw=2, label=name,
                    marker="o" if name != "identity" else None, ms=4)
            if name != "identity":
                ax.fill_between(LEVELS, t[f"{m}_q25"], t[f"{m}_q75"], color=c, alpha=0.13, lw=0)
        ax.set_title(METRIC_LABELS[m], fontsize=11, color=INK)
        ax.set_xlabel("damage level", fontsize=9, color=MUTED)
        ax.set_xticks(LEVELS)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle("Phase 1: restoration quality vs damage level — median (line), IQR (band), identity baseline (dashed)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    print(f"wrote {path}")


def triptychs(ckpts, structures, levels, out_dir):
    """ckpts: {name: (config_path, ckpt_path)}; renders src|pred_nafnet|pred_restormer|target."""
    import torch
    import yaml
    from PIL import Image
    from stem_data import load_pair
    from train_stem import build_model, infer_full
    os.makedirs(out_dir, exist_ok=True)
    preds = {}
    for name, (cfg_path, ckpt_path) in ckpts.items():
        cfg = yaml.safe_load(open(cfg_path))
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model = build_model(cfg["model"]).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        for s_name in structures:
            for lv in levels:
                s, t, mean, std = load_pair(s_name, "A", lv)
                preds[(name, s_name, lv)] = infer_full(model, s, mean, std)
        del model
        torch.cuda.empty_cache()
    for s_name in structures:
        for lv in levels:
            s, t, _, _ = load_pair(s_name, "A", lv)
            panels = [s] + [preds[(n, s_name, lv)] for n in ckpts] + [t]
            m = np.concatenate(panels, axis=1)
            base = s_name.replace(".png", "")
            Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"{base}_lvl{lv}_src_" + "_".join(ckpts) + "_tgt.png"))
    print(f"wrote triptychs to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nafnet", required=True)
    ap.add_argument("--restormer", required=True)
    ap.add_argument("--identity", required=True)
    args = ap.parse_args()

    tables = {"nafnet": pd.read_csv(args.nafnet),
              "restormer": pd.read_csv(args.restormer),
              "identity": pd.read_csv(args.identity)}

    os.makedirs(OUT, exist_ok=True)
    wide_lvl = side_by_side(tables, "level")
    wide_lvl.to_csv(os.path.join(OUT, "report_by_level.csv"))
    wide_fam = side_by_side(tables, "family")
    wide_fam.to_csv(os.path.join(OUT, "report_by_family.csv"))

    md = ["# Phase 1 report — models vs identity baseline (median [IQR])", "",
          "## By damage level", "", compact_md(tables, "level").to_markdown(), "",
          "## By material family", "", compact_md(tables, "family").to_markdown(), ""]
    with open(os.path.join(OUT, "report_tables.md"), "w") as fh:
        fh.write("\n".join(md))
    print("wrote report tables")

    breakdown_figure(tables, os.path.join(OUT, "breakdown_curves.png"))


if __name__ == "__main__":
    main()
