#!/usr/bin/env python3
"""Training harness for STEM beam-damage restoration (NAFNet / Restormer).

Modes:
  --mode overfit   Gate A: overfit one fixed batch; loss must collapse to ~0.
                   Saves source/pred/target PNGs to results/phase1/overfit/.
  --mode smoke     Gate B: cfg-limited short run (500 iters) + one FULL validation
                   (all metrics incl. atom columns) at the end; prints VRAM/throughput.
  --mode train     Full run with checkpointing/resume, light val every val_every,
                   full-metric val every full_val_every.

Both archs carry their own global residual (NAFNet: x+inp; Restormer: +inp_img),
so the network learns a correction to the input by construction.

Usage: python train_stem.py --config runs/<exp>/config.yaml --mode train [--resume]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_data import (LEVELS, PairedSTEMTrain, build_index, load_pair,
                       material_family, val_subset_structures)
from stem_losses import RestorationLoss
from stem_metrics import all_metrics

ROOT = "/blue/hennig/pawanprakash/ornl_stem"


def build_model(mcfg):
    name = mcfg["name"]
    # each repo vendors its own `basicsr`; purge any previously imported copy so
    # two models can be built in one process (e.g. triptych rendering)
    for k in [k for k in list(sys.modules) if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[k]
    sys.path = [p for p in sys.path if not p.startswith(f"{ROOT}/repos/")]
    if name == "nafnet":
        sys.path.insert(0, f"{ROOT}/repos/NAFNet")
        from basicsr.models.archs.NAFNet_arch import NAFNet
        model = NAFNet(img_channel=1, width=mcfg["width"],
                       middle_blk_num=mcfg["middle_blk_num"],
                       enc_blk_nums=mcfg["enc_blks"], dec_blk_nums=mcfg["dec_blks"])
    elif name == "restormer":
        sys.path.insert(0, f"{ROOT}/repos/Restormer")
        from basicsr.models.archs.restormer_arch import Restormer
        model = Restormer(inp_channels=1, out_channels=1, dim=mcfg["dim"],
                          num_blocks=mcfg["num_blocks"], num_refinement_blocks=mcfg["num_refinement_blocks"],
                          heads=mcfg["heads"], ffn_expansion_factor=mcfg["ffn_expansion_factor"],
                          bias=False, LayerNorm_type="WithBias", dual_pixel_task=False)
    else:
        raise ValueError(name)
    return model


def pad_to_multiple(x, m=16):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


@torch.no_grad()
def infer_full(model, src01, mean, std, device="cuda"):
    """src01: [0,1] float32 HxW numpy -> prediction in [0,1], native size."""
    x = torch.from_numpy((src01 - mean) / std)[None, None].to(device)
    x, h, w = pad_to_multiple(x, 16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
    y = y.float()[..., :h, :w].cpu().numpy()[0, 0]
    return np.clip(y * std + mean, 0.0, 1.0)


@torch.no_grad()
def validate(model, pairs, it, csv_path, with_columns=False, device="cuda"):
    model.eval()
    rows = []
    for structure, op, lv in pairs:
        s, t, mean, std = load_pair(structure, op, lv)
        pred = infer_full(model, s, mean, std, device)
        m = all_metrics(s, t, pred, with_columns=with_columns)
        m.update(structure=structure, family=material_family(structure), op=op, level=lv, iter=it)
        rows.append(m)
    model.train()

    import pandas as pd
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("structure", "family", "op", "level", "iter")]
    agg = df.groupby("level")[num_cols].median().round(4)
    nan_report = {c: int(df[c].isna().sum()) for c in num_cols if df[c].isna().any()}

    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    return df, agg, nan_report


def save_triptych(path, src01, pred01, tgt01):
    from PIL import Image
    m = np.concatenate([src01, pred01, tgt01], axis=1)
    Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["overfit", "smoke", "train"], default="train")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to initialize MODEL WEIGHTS from (fresh optimizer/schedule); for fine-tunes")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    exp = cfg["exp"]
    run_dir = os.path.join(ROOT, "runs", exp)
    os.makedirs(run_dir, exist_ok=True)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda"
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{exp}] {cfg['model']['name']} {n_params:.1f}M params, mode={args.mode}", flush=True)

    total_iters = 500 if args.mode == "smoke" else (
        cfg["train"].get("overfit_iters", 1500) if args.mode == "overfit" else cfg["train"]["total_iters"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                            betas=tuple(cfg["optim"]["betas"]),
                            weight_decay=cfg["optim"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters,
                                                       eta_min=cfg["optim"]["min_lr"])
    criterion = RestorationLoss(cfg["loss"]["fft_weight"], cfg["loss"]["charb_eps"],
                                invention_penalty=cfg["loss"].get("invention_penalty", 0.0))

    start_iter = 0
    latest = os.path.join(run_dir, "latest.pth")
    if args.init_from and not (args.resume and os.path.exists(latest)):
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialized weights from {args.init_from} (iter {ck['iter']}); fresh optimizer", flush=True)
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_iter = ck["iter"]
        print(f"resumed from iter {start_iter}", flush=True)

    train_ds = PairedSTEMTrain("train", crop=cfg["data"]["crop"], seed=seed,
                               defect_weight=cfg["loss"].get("defect_weight", 1.0),
                               weight_visible=cfg["loss"].get("defect_weight_visible"),
                               weight_destroyed=cfg["loss"].get("defect_weight_destroyed"))
    print(f"train pairs: {len(train_ds)}", flush=True)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["data"]["batch"], shuffle=True,
        num_workers=cfg["data"]["workers"], pin_memory=True, drop_last=True,
        persistent_workers=cfg["data"]["workers"] > 0)

    # fixed stratified val subset: 20 structures x all 9 levels, operation A
    val_pairs = build_index(val_subset_structures(), ops=("A",))
    print(f"val subset pairs: {len(val_pairs)}", flush=True)

    if args.mode == "overfit":
        batch = next(iter(loader))
        sb, tb = batch[0][:4].to(device), batch[1][:4].to(device)  # weight map unused in gate
        # decay LR so the loss can settle instead of oscillating at constant LR
        ofit_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_iters, eta_min=cfg["optim"]["lr"] * 1e-3)
        for it in range(total_iters):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(sb)
            loss, parts = criterion(pred.float(), tb.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ofit_sched.step()
            if it % 100 == 0 or it == total_iters - 1:
                with torch.no_grad():
                    mse = torch.mean((pred.float() - tb.float()) ** 2).item()
                print(f"overfit it {it}: loss {loss.item():.5f} charb {parts['charb']:.5f} "
                      f"fft {parts['fft']:.5f} batch-mse {mse:.2e}", flush=True)
        out_dir = os.path.join(ROOT, "results", "phase1", "overfit")
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(sb).float().cpu().numpy()
        for i in range(sb.shape[0]):
            # crops are in normalized space; rescale each panel to [0,1] for display
            def norm01(a):
                return (a - a.min()) / max(a.max() - a.min(), 1e-6)
            save_triptych(os.path.join(out_dir, f"{exp}_sample{i}.png"),
                          norm01(sb[i, 0].cpu().numpy()), norm01(pred[i, 0]),
                          norm01(tb[i, 0].cpu().numpy()))
        final_mse = torch.mean((torch.from_numpy(pred).to(device).float() - tb.float()) ** 2).item()
        print(f"GATE A {'PASSED' if final_mse < 1e-4 else 'FAILED'}: final batch MSE {final_mse:.2e} "
              f"(threshold 1e-4); triptychs in {out_dir}", flush=True)
        return

    # smoke / train loop
    metrics_csv = os.path.join(run_dir, "val_metrics.csv")
    it = start_iter
    t0 = time.time()
    n_seen = 0
    model.train()
    use_weights = (cfg["loss"].get("defect_weight", 1.0) != 1.0
                   or cfg["loss"].get("defect_weight_visible") is not None)
    full_inv_penalty = cfg["loss"].get("invention_penalty", 0.0)
    use_inv = full_inv_penalty > 0
    # optional warmup [start_iter, full_iter]: penalty 0 until start, linear to
    # full by full_iter — early reconstruction learning is not warped (the FT
    # trade-off curve justifies the endpoint; see NOTES.md 2026-07-15)
    inv_warmup = cfg["loss"].get("invention_penalty_warmup")
    while it < total_iters:
        for sb, tb, wb, ib in loader:
            if it >= total_iters:
                break
            if use_inv and inv_warmup:
                w0, w1 = inv_warmup
                criterion.invention_penalty = full_inv_penalty * min(1.0, max(0.0, (it - w0) / max(w1 - w0, 1)))
            sb, tb = sb.to(device, non_blocking=True), tb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True) if use_weights else None
            ib = ib.to(device, non_blocking=True) if use_inv else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(sb)
            loss, parts = criterion(pred.float(), tb.float(), weight=wb, inv_sites=ib)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            n_seen += 1

            if it % cfg["train"]["log_every"] == 0:
                ips = n_seen / (time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"it {it}/{total_iters} loss {loss.item():.4f} charb {parts['charb']:.4f} "
                      f"fft {parts['fft']:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{ips:.2f} it/s peak {mem:.1f} GiB", flush=True)

            do_full = (it % cfg["train"]["full_val_every"] == 0) or (args.mode == "smoke" and it == total_iters)
            do_light = it % cfg["train"]["val_every"] == 0
            if do_full or do_light:
                _, agg, nans = validate(model, val_pairs, it, metrics_csv, with_columns=do_full)
                print(f"--- val @ it {it} (full={do_full}) ---", flush=True)
                print(agg[[c for c in ("psnr", "ssim", "kl", "fft_err", "psnr_dmg", "psnr_undmg",
                                       "col_precision", "col_recall", "col_rmse") if c in agg.columns]]
                      .to_string(), flush=True)
                if nans:
                    print(f"NaN counts: {nans}", flush=True)

            if it % cfg["train"]["ckpt_every"] == 0 or it == total_iters:
                state = {"iter": it, "model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "cfg": cfg}
                torch.save(state, latest + ".tmp")
                os.replace(latest + ".tmp", latest)
                if it % cfg["train"]["milestone_every"] == 0:
                    torch.save(state, os.path.join(run_dir, f"iter_{it}.pth"))

    print(f"DONE at iter {it}; {(time.time() - t0) / 60:.1f} min, "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
