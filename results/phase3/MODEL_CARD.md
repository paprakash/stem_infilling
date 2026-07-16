# Model card — STEM beam-damage restoration (production candidate)

**Model:** `nafnet_w32_ft_evid_asym1` (checkpoint `runs/nafnet_w32_ft_evid_asym1/iter_25000.pth`)
**Recommended 2026-07-16, pending PI sign-off.** All numbers: full val split
(2,052 pairs, 76 structures never seen in training, all 3 operations, 9 levels).

## Recipe

- NAFNet-w32 (29.2M params, 1-channel), global residual, trained 100k iters from
  scratch (Charbonnier + 0.1×FFT-amplitude, 384-px crops, dihedral-only aug,
  per-image source-stat normalization), then fine-tuned 25k iters with:
  - **evidence-split defect weighting**: 10× where the defect site is visible in
    the source, 2× where damage destroyed it (masks v3: vacancy / anomalous
    column / vacuum / protect-dark / GB-line classes);
  - **asymmetric invention penalty 1.0**: relu(pred−target) at dark/vacuum sites.
- Masks: `data/defect_masks/` v3, regenerable via `scripts/make_defect_masks.py`
  (QC'd; known limitations documented in-script).

## Headline numbers (vs identity baseline / vs plain NAFNet)

| level | PSNR | col recall | col RMSE | PSNR damaged | defect preservation | vacuum phantoms |
|---|---|---|---|---|---|---|
| 1 | 33.11 (id 32.12) | 1.000 | 0.46 px | 31.12 (id 25.10) | 0.975 (id 0.996) | 0.005 |
| 8 | 32.04 (id 24.96) | 0.995 | 0.67 px | 28.43 (id 17.11) | 0.972 (id 0.968) | 0.006 |
| 28 | 27.41 (id 19.42) | 0.975 | 1.4 px | 23.92 (id 14.27) | 0.919 (id 0.847) | 0.008 |
| 36 | 26.60 (id 18.22) | 0.969 | 1.6 px | 23.22 (id 13.38) | 0.897 (id 0.838) | 0.009 |

## Operating range (per level)

- **Levels ≥ 8 — recommended range.** Exceeds the identity baseline on every
  committed metric INCLUDING defect preservation.
- **Levels 1–4 — use with the caveats below.** Restoration quality beats
  identity (PSNR +1.0 dB at level 1) but defect preservation trails identity
  (0.975 vs 0.996) and intensity-KL trails (0.030 vs 0.007). At these levels the
  source is nearly clean; bypassing restoration or using Phase-3 gating (which
  converges to identity as damage → 0) is defensible.
- **Texture**: optional matched-noise inference flag (noise synthesized from the
  source's high-pass radial spectrum; deterministic, zero invention risk)
  collapses FFT radial error ~1.1 → 0.14 and KL to 0.026–0.036 at every level.
  Report both with/without.

## Known failure modes (measured rates)

1. **Column invention (vacancy fill)**: ~70 detector-level events per 180
   val-subset images (~2/1000 true columns), concentrated on vacancy-rich
   structures; bright (median intensity ~0.5). Down from 105 (plain NAFNet);
   not zero. THE SCIENCE DIAL: the invention-penalty coefficient trades this
   against high-damage infill — preservation gains and high-damage costs scale
   together (see recipe curve in results/phase2/closeout_100k.md); the PI sets
   the operating point. Penalty 5.0 reaches 0.981 preservation at −3 dB
   damaged-PSNR; the from-scratch retrain (below) reaches 0.984 / 39 inventions
   at −3.7 dB. **The documented path to ~zero is Phase-3 inference gating**
   (predicted-damage mask restricts edits; source copied verbatim elsewhere) —
   spec in closeout_100k.md, not yet built.
2. **Column displacement near grain boundaries**: shifted detections cluster
   2.5–2.9× near the GB seam (no wholesale healing observed). GB-line mask
   class added in v3; residual effect not yet re-audited post-FT.
3. **Texture smoothing**: predictions are smoother than targets (conditional
   mean); KL/FFT gap at low levels. Structure unaffected (undamaged-region
   column recall 1.000, RMSE 0.37–0.46 px). Mitigation: matched-noise flag.
4. **Restored ≠ trustworthy inside large destroyed patches**: infill inside
   ~250-px holes reflects the lattice prior, not evidence; defects inside such
   holes are unknowable (evidence-split weighting stops the model from being
   TRAINED to guess them, but the infill itself remains a prior).

## From-scratch retrain result (documented, NOT recommended)

`nafnet_w32_scratch_asym1` (same recipe from scratch, penalty warmup 20k→40k):
preservation 0.984@1, 39 inventions, vacuum phantoms 0.003 — the cleanest
low-damage model produced — but recall@36 0.937 / damaged-PSNR 20.9 (−3.7 dB):
the recipe shifts the operating point along the trade-off curve rather than
transferring the FT's balance. Violates the no-high-damage-regression rule →
kept as a low-damage-specialist checkpoint; superseded for production by the FT.

## Pipeline notes

- Inference: per-image normalization by source stats; reflect-pad to /16;
  bf16 autocast; ~0.1 s/image on B200.
- Deployment target is CycleGAN-translated inputs (Phase 4 robustness check
  still pending — this card covers clean synthetic sources only).
