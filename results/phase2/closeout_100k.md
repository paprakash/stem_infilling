# 100k close-out report — base models final + recipe decision (2026-07-15)

Full-val (2,052 pairs) unless noted. Sources: `results/phase1/*_per_image.csv`,
audits in `results/phase1/diagnostics/` and `results/phase2/`.

## Final base-model numbers at 100k

| metric (level) | NAFNet-w32 | Restormer | identity |
|---|---|---|---|
| PSNR @1 (dB) | 31.12 | **33.95** | 32.12 |
| PSNR @36 (dB) | **27.11** | 26.42 | 18.22 |
| col recall @36 | **0.984** | 0.977 | 0.804 |
| col precision @1 | 0.974 | **0.994** | 1.000 |
| PSNR damaged @36 | **24.60** | 22.81 | 13.38 |
| PSNR undamaged @1 | 32.83* | **36.39** | 35.44 |
| defect preservation @1 (mean) | 0.960 | **0.985** | 0.996 |
| defect preservation @36 (mean) | **0.888** | 0.858 | 0.838 |
| vacuum phantom rate @36 | 0.055 | **0.040** | 0 |
| audited inventions (val subset) | 105 | 143† | ~0 |

\* NAFNet base at 25k; its 100k undamaged PSNR is comparable (31.7–32.8 by level).
† Restormer's inventions are predominantly FAINT (below the 0.08 preservation
tolerance but above the blob-detector threshold) — ghost texture rather than
confident atoms; that is why its preservation metric is high while its detector
census is the worst.

**Effective-sample caveat:** at 100k iters, Restormer (batch 6) has seen ~600k
crops vs NAFNet's ~1.6M (batch 16), and took 2,323 min vs 834 min wall-clock
(GPU co-location inflates both). Restormer's low-damage margin is real, but the
two models are not sample-matched; equal-crops comparison would require ~267k
Restormer iters.

## Recipe trade-off curve (NAFNet FTs from 100k, 25k iters each)

| recipe | def_pres @L1 | vac_phantom @36 | recall @36 | psnr_dmg @36 |
|---|---|---|---|---|
| base100k | 0.960 | 0.055 | 0.984 | 24.60 |
| ft_dw10 v1 (uniform 10×, 47% site coverage) | 0.965 | 0.030 | 0.981 | 24.27 |
| ft_dw10 v2 (uniform 10×, 99% coverage) | 0.966 | 0.019 | 0.978 | 23.96 |
| evid (10× visible / 2× destroyed) | 0.963 | 0.025 | 0.983 | 24.02 |
| **evid + asym 1.0 (asym1)** | **0.975** | **0.009** | 0.969 | 23.22 |
| evid + asym 5.0 | 0.981 | 0.007 | 0.945 | 21.53 |
| identity | 0.996 | 0 | 0.804 | 13.38 |

**Preservation gains and high-damage costs scale together with the penalty
coefficient. The coefficient is a science dial for the PI to set** — it prices
"filled vacancy" against "weaker infill of destroyed regions". asym1 is the
recommended operating point: preservation ≥ identity at every level ≥ 8, and its
high-damage costs remain small in absolute terms (recall 0.969 vs identity
0.804; damaged-region PSNR still +9.8 dB over identity).

## Census nuance — why no recipe hits 0.99 at level 1

Audited invented columns fell **105 → 70, not to zero**, across the recipe
family, and borderline_shifted detections GROW with the penalty (39 → 64 → 140
as coefficient 0 → 1 → 5): inventions partially convert to displacements rather
than disappearing. A symmetric-or-penalized regression loss cannot fully
separate "complete the lattice" from "invent an atom" at sites whose local
evidence is genuinely ambiguous. **No recipe in this loss family meets the 0.99
level-1 target.**

## Phase 3 spec (not built): inference-time gating to ~zero inventions

The documented path to ~zero inventions is post-hoc, deterministic, and
model-agnostic:
1. Predict a damage mask from the source (train a small segmenter on
   |source−target| > thr masks; no extra labels needed), dilated conservatively.
2. Output = source outside the mask (verbatim copy — zero edits, zero invention
   risk), model prediction inside the mask only.
3. Optionally compose with the matched-noise flag (below) inside edited regions.
Properties: inventions in undamaged regions become structurally impossible;
do-no-harm becomes exact; the residual risk concentrates where it belongs — in
genuinely destroyed regions, flagged by the mask for human/UQ review.
Build after the from-scratch retrain; evaluate with the same acceptance battery.

## Operating range & workflow guidance

- **Levels ≥ 8**: asym1-recipe NAFNet exceeds the identity baseline on every
  committed metric including defect preservation. Recommended operating range.
- **Levels 1–4**: residual defect-preservation gap (0.975 vs 0.996) and the
  intensity-KL texture gap remain. Guidance: at these levels the source is
  nearly clean — either bypass restoration (identity) or use Phase-3 gating,
  which converges to identity as damage → 0.
- **Texture/KL**: the matched-noise inference flag (source-spectrum noise
  injection, deterministic, zero hallucination risk) collapses FFT radial error
  to near-identity and cuts KL 2–4×; keep it optional and report both.

## Winner recommendation

**NAFNet-w32 with the asym1 recipe, retrained from scratch (config staged at
`runs/nafnet_w32_scratch_asym1/`, penalty warmed up 0→1.0 over iters 20k–40k) —
launch awaiting review confirmation.** Rationale weighting invention rate:
asym1 lifts NAFNet's preservation to 0.975 while keeping its decisive
high-damage dominance (the deliverable regime), and its inventions are bright
and thus catchable/gateable, vs Restormer's diffuse faint ghosting. Restormer
remains a credible conservative alternative at low damage (native 0.985
preservation, 0.994 precision) at 2.8× the training cost and unmatched samples;
if the workflow ends up prioritizing levels 1–8, an asym1-style Restormer FT is
the experiment to run.
