# Running log

## 2026-07-13 — architecture decision + Phase 0

- **Decision**: adopt MODEL_PLAN.md (Restormer primary + NAFNet baseline, from scratch,
  Charbonnier + FFT-amplitude loss, global residual, one blind model on levels 1–36) with
  amendments: 384-px default crops (level-36 damage patches reach ~250 px); NAFNet/Restormer
  ranking settled empirically in Phase 1; balanced sampling across levels; variable image
  sizes handled in dataloader (reflect-pad if < crop); per-level breakdown reported as
  median + IQR (damage non-monotonic per structure); splits stratified by material family,
  eval per family as well as per level.
- **Data audit** (`results/phase0/audit_summary.md`):
  - Layout: `Source_BD/operation_{A,B,C}/beam_damage_{L}/<name>.png` ↔ `Target_BD/operation_{A,B,C}/<name>.png`.
  - 618 structures, 49 material families, 9 levels {1,2,4,8,12,16,20,28,36}; 1,853 targets, 16,675 sources.
  - One integrity problem: `zrte2_Cr_doped` missing op C entirely and levels 28/36 in op B;
    sole member of family zrte2 → assigned to train.
  - Image sizes vary per structure: 456–833 px; none below 384, so 384-crops always fit.
  - Damage: low levels add localized bright deposits (~0.5–2 % pixels); high levels erase
    atomic-column patches up to ~250×250 px (level 36 ≈ 33 % pixels changed on sampled image).
    Non-monotonic per structure (level is dose, realization stochastic); location varies by level.
- **Splits** (`data/splits/`, seed 20260713): structure-level 80/10/10 stratified by family →
  466/76/76 structures = 12,569/2,052/2,052 pairs. Small families (n=3) get 1/1/1 so every
  family (except zrte2) appears in val and test.
- Repos cloned (shallow): swz30/Restormer, megvii-research/NAFNet. Both vendor their own
  `basicsr`; they will NOT be pip-installed (name collision) — we import the arch files
  directly with the repo on sys.path. See envs/INSTALL_NOTES.md.
- **Env `stem`**: Python 3.11.15, torch 2.11.0+cu128 — B200 (sm_100) verified, matmul OK.
  Exported to `envs/environment.yml`. Smoke tests passed for NAFNet-w32 (29.2M) and
  Restormer (26.1M, 1-ch) with NO repo patches needed (arch-file-only imports).
  Restormer fp32 peak 47.8 GiB @ batch 4×256² → use AMP in training.
- **Identity baselines** (`results/phase0/identity_summary.md`, all 16,675 pairs):
  median PSNR 32.2 dB (level 1) → 18.5 dB (level 36); SSIM 0.887 → 0.733; KL 0.007 → 0.132.
  Monotonic in aggregate; test split tracks pooled stats. Hardest families at level 36:
  vo2, cro2, mns2, mnse2, mno2 (oxides/Mn compounds, PSNR ≤ 17).
- **Git**: repo initialized (branch main), remote github.com/paprakash/stem_infilling.
  Tracked: scripts, docs, env exports, split manifests, small results. Untracked:
  dropbox_data/, repos/ (hashes pinned in envs/INSTALL_NOTES.md), checkpoints/logs.
- Phase 0 complete — awaiting review before Phase 1 (dataloader + training).

## 2026-07-13 — pair-count correction + Phase 1 eval commitments

- **Pair-count fix**: split report initially said train=12,569 pairs; correct number is
  12,571 (verified by direct disk count). No data is missing — the manifest's pair count
  used the *intersection* of damage levels across operations, which wrongly dropped
  `zrte2_Cr_doped` op A levels 28 & 36 (present on disk) because op B lacks those two
  levels. `scripts/phase0_splits.py` now counts levels per operation independently and
  the manifest carries an explicit `n_pairs` column. Split membership unchanged
  (train/val/test.txt identical; only the accounting changed).
  Final counts: train=12,571, val=2,052, test=2,052, total=16,675.
## 2026-07-13 — Phase 1 build + gates

- **Harness** (`scripts/`): stem_data.py (per-image source-stat normalization applied to
  both images — invertible at inference where only the source exists; 384 crops;
  reflect-pad guard; dihedral-only aug; uniform pair sampling = level-balanced by
  construction, 617-618 pairs/level), stem_losses.py (Charbonnier eps 1e-3 + 0.1×FFT-amp,
  ortho-normalized), stem_metrics.py (full committed set; damage mask |s-t|>0.06 dilated;
  LoG columns, tol = half median NN spacing), train_stem.py (modes overfit/smoke/train,
  bf16 autocast, AdamW+cosine, ckpt+resume, light val 1k / full val 5k on fixed subset).
- **Val subset** (`data/splits/val_subset.txt`): 20 val structures, one per family,
  hard families vo2 + mno2 force-included; op A × 9 levels = 180 pairs.
- **Both archs keep their built-in global residual** (NAFNet x+inp, Restormer +inp_img).
- **Gate A (single-batch overfit) PASSED both**: NAFNet MSE 1.8e-6 @1500 it; Restormer
  4.8e-6 @4000 it (needed 4000 not 1500 at its lr 3e-4 — loss was still descending, not a
  pathology; fixed by adding cosine decay to overfit mode + more iters). Triptychs in
  results/phase1/overfit/ visually match targets.
- **Gate B NAFNet PASSED**: 2.14 it/s @ batch16×384², 25.7 GiB peak, all metrics × 9
  levels, no NaNs. At 500 it already beats identity at levels ≥16 but LOSES to identity
  at level 1 (29.5 vs 32.1 dB) and fft_err ≫ identity everywhere — watch both at 25k.
- **Identity full-metric baseline** (val, results/phase1/identity_val_per_image.csv):
  col_recall 0.994→0.803 by level; col_precision 1.0 flat; psnr_dmg 25.1→13.4 dB;
  psnr_undmg ~33-35 dB (do-no-harm bar); col_rmse 0.39-0.66 px.

## 2026-07-14 — Phase 1 first report (both models @ iter 25,000, FULL val split, 2,052 pairs)

Artifacts: results/phase1/{report_by_level.csv, report_by_family.csv, report_tables.md,
breakdown_curves.png, triptychs/, *_iter25000_val_per_image.csv}. Trainings continue
(100k total; NAFNet ~85k, Restormer ~26k at time of writing, 2.0 / 0.6 it/s co-located).

- **Key question (beat identity at level 1?)** at 25k: Restormer YES on PSNR (32.34 vs
  32.12) but NO on KL (0.032 vs 0.0065). NAFNet NO on PSNR (31.78) and KL (0.025). BUT
  NAFNet's later checkpoints close it: subset-val lvl-1 PSNR 32.6 @25k → 33.5 @80k.
  KL above identity at low levels for both is the honest miss so far.
- **High damage: decisively beaten** (lvl 36, vs identity): NAFNet PSNR 26.4 (+8.2 dB),
  col_recall 0.983 (id 0.803), col_rmse 1.97 px, psnr_dmg 23.4 (+10.1 dB). Restormer
  PSNR 23.6, col_recall 0.913, col_rmse 3.17 px. NAFNet dominates every metric at
  levels ≥8 at equal iters.
- **Do-no-harm split**: Restormer preserves undamaged pixels at identity level
  (psnr_undmg 35.6 vs id 35.4 @lvl1); NAFNet degrades them (~32.8, −2.6 dB) while
  fixing damaged pixels better (30.1 vs 27.2 @lvl1). Restormer = conservative,
  NAFNet = aggressive. Watch NAFNet's undamaged-region error at 100k.
- **Column precision** both ~0.97 vs identity 1.00 → small spurious/shifted detection
  rate — the hallucination-relevant number, must be tracked (see triptychs: Restormer
  leaves faint ghost texture in big infills at lvl 36; NAFNet's infill is cleaner).
- **FFT radial log-error ≫ identity for both** (~0.85–1.3 vs 0.02–0.12): predictions
  are much smoother than targets — the target's noise texture is not reproduced
  (conditional-mean regression). Structural metrics are fine, so likely acceptable
  for the workflow; if the group wants matched texture, raise fft_weight or add a
  mild high-frequency term in Phase 2. NOTE: identity "wins" this metric trivially
  because source and target share the same noise process.
- **Caveat**: comparison is at equal iters (25k), where Restormer has seen ~2.7× fewer
  patches than NAFNet at equal wall-clock shares (batch 6 vs 16, 0.6 vs 2.0 it/s).
  Final verdict at 100k / equal-effort checkpoints in the Phase-1 close-out.
- Fixed: build_model now purges vendored `basicsr` modules between loads (two models
  in one process, e.g. triptych rendering).

## 2026-07-14 — Phase 1 diagnostics on the 25k predictions (val subset, op A, 180 imgs/model)

Artifacts: results/phase1/diagnostics/ (contact sheets, hallucination_cases.csv,
undamaged_column_metrics.csv, noise_probe.csv). Two of my first-pass diagnostic scripts
had bugs (stats capped by contact-sheet sampling → vo2-biased counts; mask-edge matches
deflating restricted precision) — both fixed, numbers below are from the fixed runs.

### 1. Hallucination audit — REAL but small and localized
Unbiased FP census (NAFNet 747, Restormer 553 FPs across 180 images):
- ~72% (NAFNet) / 46% (Restormer) are blob-detector artifacts at image borders or
  damage-mask boundaries; ~10-22% are shifted/borderline detections of real columns.
- **Genuine inventions (source dark AND target dark, prediction bright ~0.35-0.43):**
  NAFNet 80, Restormer 120. Two sub-modes:
  - **In-lattice vacancy fill — the science problem**: 63 cases each model. The
    periodicity prior fills TRUE anion-vacancy sites; heavily concentrated on
    vacancy-rich structures (nbte2_Vacancy-Anion-12A: 31 NAFNet / 19 Restormer;
    also nbs2_Anti-Metal, tise2_Adatom-Li). Rate ≈ 2-3 per 1000 true columns,
    ~0.35/image, present at ALL damage levels incl. 1-2.
  - Vacuum extension (phantom lattice in dark field-of-view margins): 17 NAFNet /
    57 Restormer; likely crop-away-able in the workflow.
- Verdict: not a blocker at this rate, but it erases exactly the defects the group
  studies → Phase 2 must include a defect-preservation mitigation (defect-site loss
  weighting, source-consistency guard, or confidence masking) and this audit must be
  rerun at 100k and on the final model.

### 2. Undamaged-region column metrics — smoothing unification PARTIALLY confirmed
Restricted to undamaged pixels (median): **u_recall = 1.000 at every level, both
models; u_rmse 0.37-0.46 px** (identity-grade sub-pixel accuracy). u_precision
0.92-0.94 (NAFNet) / 0.94-0.99 (Restormer) — the shortfall is the invention channel
above plus detector artifacts, NOT missing/displaced real structure.
**Conclusion (reframing the open questions):** the KL gap, do-no-harm PSNR gap, and
FFT-spectrum gap ARE one smoothing/texture phenomenon — no real structure is lost in
undamaged regions. The column-precision gap is the one genuinely structural issue
(vacancy infill), and it is separate from smoothing.

### 3. Noise-matching probe — texture gap closable post-hoc, zero hallucination risk
Injecting noise matched to the SOURCE's high-pass radial spectrum (deployment-legal)
into predictions: **FFT radial error collapses 0.84-1.35 → 0.12-0.16** (identity
0.02-0.11) and **KL drops 2-4×** at every level (e.g. Restormer lvl 36: 0.194→0.044;
NAFNet lvl 1: 0.035→0.023; identity lvl 1 = 0.0065). A deterministic post-process
closes most of the texture gap → no loss-engineering needed for it; remaining low-level
KL residual is small. Adopt as an optional inference flag in Phase 2, report both.

## 2026-07-14 — Phase 2 build: defect masks, preservation metric, FT baselines

- **Defect masks** (`data/defect_masks/`, regenerable via scripts/make_defect_masks.py):
  all 1,853 targets, classes vacancy/anomalous-column/vacuum. QC over 3 detector
  iterations (results/phase2/defect_mask_qc.png): Defect-Free structures come out
  clean; Vacancy-Anion-NN nominal sanity 3→3, 6→7, 9→7, 12→10. Known limitations
  (documented in-script): stripe-artifact anomaly FPs on a minority of structures;
  dim-sublattice materials (tis2, ws2 …) get their near-invisible anion sites flagged
  as "vacancies" — intentional for training (dark sites that must stay dark), mildly
  dilutive for the metric.
- **Defect-preservation metric** (stem_metrics.defect_preservation): per-site
  mean|pred−target| ≤ 0.08 within mask disks + vacuum phantom rate (pred−target > 0.1).
  Wired into eval_checkpoint + eval_identity. USE MEANS, NOT MEDIANS (median hides
  the invention tail).
- **Baselines on full val (means)**: identity 0.996 (lvl 1) → 0.838 (lvl 36);
  NAFNet@100k 0.960 (lvl 1) → 0.888 (lvl 36); crossover ~lvl 12-16. NAFNet's ~4%
  defect erasure at low damage = the quantified hallucination problem; NAFNet
  vacuum-phantom rate 2-6% (identity 0). FT acceptance: lvl 1-8 preservation
  ≥ ~0.99, vacuum phantoms ~0, no high-level recall/psnr_dmg give-back.
- **NAFNet@100k full val** (vs @25k): lvl-36 PSNR 27.11 (was 26.40), col_recall
  0.984, col_rmse 1.53 px; lvl-1 PSNR 31.12 — full-val lvl-1 PSNR still below
  identity 32.12 despite subset-val suggesting otherwise; KL 0.034 vs id 0.0065.
  NAFNet is a high-damage specialist; the low-damage margin remains texture-driven.
- **Fine-tunes launched**: nafnet_w32_ft_dw10 (from 100k, 10× defect+vacuum weight,
  25k iters, lr 2.5e-4). Restormer FT queued behind its base run (~40k/100k).

## 2026-07-15 — FT-dw10 result + mask-coverage root cause + FT v2

- **FT dw10 (v1 masks) fell short of acceptance**: defect preservation lvl 1 only
  0.960→0.965 (target ~0.99); vacuum phantoms halved not zeroed; audit inventions
  83→76. No high-level regression (recall 0.984→0.981 @36, psnr_undmg +1.3 dB).
- **Root cause: mask coverage, not weight size.** Only 47.5% of audited invention
  sites fell inside a v1 mask disk — the strict gates that keep the METRIC clean
  (intensity gate, size window, border-band exclusion) leave half the inventable
  sites unweighted in TRAINING.
- **Fix: two mask products in one array.** New class 4 "protect_dark" (training
  only): dark pixels > 0.7·d from any detected column incl. border band. Coverage
  of audited invention sites: **99.0%** (198/200); adds ~7% of pixels to the
  weighted set. Metric still uses strict classes 1/2 (+3 for phantoms) only.
- FT v2 launched: nafnet_w32_ft_dw10v2, same recipe, v2 masks.

- **Phase 1 eval harness scope (committed, not deferred)**: from the first NAFNet/Restormer
  validation onward, the per-level (median+IQR) and per-family breakdowns include ALL of:
  PSNR, SSIM, intensity-histogram KL, 2D-FFT radial spectrum error, **atom-column
  detection precision/recall + position RMSE** (blob detection, e.g. Laplacian-of-Gaussian
  on target vs. prediction with tolerance ~ half the min column spacing), and **separate
  error on damaged vs. undamaged regions** (do-no-harm check; damage mask derived as
  |source − target| > threshold since ground truth is available at eval time).
