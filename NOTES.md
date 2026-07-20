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

## 2026-07-15 — GB audit + evidence-conditioned weighting build

- **GB audit** (results/phase2/gb_strips.png, gb_clustering.csv): most targets carry a
  vertical registry-shift seam near center (+ sometimes horizontal). REINTERPRETATION:
  the "scan-line stripe" that confused the anomaly detector in mask QC is (at least
  often) the GB seam itself — flagging columns on it was arguably correct.
  Verdict: **wholesale healing NOT confirmed** — seams survive in predictions even
  under heavy damage (tate2/tas2 L36 restore the seam correctly). But
  borderline-shifted detections cluster **2.5-2.9× near the seam** (both base100k and
  ft): column-placement wobble = healing pressure without erasure. Invented columns
  are a bulk phenomenon (clustering 0.67), not GB-driven.
- **Audit baseline correction**: base-100k inventions = 105 (vs 83 at 25k — more
  training made it worse). FT-dw10's true effect: 105 → 76 (−28%).
- **Masks v3 plan**: + class 5 "gb_line" (training-only): deepest central-band dip of
  row/col median profile, depth-gated ≥0.06, band ±0.4d. NOT regenerated yet — the
  running ft_dw10v2 reads masks lazily; regen only after it completes to keep its
  protect-dark ablation clean.
- **Evidence-conditioned weighting implemented** (configs ready, launch after dw10v2
  acceptance): defect site ∧ visible-in-source → 10×; defect site ∧ destroyed → 2×
  (damage mask on the fly in the dataloader). Plus optional asymmetric invention
  penalty: 5 × mean(relu(pred−target)) at classes {1,3,4} — runs as separate ablation
  (nafnet_w32_ft_evid / _evid_asym).
- **dw10-v1 hedging check @ lvl 36** (base100k → ft_dw10): defect preservation
  +0.010 (0.888→0.898) at cost psnr_dmg −0.33 dB (24.60→24.27), col_recall −0.003.
  Mild hedging signature — the evidence split exists to eliminate exactly this.
- Decision rule (user): winning recipe = best low-level defect preservation subject to
  no high-damage regression; goes into the final from-scratch retrain.

## 2026-07-15 — FT-dw10v2 acceptance: symmetric weighting saturates

Full-val means, base100k → ft_v1 (47% coverage) → ft_v2 (99% coverage):
- defect preservation lvl 1: 0.9596 → 0.9653 → 0.9663 (identity 0.9955). Coverage was
  NOT the bottleneck either — 10× symmetric Charbonnier saturates at ~−30% inventions
  (audit: 105 → 76 → 73). Filling a vacancy and erasing a column cost the same under a
  symmetric loss; the conditional-mean pull wins.
- vacuum phantoms respond well: 0.0194 → 0.0090 → 0.0073 (lvl 1); 0.055 → 0.030 → 0.019 (36).
- Hedging cost grows with weighted area: lvl-36 psnr_dmg 24.60 → 24.27 → 23.96,
  recall 0.9844 → 0.9811 → 0.9778. Uniform weights inside destroyed regions teach
  hedging — the evidence split exists to remove exactly this.
- Masks v3 (with GB class 5) generated after ft_v2 finished (lazy-read safety);
  nafnet_w32_ft_evid and _evid_asym launched from 100k on v3 masks.

## 2026-07-15 — evidence FTs evaluated: the trade-off curve brackets the answer

Full-val means/medians (def_pres@1 | recall@36 | psnr_dmg@36):
base 0.960|0.984|24.60 → ft_v2 0.966|0.978|23.96 → evid 0.963|0.983|24.02 →
evid_asym(5) 0.981|0.945|21.53. Identity bar: 0.996|0.804|13.4.
- Evidence split WORKS as designed: recall@36 recovered to 0.983 (hedging gone) but
  symmetric weighting still doesn't lift preservation.
- Asymmetric penalty is the ONLY lever that moves preservation (0.963→0.981, vacuum
  phantoms →0.006) but coeff 5 violates the no-regression rule (−3 dB psnr_dmg@36,
  recall −0.04, borderline_shifted 39→140 = global under-filling).
- Interpolation launched: nafnet_w32_ft_evid_asym1 (evid + invention_penalty 1.0).
  Decision after its battery; winning recipe → final from-scratch retrain.

## 2026-07-15 — asym1 interpolation: the recipe trade-off curve is complete

def_pres@1 | vac_ph@36 | recall@36 | psnr_dmg@36 (full val):
base 0.960|0.055|0.984|24.60 · evid 0.963|0.025|0.983|24.02 ·
**asym1 0.975|0.009|0.969|23.22** · asym5 0.981|0.007|0.945|21.53 · id 0.996|0|0.804|13.38
- Preservation gain and high-damage cost scale together with the penalty coefficient —
  no free lunch in this loss family; the coefficient IS the science dial.
- asym1 ≥ identity on defect preservation at every level ≥ 8; residual gap only at 1-4.
- Recommendation for the from-scratch retrain: **evid + invention_penalty 1.0 (asym1)**
  — largest preservation gain whose absolute costs stay small (recall@36 0.969 ≫ id
  0.804; psnr_dmg still +9.8 dB over id). Strict no-regression alternative: evid.
  Decision deferred to review at 100k close-out.

## 2026-07-15 — 100k CLOSE-OUT (see results/phase2/closeout_100k.md for full report)

- Restormer@100k: PSNR@1 33.95 (beats identity 32.12), native defect preservation
  0.985@1 / precision 0.994 — best low-damage model; but PSNR@36 26.42 < NAFNet 27.11,
  preservation@36 0.858 < 0.888, and 143 audited (faint) inventions. Effective-sample
  caveat: 600k crops vs NAFNet 1.6M; 2,323 vs 834 min wall-clock.
- Winner: NAFNet + asym1 from-scratch retrain (staged, launch gated on review).
- Census nuance, Phase-3 inference-gating spec, operating-range guidance, and the
  matched-noise flag are all recorded in the close-out report.

## 2026-07-16 — from-scratch retrain result + MODEL CARD (results/phase3/MODEL_CARD.md)

- scratch_asym1 @100k: def_pres@1 0.984 / 39 inventions / vac_ph 0.003 (all best) BUT
  recall@36 0.937, psnr_dmg@36 20.9 (−3.7 dB vs base) — the recipe from scratch slides
  ALONG the trade-off curve to a more conservative point instead of transferring the
  FT balance. Tracker's auto-pause never fired (rule required all three gates below;
  def_pres stayed above throughout — one-sided divergence).
- Decision per rule (no high-damage regression): **production candidate =
  nafnet_w32_ft_evid_asym1**; scratch kept as low-damage-specialist checkpoint.
- Matched-noise probe on the candidate: KL → 0.026–0.036, FFT → 0.13–0.17 at all levels.
- Model card written (recipe, masks v3, per-level operating range, failure modes with
  rates, science-dial statement, Phase-3 gating spec). Pending PI sign-off.

## 2026-07-16 — GB re-audit + Phase 3 build + Phase 4 dependency flag

- **GB re-audit verdict: the gb_line weight class did NOT earn its place** —
  borderline-shifted clustering near seams is 2.63× for ft_evid_asym1 vs 2.54× at
  base100k (unchanged within noise). Keep class 5 in masks (harmless, tiny area) but
  it is not a working mitigation; GB displacement remains an open failure mode
  (documented in model card).
- **Phase 3 gating in progress**: damage_seg_v1 (NAFNet-w16, 30k iters, BCE pos_weight
  5) training on dilate(|s−t|>0.06) masks, train split only. Gate evaluator
  (scripts/eval_gated.py): source verbatim outside predicted mask (thr 0.3 + dilate 4,
  conservative), ft_evid_asym1 inside, optional matched-noise in edited regions;
  reports standard battery + segmenter P/R per level + miss-rate cost
  (psnr_missed_dmg) + frac_edited.
- **Phase 4 external data dependency (flagged, do not simulate)**: CycleGAN-robustness
  evaluation requires CycleGAN-translated source images from the group's dose-translation
  pipeline stage. Not available in this repo; no stand-in will be fabricated without
  discussion. Blocking item for the Phase-4 row of MODEL_PLAN.md.

## 2026-07-16 — Phase 3 gate sweep verdict: premise falsified for segmenter v1

- Sweep (12 settings, full val): high-damage side is essentially free at thr ≤0.7
  (psnr_dmg −0.13 dB, recall preserved); tight settings converge at low damage
  (def_pres@1 0.990, 0.6% edited at thr 0.95+size400) — but **inventions never drop**
  (lvl 1-4: 23 vs 23 ungated at the tightest setting). The segmenter's confident
  low-damage detections ARE the vacancy sites: trained to exclude them (|s−t| small
  there) yet generalizes "dark anomaly = damage". Zero-invention guarantee holds only
  if segmenter precision holds at invention-prone sites — it doesn't in v1.
- Also: gating inflates borderline_shifted (paste seams create detection shifts:
  189→698 as thr rises) — cosmetic but worth knowing.
- Model card updated: gated mode evaluated, NOT recommended as-built; v2 path spec'd
  (hard-negative defect sites in segmenter training — defect masks provide the labels).
- Paused for review before building segmenter v2.

## 2026-07-17 — Segmenter v2 + feathered gate: level-conditioned gating RECOMMENDED

- **Hard negatives worked**: defect-site FP (the guarantee metric, now reported per
  level) 0.15-0.19 → 0.019-0.07 at tight thresholds; lvl 1-4 inventions 23 → 0 at
  thr 0.9-0.95+s400 with def_pres@1 = 0.9955 = identity and ≤0.5 % pixels edited.
  Total census 70 → 15 (pure tight) / 28 (adaptive).
- **No single setting meets both checks** (tight costs −2.1..−3.7 dB @36). Adaptive
  probe-area rule evaluated and REJECTED: segmenter over-prediction variance makes
  lvl-1 probe areas span 7-92 % — no separating threshold exists.
- **Resolution: level-conditioned gate** (damage level = known acquisition parameter):
  thr 0.9+s400 at levels 1-4, thr 0.5+s400 at ≥8 (psnr_dmg@36 −0.24 dB, recall 0.970).
  Model card gated section rewritten accordingly; v1 verdict kept as historical.
- Feathering: borderline_shifted 698 → 165-330; still 2.6-5× ungated (blend-zone
  detection shifts) — cosmetic residual, documented.
- Stopped here per instruction; next decisions (PI): adopt level-conditioned gated
  mode, science-dial setting, Phase-4 CycleGAN data.

- **Phase 1 eval harness scope (committed, not deferred)**: from the first NAFNet/Restormer
  validation onward, the per-level (median+IQR) and per-family breakdowns include ALL of:
  PSNR, SSIM, intensity-histogram KL, 2D-FFT radial spectrum error, **atom-column
  detection precision/recall + position RMSE** (blob detection, e.g. Laplacian-of-Gaussian
  on target vs. prediction with tolerance ~ half the min column spacing), and **separate
  error on damaged vs. undamaged regions** (do-no-harm check; damage mask derived as
  |source − target| > threshold since ground truth is available at eval time).
