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
- **Phase 1 eval harness scope (committed, not deferred)**: from the first NAFNet/Restormer
  validation onward, the per-level (median+IQR) and per-family breakdowns include ALL of:
  PSNR, SSIM, intensity-histogram KL, 2D-FFT radial spectrum error, **atom-column
  detection precision/recall + position RMSE** (blob detection, e.g. Laplacian-of-Gaussian
  on target vs. prediction with tolerance ~ half the min column spacing), and **separate
  error on damaged vs. undamaged regions** (do-no-harm check; damage mask derived as
  |source − target| > threshold since ground truth is available at eval time).
