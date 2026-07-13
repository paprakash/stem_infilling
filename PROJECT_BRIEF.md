# Project brief: reconstructing beam-damaged STEM images

## Context

Our group images 2D materials (e.g., CoSe2 with Fe adatoms) with STEM. A separate, existing CycleGAN stage translates low-dose to high-dose images. This project is the next stage of that pipeline: given an image with electron-beam damage (source), reconstruct the pristine structure (target) — i.e., figure out which atoms should be there and infill them, and remove beam-deposited artifacts. For now we work off a static dataset, not the live pipeline.

## Task (from PI)

Find an architecture — trained from scratch on our data or fine-tuned from a pretrained model — that translates source (beam-damaged) into target (undamaged). It might be an image model; the choice of model/architecture class is part of the task. Perfection is not required: if the model only works above some damage level (e.g., good for 12–36, poor for 1–8), that is still useful — we would just restrict the workflow to the usable range. Training on all damage levels at once is the preferred outcome if it works.

## Data

- Location: `/blue/hennig/pawanprakash/ornl_stem/dropbox_data/` with `source/` and `target/` directories.
- Both contain operations A, B, C: the same parent structures under different STEM operations (random operator on grain boundaries, rotations, etc.). They are paired: source op-A translates to target op-A, etc. — different looks of the same material to multiply the data.
- Source only, per operation: several beam-damage levels (1, 2, 4, … up to 36). Each damaged image is the exact same structure as its target, so the data is fully PAIRED (aligned source→target supervision).
- Counts: ~1,853 target images, ~16,675 source images (≈9 damage levels per target). Grayscale PNG, ~584×674 px.
- Empirical note from example images (cose2_Fe_adatom, levels 1,2,8,16,20 vs target): damage level increases with the number; low levels (1–2) mainly ADD intensity (adatom/interstitial-like deposits), high levels (16–20) REMOVE entire atomic columns over extended patches. Both corruption modes must be handled. No damage mask is available at inference.

## Deliverables

1. A trained model translating damaged → pristine.
2. Per-damage-level evaluation: PSNR/SSIM plus physics metrics (per-image intensity-histogram KL divergence — same metric the group uses for CycleGAN; 2D-FFT radial spectrum error; atom-column detection precision/recall and position RMSE; error on damaged vs. undamaged regions separately).
3. The breakdown curve: performance vs. damage level 1–36, defining the model's usable operating range for the workflow.
4. Eventually: an inference stage that slots into the pipeline after the CycleGAN dose-translation step (so robustness to CycleGAN outputs matters, not just clean synthetic sources).

## Constraints

- Compute: B200 GPU node(s) on HPC, multi-GPU possible. Data size is fixed (no new simulations for now).
- Scientific integrity beats visual quality: a model that hallucinates or misplaces atoms is worse than a blurry one. Uncertainty quantification is a valued extension.
- Train/test split must be by parent structure (all damage levels and all operations of a structure in the same split) to avoid leakage.
