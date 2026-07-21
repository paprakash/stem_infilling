# STEM Beam-Damage Restoration — project root

**STATUS: COMPLETE except Phase 4** (CycleGAN-robustness; blocked on external data —
group-pipeline CycleGAN outputs, do not simulate a stand-in). Deployed system: see
`PROJECT_SUMMARY.md`, `results/phase3/MODEL_CARD.md`, release bundle in
`models_release/` (+ copy at /blue/hennig/pawanprakash/stem_release_v1.0), tag
v1.0-blind-gate. Running log: NOTES.md.

Restore beam-damaged STEM images (source) to their pristine structures (target): remove beam-deposited artifacts and infill missing atomic columns. Supervised paired image-to-image restoration. Full context in `PROJECT_BRIEF.md`; working plan in `MODEL_PLAN.md`.

## Data (READ-ONLY — never modify)

- `dropbox_data/` → `source/` and `target/`, each with operations A/B/C (paired "looks" of the same parent structures).
- ~1,853 targets, ~16,675 sources. Source has beam-damage levels 1, 2, 4 … up to 36 per structure; target is the same structure undamaged. Grayscale PNG, ~584×674 px.
- Damage semantics: low levels mostly ADD intensity (adatom-like deposits); high levels REMOVE atomic columns over extended patches.
- Verify the actual folder/filename layout empirically before writing the dataloader (expected pattern like `name_<level>.png` ↔ `name.png`, but audit first).
- All derived/preprocessed data goes under `data/`, never inside `dropbox_data/`.

## Non-negotiable ML rules

- Train/val/test split by PARENT STRUCTURE: all damage levels AND all operations (A/B/C) of a structure stay in the same split. Splitting by image leaks near-duplicates.
- Evaluate per damage level, always. The headline deliverable is the breakdown curve (metrics vs. level 1–36).
- Physics metrics alongside PSNR/SSIM: intensity-histogram KL, 2D-FFT radial spectrum error, atom-column precision/recall + position RMSE, and separate error on damaged vs. undamaged regions (do-no-harm check).
- Per-image normalization; dihedral augmentation only (flips/90° rotations). No elastic or intensity distortions.

## Directory conventions

```
ornl_stem/
├── dropbox_data/   # raw pairs (read-only)
├── repos/          # cloned model repos
├── data/           # splits, manifests, preprocessed patches
├── runs/           # one subdir per experiment: config, ckpts, logs
├── results/        # eval tables, breakdown curves, figures
└── envs/           # environment.yml exports, install notes
```

## Environment & jobs (B200 node, CC runs in a screen terminal)

- B200 = Blackwell (sm_100): needs recent PyTorch with CUDA 12.8+ wheels (torch ≥ 2.7 / cu128). Older restoration repos (Restormer/NAFNet pin old basicsr/torch) WILL need dependency fixes — expect this, patch the repo env rather than downgrading torch below Blackwell support.
- Use conda env `stem` (export to `envs/environment.yml` after changes).
- Never run training in the foreground. Launch: `nohup python train.py ... > runs/<exp>/train.log 2>&1 &` then poll the log. Always enable checkpointing + resume.
- Escalation ladder before any long run: (1) GPU sanity (`torch.cuda.get_device_name`, small matmul), (2) overfit a single batch, (3) ~500-iter smoke test, (4) full run.
- One experiment = one `runs/<name>/` with the exact config committed. No untracked hyperparameter changes.

## Style

- Ask before installing anything large or deleting anything. Keep a running `NOTES.md` log of decisions and results (date, experiment, outcome, next step).
