# STEM Beam-Damage Restoration — project summary (v1.0, 2026-07-21)

**Problem.** Restore beam-damaged STEM images of 2D materials to their pristine
structures: remove beam-deposited artifacts, infill atomic columns erased by
damage — WITHOUT inventing atoms or erasing the point defects (vacancies,
adatoms, dopants) the group studies. Slots into the group's pipeline after
CycleGAN dose translation.

**Data.** 618 parent structures (49 TMD/oxide families) × 3 STEM operations ×
9 damage levels {1..36}; 16,675 aligned source→target pairs, grayscale
456–833 px. Low levels add deposits; high levels erase column patches up to
~250 px. Damage realization is stochastic and non-monotonic in the dose label.
Structure-level 80/10/10 split, stratified by family (12,571/2,052/2,052 pairs).

**Final system** (`models_release/`, tag v1.0-blind-gate):
```
source ──► segmenter v2 (NAFNet-w16, hard-negative defect sites)
   │              │ probability map
   │              ▼
   │       hysteresis gate: loose (p>0.5) components containing a
   │       strict seed (p>0.9, ≥400 px) → editable mask (feathered 6 px)
   │              │
   ├──────────────┼────────► outside mask: source VERBATIM
   ▼              ▼
restorer (NAFNet-w32 + evidence-split defect weights 10×/2×
+ asymmetric invention penalty 1.0)   → inside mask (+ optional
matched-noise flag for texture)       → review flag if large unseeded region
```

**Headline numbers** (full val, median unless noted; identity = do-nothing):
| level | PSNR (id) | column recall (id) | PSNR damaged px (id) | defect preservation, mean (id) |
|---|---|---|---|---|
| 1 | 33.1 (32.1) | 1.000 (0.994) | 31.1 (25.1) | 0.993 (0.996) |
| 8 | 32.0 (25.0) | 0.995 (0.953) | 28.4 (17.1) | 0.972 (0.968) |
| 36 | 26.6 (18.2) | 0.969 (0.804) | 23.0 (13.4) | 0.897 (0.838) |

**Invention arc**: 105 (base model, val-subset census) → 70 (asym1 recipe) →
**2 at levels 1–4 / ~24 total (blind gated)** vs oracle gate 1/23. The blind
hysteresis gate matches the level-conditioned oracle within 0.03 dB at level 36
with no dose input.

**Five scientific findings**
1. **Defect erasure grows with training** (inventions 83→105 from 25k→100k
   iters): the periodicity prior strengthens faster than defect fidelity.
2. **Symmetric-loss saturation**: uniform defect-site weighting cannot separate
   fill-vs-erase — 99% site coverage barely beat 47% (−30% inventions cap).
3. **The asymmetric invention penalty is a science dial**: preservation gains
   and high-damage restoration costs scale together with its coefficient
   (0→1→5: preservation 0.960→0.975→0.981; damaged-PSNR@36 24.6→23.2→21.5).
4. **Curriculum effect**: the same recipe from scratch lands at a DIFFERENT
   trade-off point (preservation 0.984 but −3.7 dB high-damage infill) than
   fine-tuning after clean pretraining — recipe balance does not transfer
   across training regimes.
5. **Dose ≠ realized damage**: per-structure damage is non-monotonic in dose,
   and a dose classifier hits only ~0.3 accuracy on low levels — appearance-
   based mask statistics (hysteresis), not dose inference, is the deployable
   signal.

**Open items**: Phase 4 robustness on CycleGAN-translated inputs (external data
dependency — group pipeline outputs needed, no stand-in); PI sign-off on the
invention-penalty coefficient and review-flag bound (both documented dials).
