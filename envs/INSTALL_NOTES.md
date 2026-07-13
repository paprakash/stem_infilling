# Install notes — env `stem`

Target: B200 (Blackwell, sm_100) → needs torch ≥ 2.7 with cu128 wheels.

## Environment

- `conda create -n stem python=3.11`
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- `pip install einops numpy pandas scikit-image pillow opencv-python-headless pyyaml tqdm tabulate matplotlib ptflops`

## Repo strategy (decided 2026-07-13)

Restormer and NAFNet each vendor their own `basicsr` package (diverged forks of the same
name). Installing both via `python setup.py develop` would collide in site-packages, and
their pinned basicsr/torch versions predate Blackwell. Instead:

- Neither repo is pip-installed, and `repos/` is not tracked in git. Pinned clones
  (shallow, cloned 2026-07-13):
  - `swz30/Restormer` @ `68dc6ac472db26f16361150cb7a96a1bc87da93f`
  - `megvii-research/NAFNet` @ `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Model architectures are imported directly with the repo root prepended to `sys.path`
  (`Restormer/basicsr/models/archs/restormer_arch.py`, `NAFNet/basicsr/models/archs/NAFNet_arch.py`).
- Training/eval harness is our own (scripts/), not the repos' BasicSR pipelines — we need
  custom losses (FFT-amplitude), per-level validation, and per-image normalization anyway.

## Patches applied

**None needed** (verified 2026-07-13). With the arch-file-only import strategy, both
`restormer_arch.py` (needs only torch + einops) and `NAFNet_arch.py` (needs only torch +
the repo's `arch_util`/`local_arch`, pure-torch) import and run cleanly on
torch 2.11.0+cu128 / Python 3.11. Neither repo contains the removed
`torchvision.transforms.functional_tensor` import. The repos' full BasicSR training
pipelines were NOT exercised and would need patches if ever used.

## Smoke test (B200, 2026-07-13)

`scripts/smoke_test_models.py`: sm_100 detected, matmul OK.
- NAFNet-w32 (SIDD config): 29.2M params, fwd+bwd @ 4x1x256x256 OK, peak 4.8 GiB.
- Restormer (paper defaults, 1-ch): 26.1M params, fwd+bwd OK, peak 47.8 GiB (fp32) —
  plan AMP + modest batch for training.
