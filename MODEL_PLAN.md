# Working plan (decided 2026-07-12) — do not read before giving an independent recommendation

## Architecture decision

- **Primary: Restormer** (CVPR 2022, ~26M params), trained from scratch. UNet-shaped restoration transformer; channel-wise self-attention gives a global receptive field at every scale — suited to infilling extended damaged patches from surrounding lattice periodicity. Mature BasicSR-ecosystem code; the X-Restormer comparative study (arXiv 2310.11881) found its design generalizes best across restoration tasks.
- **Parallel baseline: NAFNet** (ECCV 2022). Simpler, ~2× cheaper, matches/beats Restormer on classic denoising. Same BasicSR harness → nearly free to run both; if NAFNet ties, keep the cheaper model.
- **From scratch, not fine-tuned**: 16,675 aligned pairs + 256–384 px random crops + 8-fold dihedral augmentation is ample for 20–30M-param nets; pretrained weights encode natural-image photometric noise, not structural lattice corruption. One optional ablation: SwinIR grayscale-denoising warm start.
- **Rejected as base**: pix2pix/CycleGAN (paired data makes them unnecessary; weaker backbones; hallucination risk), LaMa/SD inpainting (need masks we don't have; natural-image priors; hallucination is the worst failure mode for science). LaMa's Fourier-convolution idea is absorbed via the FFT loss.
- **Phase-3 option**: Palette-style conditional diffusion — posterior sampling gives uncertainty maps; audit hallucination carefully. Alternatively deep ensembles (AtomAI-style).

## Losses

Charbonnier/L1 + FFT-amplitude loss (L1 on 2D Fourier magnitudes — enforces lattice periodicity, matches the group's FFT-based evaluation) ± MS-SSIM. Global residual (predict correction to input) so identity on undamaged regions is easy. No adversarial term unless outputs oversmooth.

## Training regime

One blind model on all levels 1–36 jointly; per-level validation throughout. Ablations if joint training struggles: damage-level conditioning (FiLM/embedding); level-subset training (e.g., 12–36 only) to map the operating range. BasicSR, AdamW, cosine LR, ~200–300k iters, batch 16–32/GPU, mixed precision.

## Phases

| Phase | Work | Outcome |
|---|---|---|
| 0 | Data audit (level inventory, pairing integrity, intensity stats); structure-level splits; dataloader; identity baselines per level (metrics of raw source vs. target) | Difficulty map; frozen splits |
| 1 | Train NAFNet + Restormer, L1+FFT loss, all levels; full per-level eval | First breakdown curve; pick backbone |
| 2 | Ablations: loss weights, crop size, level conditioning, level subsets, SwinIR warm start | Best recipe; operating range |
| 3 | Only if needed: light GAN term, deep ensembles, or conditional diffusion for uncertainty | Uncertainty-aware model |
| 4 | Robustness to CycleGAN-translated inputs; matching noise augmentation if needed; package inference for pipeline | Deployable stage |

## Known risks

Pipeline domain shift (deployed input = CycleGAN output, not clean source); ambiguity at high damage (deterministic model averages — flag with uncertainty, don't trust single answers); two damage regimes (additive at low levels, subtractive at high — verify both are handled); actual level coverage per structure unknown until Phase-0 audit.
