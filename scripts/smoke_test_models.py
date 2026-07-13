#!/usr/bin/env python3
"""Smoke test: instantiate Restormer and NAFNet (1-ch in/out), run one
forward+backward pass on a random 256x256 grayscale batch on the GPU.
"""
import sys
import time

import torch

ROOT = "/blue/hennig/pawanprakash/ornl_stem"


def gpu_sanity():
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  capability sm_{cap[0]}{cap[1]}  torch {torch.__version__}  cuda {torch.version.cuda}")
    a = torch.randn(2048, 2048, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print(f"matmul OK, result mean {b.mean().item():.4f}")


def run_one(name, model, batch=4, size=256):
    model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(batch, 1, size, size, device="cuda")
    y = torch.randn(batch, 1, size, size, device="cuda")
    t0 = time.time()
    out = model(x)
    loss = torch.nn.functional.l1_loss(out, y)
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"{name}: {n_params:.1f}M params, out {tuple(out.shape)}, loss {loss.item():.4f}, "
          f"{grads} grad tensors, fwd+bwd {dt:.2f}s, peak mem {mem:.2f} GiB")
    del model, x, y, out
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def restormer():
    sys.path.insert(0, f"{ROOT}/repos/Restormer")
    from basicsr.models.archs.restormer_arch import Restormer
    m = Restormer(inp_channels=1, out_channels=1)  # paper defaults otherwise
    sys.path.pop(0)
    for k in [k for k in list(sys.modules) if k.startswith("basicsr")]:
        del sys.modules[k]
    return m


def nafnet():
    sys.path.insert(0, f"{ROOT}/repos/NAFNet")
    from basicsr.models.archs.NAFNet_arch import NAFNet
    # NAFNet-width32 config (SIDD denoising variant)
    m = NAFNet(img_channel=1, width=32, middle_blk_num=12,
               enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    sys.path.pop(0)
    for k in [k for k in list(sys.modules) if k.startswith("basicsr")]:
        del sys.modules[k]
    return m


if __name__ == "__main__":
    gpu_sanity()
    run_one("NAFNet-w32", nafnet())
    run_one("Restormer", restormer())
    print("SMOKE TEST PASSED")
