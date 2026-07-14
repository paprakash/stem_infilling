"""Training losses: Charbonnier + FFT-amplitude (lattice-periodicity prior)."""
import torch


def charbonnier(pred, target, eps=1e-3, weight=None):
    l = torch.sqrt((pred - target) ** 2 + eps * eps)
    if weight is None:
        return l.mean()
    return (l * weight).sum() / weight.sum()


def fft_amplitude_l1(pred, target):
    """L1 on 2D-FFT magnitudes, orthonormal scaling so values are O(1)."""
    ap = torch.abs(torch.fft.rfft2(pred.float(), norm="ortho"))
    at = torch.abs(torch.fft.rfft2(target.float(), norm="ortho"))
    return (ap - at).abs().mean()


class RestorationLoss(torch.nn.Module):
    def __init__(self, fft_weight=0.1, charb_eps=1e-3):
        super().__init__()
        self.fft_weight = fft_weight
        self.charb_eps = charb_eps

    def forward(self, pred, target, weight=None):
        lc = charbonnier(pred, target, self.charb_eps, weight=weight)
        lf = fft_amplitude_l1(pred, target)
        return lc + self.fft_weight * lf, {"charb": lc.item(), "fft": lf.item()}
