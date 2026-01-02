# metrics/base_metrics.py

"""
base_metrics.py

Collection of 3D image fusion quality metrics:
 - SSIM, PSNR, CC, MSE (volume-wise)
 - Entropy, Mutual Information (slice-wise avg)

Dependencies: MONAI, NumPy, Torch, Scikit-learn, Skimage
"""

import numpy as np
import torch
from monai.metrics import SSIMMetric, PSNRMetric
from sklearn.metrics import mutual_info_score
from .registry import register_metric

# ------------------------------------------------------------------------------
# Volume-level Metrics (3D)
# ------------------------------------------------------------------------------

# @register_metric("ssim")
# def ssim_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
#     """
#     Compute SSIM over a full 3D volume [1, D, H, W] using MONAI.
#     """
#     metric = SSIMMetric(spatial_dims=3)
#     return float(metric(pred.unsqueeze(0), target.unsqueeze(0)).item())


# @register_metric("psnr")
# def psnr_3d(pred, target, data_range=255.0):
#     """
#     Compute PSNR over a full 3D volume [1, D, H, W] using MONAI.
#     """
#     metric = PSNRMetric(max_val=data_range)
#     return float(metric(pred.unsqueeze(0), target.unsqueeze(0)).item())

@register_metric("ssim")
def ssim_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM over a full 3D volume [1, D, H, W] using MONAI.
    """
    pred = pred.to(target.device)
    metric = SSIMMetric(spatial_dims=3)
    return float(metric(pred.unsqueeze(0), target.unsqueeze(0)).item())


@register_metric("psnr")
def psnr_3d(pred: torch.Tensor, target: torch.Tensor, data_range=255.0) -> float:
    """
    Compute PSNR over a full 3D volume [1, D, H, W] using MONAI.
    """
    pred = pred.to(target.device)
    metric = PSNRMetric(max_val=data_range)
    return float(metric(pred.unsqueeze(0), target.unsqueeze(0)).item())


@register_metric("cc")
def cc_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute correlation coefficient (Pearson) between predicted and target 3D volumes.
    """
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    if np.std(pred_np) == 0 or np.std(target_np) == 0:
        return 0.0
    return float(np.corrcoef(pred_np, target_np)[0, 1])

@register_metric("mse")
def mse_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean squared error for 3D volumes.
    """
    return float(torch.nn.functional.mse_loss(pred, target).item())

# ------------------------------------------------------------------------------
# Slice-wise Metrics (Entropy, MI) averaged across volume
# ------------------------------------------------------------------------------

@register_metric("entropy")
def entropy(volume: torch.Tensor, _: torch.Tensor = None) -> float:
    """
    Compute average entropy over D slices in bits (base-2).
    Accepts volume in either [0..1] or [0..255].
      - If max <= 1.0 + eps, we treat it as [0..1] => scale *255.
      - Otherwise we treat it as [0..255] => just clip & cast to uint8.

    volume shape: [1, D, H, W]
    Returns:
        The mean of per-slice entropies (bits).
    """
    eps = 1e-3  # small tolerance to handle minor floating rounding
    vol_np = volume.squeeze(0).cpu().numpy()  # => [D, H, W]
    entropies = []

    # Check if data is already in [0..255] or in [0..1]
    global_max = vol_np.max()
    treat_as_unit_range = (global_max <= 1.0 + eps)

    for i in range(vol_np.shape[0]):
        slice_ = vol_np[i]

        if treat_as_unit_range:
            # scale from [0..1] => [0..255]
            slice_ = (slice_ * 255.0).clip(0, 255).astype(np.uint8)
        else:
            # data is presumably [0..255] already
            slice_ = slice_.clip(0, 255).astype(np.uint8)

        # Build histogram [0..255]
        hist, _ = np.histogram(slice_, bins=256, range=(0, 256), density=True)
        # Drop zero bins
        hist = hist[hist > 0]

        if len(hist) == 0:
            # e.g. entire slice is 0 or a single color
            entropies.append(0.0)
            continue

        slice_entropy = -np.sum(hist * np.log2(hist))
        entropies.append(slice_entropy)

    return float(np.mean(entropies))



@register_metric("mi")
def mutual_information(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
    target_np = (target.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mi_scores = []
    for i in range(pred_np.shape[0]):
        mi = mutual_info_score(pred_np[i].flatten(), target_np[i].flatten())
        mi_scores.append(mi)
    return float(np.mean(mi_scores))