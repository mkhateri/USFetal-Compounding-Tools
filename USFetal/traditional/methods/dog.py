import torch
from utils.patch_processing import patchify, unpatchify
from utils.logging_setup import get_logger
import numpy as np
from scipy.ndimage import gaussian_filter

logger = get_logger("DoG Compounding")


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def _fuse_list(arr_list, rule="mean", weights=None, normalize=False):
    """
    Fuse list of same-shaped numpy arrays.
    rule: "mean" | "max" | "absmax" | "weighted"
    weights: used if rule=="weighted" (len == len(arr_list))
    normalize: L2-normalize each array before fusion (energy balancing)
    """
    assert len(arr_list) > 0
    A = np.stack(arr_list, axis=0)  # [M, ...]
    if normalize:
        rms = np.sqrt(np.mean(A**2, axis=tuple(range(1, A.ndim)), keepdims=True) + 1e-8)
        A = A / (rms + 1e-8)

    rule = (rule or "mean").lower()
    if rule == "mean":
        return A.mean(axis=0).astype(np.float32)
    elif rule == "max":
        return A.max(axis=0).astype(np.float32)
    elif rule == "absmax":
        idx = np.argmax(np.abs(A), axis=0)
        return np.take_along_axis(A, np.expand_dims(idx, axis=0), axis=0)[0].astype(np.float32)
    elif rule == "weighted":
        if weights is None or len(weights) != len(arr_list):
            raise ValueError("weights must match #arrays for 'weighted' rule.")
        w = np.asarray(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        fused = np.tensordot(w, A, axes=(0, 0))  # [...shape]
        return fused.astype(np.float32)
    else:
        raise ValueError(f"Unknown fusion rule: {rule}")


# ---------------------------------------------
# Basic DoG (kept for completeness)
# ---------------------------------------------
def difference_of_gaussians(vol, sigma1, sigma2):
    return gaussian_filter(vol, sigma=sigma1) - gaussian_filter(vol, sigma=sigma2)


def multi_scale_dog(vol, sigma_pairs):
    """Sum DoG responses over consecutive scales (band-pass only)."""
    dog_sum = np.zeros_like(vol, dtype=np.float32)
    for (s1, s2) in sigma_pairs:
        dog_sum += difference_of_gaussians(vol, s1, s2)
    return dog_sum


# ---------------------------------------------
# Standard DoG (two fixed sigmas) with view fusion rule
# ---------------------------------------------
def run_dog_standard(views: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Config (with defaults):
      patch_size: (96,96,96)
      overlap: (48,48,48)
      sigma1: 0.5
      sigma2: 2.0
      rule_views: "mean"      # mean|max|absmax|weighted
      view_weights: None      # if weighted
      normalize_views: False
      band_gain: 1.0
      add_mean: True          # add per-voxel mean across views as baseline
    """
    logger.info("Running dog[standard] compounding.")

    B, V, C, D, H, W = views.shape
    patch_size      = config.get("patch_size", (96, 96, 96))
    overlap         = config.get("overlap", (48, 48, 48))
    sigma1          = float(config.get("sigma1", 0.5))
    sigma2          = float(config.get("sigma2", 2.0))
    rule_views      = str(config.get("rule_views", "mean")).lower()
    view_weights    = config.get("view_weights", None)
    normalize_views = bool(config.get("normalize_views", False))
    band_gain       = float(config.get("band_gain", 1.0))
    add_mean        = bool(config.get("add_mean", True))

    patches = patchify(views, patch_size, overlap)
    fused_patches = []

    for patch in patches:
        # [1, V, 1, pD, pH, pW] -> [V, pD, pH, pW]
        x = patch[0, :, 0]

        # DoG per view
        per_view_dog = []
        for v in range(x.shape[0]):
            vol_np = x[v].cpu().numpy()
            dog_np = difference_of_gaussians(vol_np, sigma1, sigma2).astype(np.float32)
            per_view_dog.append(dog_np)

        # Fuse across views according to rule
        band_np = _fuse_list(per_view_dog, rule=rule_views, weights=view_weights, normalize=normalize_views)
        band_np *= band_gain

        # Add brightness anchor (mean across views) if requested
        base_np = x.mean(dim=0).cpu().numpy() if add_mean else 0.0
        fused_np = band_np + base_np

        fused_patch = torch.tensor(fused_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,pD,pH,pW]
        fused_patches.append(fused_patch)

    output_shape = (B, 1, C, D, H, W)
    return unpatchify(
        fused_patches, patch_size, overlap, output_shape,
        weight_method="gaussian", normalization_method="sum"
    )


# ---------------------------------------------
# Multi-scale DoG (configurable fusion rules)
# ---------------------------------------------
def run_dog_multiscale(views: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Multiscale DoG with configurable fusion rules.

    Config (with defaults):
      patch_size: (96,96,96)
      overlap: (48,48,48)
      sigma_scales: [0.5, 1.0, 2.0, 3.0, 4.0]  # consecutive pairs used
      rule_views: "mean"       # fuse across views per scale: mean|max|absmax|weighted
      rule_scales: "sum"       # combine per-scale maps: sum|max|absmax|weighted
      view_weights: None       # if rule_views == "weighted", len == V
      scale_weights: None      # if rule_scales == "weighted", len == len(sigma_scales)-1
      normalize_views: False   # L2 norm before view fusion
      normalize_scales: False  # L2 norm before scale fusion
      band_gain: 1.0           # multiply final band-pass
      add_mean: True           # add per-voxel mean across views
    """
    logger.info("Running dog[multiscale] compounding (coarse to fine, fusion rules).")

    B, V, C, D, H, W = views.shape
    patch_size      = config.get("patch_size", (96, 96, 96))
    overlap         = config.get("overlap", (48, 48, 48))
    sigma_list      = config.get("sigma_scales", [0.5, 1.0, 2.0, 3.0, 4.0])

    rule_views      = str(config.get("rule_views", "mean")).lower()
    rule_scales     = str(config.get("rule_scales", "sum")).lower()
    view_weights    = config.get("view_weights", None)
    scale_weights   = config.get("scale_weights", None)
    normalize_views = bool(config.get("normalize_views", False))
    normalize_scales= bool(config.get("normalize_scales", False))
    band_gain       = float(config.get("band_gain", 1.0))
    add_mean        = bool(config.get("add_mean", True))

    if len(sigma_list) < 2:
        raise ValueError("sigma_scales must have >= 2 values.")
    sigma_pairs = [(sigma_list[i], sigma_list[i+1]) for i in range(len(sigma_list) - 1)]
    logger.info(f"Using multi-scale sigma pairs: {sigma_pairs}")

    patches = patchify(views, patch_size, overlap)
    fused_patches = []

    for patch in patches:
        x = patch[0, :, 0]  # [V, pD, pH, pW]

        # 1) Per-scale: compute DoG per view, then fuse across views
        fused_per_scale = []
        for (s1, s2) in sigma_pairs:
            per_view_dogs = []
            for v in range(x.shape[0]):
                vol_np = x[v].cpu().numpy()
                g1 = gaussian_filter(vol_np, sigma=s1)
                g2 = gaussian_filter(vol_np, sigma=s2)
                per_view_dogs.append((g1 - g2).astype(np.float32))

            fused_scale = _fuse_list(
                per_view_dogs, rule=rule_views, weights=view_weights, normalize=normalize_views
            )
            fused_per_scale.append(fused_scale)

        # 2) Combine per-scale fused maps according to rule_scales
        if rule_scales == "sum":
            band_np = np.sum(np.stack(fused_per_scale, axis=0), axis=0).astype(np.float32)
        else:
            band_np = _fuse_list(
                fused_per_scale, rule=rule_scales, weights=scale_weights, normalize=normalize_scales
            )

        # 3) Global gain on band-pass
        band_np *= band_gain

        # 4) Add brightness anchor (mean across views) if requested
        base_np = x.mean(dim=0).cpu().numpy() if add_mean else 0.0
        fused_np = band_np + base_np

        fused_patch = torch.tensor(fused_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,pD,pH,pW]
        fused_patches.append(fused_patch)

    output_shape = (B, 1, C, D, H, W)
    fused_volume = unpatchify(
        fused_patches, patch_size, overlap, output_shape,
        weight_method="gaussian", normalization_method="sum"
    )
    return fused_volume


# ---------------------------------------------
# Registry
# ---------------------------------------------
METHOD_REGISTRY = {
    "dog[standard]": run_dog_standard,
    "dog[multiscale]": run_dog_multiscale,
}
