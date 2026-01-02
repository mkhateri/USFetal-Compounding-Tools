import torch
import numpy as np
import logging
from typing import List, Tuple

# =============================================================================
# Patchify, Weight Map & Unpatchify Functions for 6D Volumes
"""
patch_processing.py

This module provides core utilities for patch-based processing of 6D medical volumes.
It includes:
  - patchify(): Split a 6D volume [B, V, C, D, H, W] into overlapping 3D patches.
  - generate_weight_map(): Create smooth weight maps for blending patches.
  - unpatchify(): Reconstruct the full volume from weighted overlapping patches.

These functions are essential for patch-based fusion methods such as PCA or DoG,
where overlapping patches are independently processed and then merged smoothly.

The test block at the end verifies that patchify + unpatchify return a lossless reconstruction.

Used in: compounding.py, methods/pca.py, methods/dog.py, etc.
"""

def patchify(volume, patch_size, overlap):
    """
    Split a 6D volume of shape [B, V, C, D, H, W] into overlapping patches.
    """
    B, V, C, D, H, W = volume.shape
    pD, pH, pW = patch_size
    oD, oH, oW = overlap
    patches = []
    for d in range(0, D, pD - oD):
        for h in range(0, H, pH - oH):
            for w in range(0, W, pW - oW):
                d_start = min(d, D - pD)
                h_start = min(h, H - pH)
                w_start = min(w, W - pW)
                patch = volume[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
                #logger.debug(f"Patch extracted: d_start={d_start}, h_start={h_start}, w_start={w_start}, shape={patch.shape}")
                patches.append(patch)
    return patches

def generate_weight_map(patch_size, overlap, method="gaussian", normalization_method="sum"):
    """
    Generate a smooth transition weight map for merging overlapping patches.
    """
    pD, pH, pW = patch_size
    oD, oH, oW = overlap
    weight_map = np.ones((pD, pH, pW), dtype=np.float32)
    if method == "average":
        return torch.tensor(weight_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    for axis, o in enumerate([oD, oH, oW]):
        if o > 0:
            axis_size = pD if axis == 0 else pH if axis == 1 else pW
            x = np.linspace(0, axis_size - 1, axis_size)
            sigma = o / 3.0
            if method == "gaussian":
                decay = np.exp(-0.5 * ((x - (o - 1)) / sigma) ** 2)
            elif method == "exponential":
                decay = np.exp(-((x - (o - 1)) / sigma))
            elif method == "linear":
                decay = np.linspace(0, 1, o)
                full = np.ones(axis_size)
                full[:o] = decay
                full[-o:] = decay[::-1]
                decay = full
            else:
                decay = np.ones_like(x)
            if axis == 0:
                weight_map *= decay[:, None, None]
            elif axis == 1:
                weight_map *= decay[None, :, None]
            elif axis == 2:
                weight_map *= decay[None, None, :]
    if normalization_method == "sum":
        weight_map /= np.sum(weight_map)
    elif normalization_method == "max":
        weight_map /= weight_map.max()
    return torch.tensor(weight_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def unpatchify(patches, patch_size, overlap, output_shape, weight_method="gaussian", normalization_method="sum"):
    """
    Reconstruct a 6D volume [B, 1, C, D, H, W] from overlapping patches using a smooth weighting map.
    """
    B, _, C, D, H, W = output_shape
    pD, pH, pW = patch_size
    weight_map = generate_weight_map(patch_size, overlap, method=weight_method, normalization_method=normalization_method)
    recon = torch.zeros(output_shape, dtype=patches[0].dtype, device=patches[0].device)
    weight_sum = torch.zeros(output_shape, dtype=patches[0].dtype, device=patches[0].device)
    
    d_indices = [min(d, D - pD) for d in range(0, D, pD - overlap[0])]
    h_indices = [min(h, H - pH) for h in range(0, H, pH - overlap[1])]
    w_indices = [min(w, W - pW) for w in range(0, W, pW - overlap[2])]
    #logger.debug(f"Unpatchify indices: d={d_indices}, h={h_indices}, w={w_indices}")
    
    patch_idx = 0
    for d_start in d_indices:
        for h_start in h_indices:
            for w_start in w_indices:
                #current_slice_shape = (B, 1, C, pD, pH, pW)
                # logger.debug(
                #     f"Unpatchify: d_start={d_start}, h_start={h_start}, "
                #     f"w_start={w_start}, expected slice shape={current_slice_shape}"
                # )
                try:
                    recon[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW] += patches[patch_idx] * weight_map
                    weight_sum[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW] += weight_map
                except Exception as e:
                    # logger.error(f"Error at patch index {patch_idx} with indices d={d_start}, h={h_start}, w={w_start}: {e}")
                    raise
                patch_idx += 1
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    recon /= weight_sum
    return recon

if __name__ == "__main__":
    # Create a dummy 6D volume: [B=1, V=2, C=1, D=64, H=64, W=64]
    dummy_volume = torch.rand(1, 2, 1, 64, 64, 64)

    patch_size = (32, 32, 32)
    overlap = (16, 16, 16)

    print("Running patchify...")
    patches = patchify(dummy_volume, patch_size, overlap)
    print(f"Extracted {len(patches)} patches, each of shape: {patches[0].shape}")

    print("Running unpatchify...")
    output_shape = dummy_volume.shape
    recon = unpatchify(
        patches,
        patch_size,
        overlap,
        output_shape,
        weight_method="gaussian",
        normalization_method="sum"
    )

    print("Reconstruction done.")
    diff = torch.mean(torch.abs(dummy_volume - recon))
    print(f"Mean absolute difference between original and reconstructed volume: {diff.item():.6f}")
