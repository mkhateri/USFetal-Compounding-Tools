import torch

def patchify_multiview(volume_6d, patch_size, overlap):
    """
    Patchify a 6D volume: shape [B, V, C, D, H, W].

    We produce a list of patches, each of shape [B, V, C, pD, pH, pW].
    For each dimension (D, H, W) we move in steps of (patch_size - overlap).

    Returns:
      list of Tensors, each -> [B, V, C, pD, pH, pW].
    """
    B, V, C, D, H, W = volume_6d.shape
    pD, pH, pW = patch_size
    oD, oH, oW = overlap

    patches = []
    for d in range(0, D, pD - oD):
        d_start = min(d, D - pD)
        for h in range(0, H, pH - oH):
            h_start = min(h, H - pH)
            for w in range(0, W, pW - oW):
                w_start = min(w, W - pW)

                patch = volume_6d[
                    :,
                    :,
                    :,
                    d_start : d_start + pD,
                    h_start : h_start + pH,
                    w_start : w_start + pW
                ]  # => [B, V, C, pD, pH, pW]

                patches.append(patch)
    return patches

def unpatchify_multiview(patches, patch_size, overlap, output_shape):
    """
    Reconstruct a 6D volume [B, V, C, D, H, W] from patches [B, V, C, pD, pH, pW].
    `patches` should be a stacked Tensor: shape [num_patches, B, V, C, pD, pH, pW].
    We'll accumulate them with weighting and then divide to account for overlaps.

    output_shape => (B, V, C, D, H, W)
    """
    B, V, C, D, H, W = output_shape
    pD, pH, pW = patch_size
    oD, oH, oW = overlap

    # We'll create an empty volume & a weight map to accumulate overlapping patches
    recon = torch.zeros(output_shape, dtype=patches.dtype, device=patches.device)
    weight = torch.zeros(output_shape, dtype=patches.dtype, device=patches.device)

    idx = 0
    for d in range(0, D, pD - oD):
        d_start = min(d, D - pD)
        for h in range(0, H, pH - oH):
            h_start = min(h, H - pH)
            for w in range(0, W, pW - oW):
                w_start = min(w, W - pW)

                # shape => [B, V, C, pD, pH, pW]
                patch = patches[idx]
                recon[
                    :,
                    :,
                    :,
                    d_start : d_start + pD,
                    h_start : h_start + pH,
                    w_start : w_start + pW
                ] += patch

                weight[
                    :,
                    :,
                    :,
                    d_start : d_start + pD,
                    h_start : h_start + pH,
                    w_start : w_start + pW
                ] += 1.0
                idx += 1

    # Avoid division by zero
    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    recon /= weight
    return recon