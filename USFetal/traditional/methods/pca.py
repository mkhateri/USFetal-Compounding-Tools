"""
pca.py

PCA-based compounding strategies for multi-view 3D ultrasound volumes (USFetal Toolbox).

Variants:
- pca[standard]       → global PCA on entire volumes
- pca[patchwise]      → PCA applied patch-by-patch with overlap

Config (all optional):
- k: int | "auto" | None  (default 1)   # number of PCs to combine; "auto"/None uses EVR threshold
- evr_thresh: float (default 0.95)      # cumulative EVR target when k is auto
- min_k: int (default 1)                # lower bound on auto-k
- max_k: int (default 8)                # upper bound on auto-k (also clamped to V and r)
- ncomp_cap: int (default 16)           # cap PCA fitted components (also clamped to V)

- weight_mode: "evr"|"uniform"|"custom" (default "evr")
- pc_weights: list[float]               # used if weight_mode == "custom"
- detail_gamma: float >= 0              # softly up-weights higher PCs (default 0.0)
- align_sign: bool                      # stabilize PC signs (default True)
- add_global_mean: bool                 # standard only; add mean over views (default False)
- add_patch_mean: bool                  # patchwise only; add per-patch mean over views (default True)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from utils.logging_setup import get_logger
from utils.patch_processing import patchify, unpatchify

logger = get_logger("PCA Compounding")


# ------------------------------ helpers ------------------------------
def _select_k_fixed_or_auto(
    k_cfg, evr: np.ndarray, r: int, V: int, evr_thresh: float, min_k: int, max_k: int
) -> int:
    """
    Decide how many PCs to combine.
    - If k_cfg is int -> clamp to [1, min(r, V, max_k)] and return.
    - If k_cfg is "auto" or None -> choose smallest k with cumEVR >= evr_thresh,
      then clamp to [min_k, min(r, V, max_k)].
    """
    hard_cap = max(1, min(r, V, int(max_k)))
    if isinstance(k_cfg, int):
        return max(1, min(int(k_cfg), hard_cap))

    # auto
    if evr is None or evr.size == 0:
        return max(1, min(int(min_k), hard_cap))

    cumevr = np.cumsum(evr[:hard_cap])
    k_auto = int(np.searchsorted(cumevr, float(evr_thresh)) + 1)  # +1 because indices start at 0
    k_auto = max(int(min_k), min(k_auto, hard_cap))
    return k_auto


def _align_signs(components: np.ndarray, scores: np.ndarray, ref: np.ndarray):
    """
    Flip component/sign if negatively correlated with ref (a view-space vector).
    Stabilizes score signs across patches/batches.
    """
    C = components.copy()
    S = scores.copy()
    for j in range(C.shape[0]):
        if float(C[j] @ ref) < 0.0:
            C[j] *= -1.0
            S[:, j] *= -1.0
    return C, S


def _pc_weights(evr: np.ndarray,
                k_sel: int,
                weight_mode: str = "evr",
                pc_weights: list[float] | None = None,
                detail_gamma: float = 0.0) -> np.ndarray:
    """
    Build weights for PCs 1..k_sel.
    - "custom": use provided pc_weights
    - "uniform": equal weights
    - "evr": explained variance ratio (default)
    Then apply detail emphasis: multiply j-th weight by (j)**detail_gamma.
    """
    wm = (weight_mode or "evr").lower()
    if wm == "custom" and pc_weights is not None and len(pc_weights) >= k_sel:
        w = np.asarray(pc_weights[:k_sel], dtype=float)
    elif wm == "uniform":
        w = np.ones(k_sel, dtype=float)
    else:  # "evr" default
        if evr is None or evr.size < k_sel:
            w = np.ones(k_sel, dtype=float)
        else:
            w = evr[:k_sel].astype(float)

    if detail_gamma > 0.0:
        w = w * (np.arange(1, k_sel + 1, dtype=float) ** float(detail_gamma))

    s = float(w.sum())
    return (w / s) if s > 0 else (np.ones(k_sel, dtype=float) / float(k_sel))


def _pca_ncomp(V: int, ncomp_cap: int | None) -> int:
    if ncomp_cap is None:
        return max(1, min(V, 16))
    return max(1, min(int(ncomp_cap), V, 16))


# ============================================================================
# Global PCA (auto-k supported; default behavior unchanged if k is int)
# ============================================================================
def run_pca_standard(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running pca[standard] compounding.")
    B, V, C, D, H, W = views.shape
    assert C == 1, "Expected single-channel volumes (C=1)."

    # config
    k_cfg = config.get("k", 1)  # int | "auto" | None
    evr_thresh = float(config.get("evr_thresh", 0.95))
    min_k = int(config.get("min_k", 1))
    max_k = int(config.get("max_k", 8))
    ncomp_cap = config.get("ncomp_cap", 16)

    weight_mode = config.get("weight_mode", "evr")
    pc_weights = config.get("pc_weights", None)
    detail_gamma = float(config.get("detail_gamma", 0.0))
    align_sign = bool(config.get("align_sign", True))
    add_global_mean = bool(config.get("add_global_mean", False))  # keep False to match prior code

    device = views.device
    fused_vols = []

    # V == 1: just return the single (masked) view
    if V == 1:
        return views[:, :1]

    for b in range(B):
        # X: [N, V] (N = D*H*W)
        X = views[b, :, 0].reshape(V, -1).T.detach().cpu().numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.any(X):
            fused = torch.zeros((D, H, W), dtype=torch.float32, device=device)
            fused_vols.append(fused.unsqueeze(0))
            continue

        pca = PCA(n_components=_pca_ncomp(V, ncomp_cap))
        scores = pca.fit_transform(X)       # [N, r]
        comps = pca.components_             # [r, V]
        evr = pca.explained_variance_ratio_
        r = scores.shape[1]

        if align_sign:
            ref = X.mean(axis=0)
            comps, scores = _align_signs(comps, scores, ref)

        k_sel = _select_k_fixed_or_auto(k_cfg, evr, r, V, evr_thresh, min_k, max_k)
        w = _pc_weights(evr, k_sel, weight_mode, pc_weights, detail_gamma)  # [k_sel]
        fused_flat = (scores[:, :k_sel] @ w)  # [N]

        if add_global_mean:
            fused_flat = fused_flat + X.mean(axis=1)

        fused = torch.from_numpy(fused_flat).to(device=device, dtype=torch.float32).view(D, H, W)
        fused_vols.append(fused.unsqueeze(0))

    return torch.stack(fused_vols, dim=0).unsqueeze(1)  # [B, 1, D, H, W]


# ============================================================================
# Patchwise PCA (auto-k supported; default = old PC1 + mean_patch)
# ============================================================================
def run_pca_patchwise(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running pca[patchwise] compounding.")

    B, V, C, D, H, W = views.shape
    patch_size = config.get("patch_size", (96, 96, 96))
    overlap = config.get("overlap", (48, 48, 48))

    # config
    k_cfg = config.get("k", 1)  # int | "auto" | None
    evr_thresh = float(config.get("evr_thresh", 0.95))
    min_k = int(config.get("min_k", 1))
    max_k = int(config.get("max_k", 8))
    ncomp_cap = config.get("ncomp_cap", 16)

    weight_mode = config.get("weight_mode", "evr")
    pc_weights = config.get("pc_weights", None)
    detail_gamma = float(config.get("detail_gamma", 0.0))
    align_sign = bool(config.get("align_sign", True))
    add_patch_mean = bool(config.get("add_patch_mean", True))  # True matches your old code

    patches = patchify(views, patch_size, overlap)
    logger.info(f"Total patches extracted: {len(patches)}")

    device = views.device
    fused_patches = []

    # V == 1: trivial — just stitch the single view via unpatchify weights
    if V == 1:
        # Build zero-op fused_patches simply from the first (only) view
        for patch in patches:
            if patch.dim() == 6:
                pv = patch[0, 0, 0]  # [pD, pH, pW]
            elif patch.dim() == 5:
                pv = patch[0, 0]     # [pD, pH, pW]
            else:
                raise ValueError(f"Unexpected patch dim {patch.dim()} for shape {tuple(patch.shape)}")
            fused_patches.append(pv.unsqueeze(0).unsqueeze(0))  # [1,1,pD,pH,pW]
        # stitch
        output_shape6 = (B, 1, 1, D, H, W)
        output_shape5 = (B, 1, D, H, W)
        try:
            return unpatchify(fused_patches, patch_size, overlap, output_shape6,
                              weight_method="gaussian", normalization_method="sum")
        except Exception:
            return unpatchify(fused_patches, patch_size, overlap, output_shape5,
                              weight_method="gaussian", normalization_method="sum")

    for patch in patches:
        # Handle both shapes: (Bp, V, 1, pD, pH, pW) or (V, 1, pD, pH, pW)
        if patch.dim() == 6:
            pv = patch[0, :, 0]  # [V, pD, pH, pW]
        elif patch.dim() == 5:
            pv = patch[:, 0]     # [V, pD, pH, pW]
        else:
            raise ValueError(f"Unexpected patch dim {patch.dim()} for shape {tuple(patch.shape)}")

        pD, pH, pW = pv.shape[1:]
        X = pv.reshape(pv.shape[0], -1).T.detach().cpu().numpy()  # [N, V]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.any(X):
            fused_patch = torch.zeros((pD, pH, pW), dtype=torch.float32, device=device)
            fused_patches.append(fused_patch.unsqueeze(0).unsqueeze(0))
            continue

        pca = PCA(n_components=_pca_ncomp(V, ncomp_cap))
        scores = pca.fit_transform(X)          # [N, r]
        comps = pca.components_                # [r, V]
        evr = pca.explained_variance_ratio_
        r = scores.shape[1]

        if align_sign:
            ref = X.mean(axis=0)
            comps, scores = _align_signs(comps, scores, ref)

        k_sel = _select_k_fixed_or_auto(k_cfg, evr, r, V, evr_thresh, min_k, max_k)
        w = _pc_weights(evr, k_sel, weight_mode, pc_weights, detail_gamma)
        fused_flat = (scores[:, :k_sel] @ w)   # [N]

        if add_patch_mean:
            fused_flat = fused_flat + X.mean(axis=1)

        fused_patch = torch.from_numpy(fused_flat).to(device=device, dtype=torch.float32).view(pD, pH, pW)
        fused_patches.append(fused_patch.unsqueeze(0).unsqueeze(0))  # [1,1,pD,pH,pW]

    # --- unpatchify: try APIs that expect 6D first, then 5D fallback ---
    output_shape6 = (B, 1, 1, D, H, W)  # one fused "view", single channel
    output_shape5 = (B, 1, D, H, W)

    try:
        fused_volume = unpatchify(
            fused_patches, patch_size, overlap, output_shape6,
            weight_method="gaussian", normalization_method="sum"
        )
        logger.info("pca[patchwise]: used 6D output_shape (B, V=1, C=1, D, H, W).")
        return fused_volume
    except Exception as e6:
        logger.warning(f"pca[patchwise]: 6D unpatchify failed ({e6}); trying 5D output_shape...")
        fused_volume = unpatchify(
            fused_patches, patch_size, overlap, output_shape5,
            weight_method="gaussian", normalization_method="sum"
        )
        logger.info("pca[patchwise]: used 5D output_shape (B, 1, D, H, W).")
        return fused_volume


# --------------------------------------------------------------
# Register
# --------------------------------------------------------------
METHOD_REGISTRY = {
    "pca[standard]": run_pca_standard,
    "pca[patchwise]": run_pca_patchwise,
}
