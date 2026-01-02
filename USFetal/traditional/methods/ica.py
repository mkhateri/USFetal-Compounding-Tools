"""
ica.py

ICA-based compounding strategies for multi-view 3D ultrasound volumes (USFetal Toolbox).

Variants:
- ica[standard]    → global ICA on entire volumes
- ica[patchwise]   → ICA applied patch-by-patch with overlap

Config (all optional; mirrors PCA):
- k: int (default 1)                           # number of ICs to combine in fusion
- weight_mode: "kurtosis"|"uniform"|"custom"   # weighting of first k ICs (default "kurtosis")
- ic_weights: list[float]                      # used if weight_mode == "custom"
- detail_gamma: float >= 0                     # up-weights higher-index ICs via (j+1)**gamma (default 0.0)
- align_sign: bool                             # stabilize IC signs via mixing alignment (default True)
- add_global_mean: bool                        # standard only; add mean over views (default False)
- add_patch_mean: bool                         # patchwise only; add per-patch mean over views (default True)

# Convergence/solver controls
- ncomp_cap: int                               # hard cap on fitted components per (batch|patch) (default 8)
- standardize: bool                            # z-score each view (feature) before ICA (default True)
- restarts: int                                # number of random restarts; pick best non-Gaussianity (default 3)
- ica_max_iter: int                            # FastICA max_iter (default 1000)
- ica_tol: float                               # FastICA tolerance (default 1e-3)
- ica_random_state: int|None                   # base seed for restarts (default 0)
- algorithm: "parallel"|"deflation"            # FastICA algorithm (default "parallel")
- fun: "logcosh"|"exp"|"cube"                  # contrast function (default "logcosh")
- alpha: float                                 # only for logcosh; ~1.0–1.4 typical (default 1.2)

Registered in: methods/registry.py
"""

from __future__ import annotations

import warnings
from typing import Tuple, Optional, List

import numpy as np
import torch
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from utils.logging_setup import get_logger
from utils.patch_processing import patchify, unpatchify

logger = get_logger("ICA Compounding")


# ------------------------------ helpers ------------------------------
def _select_k(k: Optional[int], max_k: int) -> int:
    if k is None:
        return 1
    return max(1, min(int(k), max_k))


def _safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_standardize(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardize features (columns) of X; returns (Xz, mean, std).
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    Xz = (X - mu) / np.clip(sd, eps, None)
    return Xz, mu, sd


def _align_signs_ica(S: np.ndarray, A_T: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align ICA sign ambiguity by making each mixing row positively correlated with ref.
    Parameters
    ----------
    S   : [N, r]   sources
    A_T : [r, V]   mixing^T  (pass A.T so rows enumerate components)
    ref : [V]      view-space reference (e.g., mean over samples)
    Returns
    -------
    S_aligned, A_T_aligned
    """
    S2 = S.copy()
    A2 = A_T.copy()
    for j in range(A2.shape[0]):
        if float(np.dot(A2[j], ref)) < 0.0:
            A2[j] *= -1.0
            S2[:, j] *= -1.0
    return S2, A2


def _kurtosis_abs_excess(s: np.ndarray) -> float:
    """
    Univariate excess kurtosis magnitude for 1D vector s (population moments).
    Robust to near-constant signals.
    """
    s = s.astype(np.float64, copy=False)
    m = s.mean()
    c = s - m
    m2 = np.mean(c * c)
    if m2 <= 1e-12:
        return 0.0
    m4 = np.mean(c * c * c * c)
    kurtosis = m4 / (m2 * m2)
    return abs(kurtosis - 3.0)


def _ic_weights(S: np.ndarray,
                k_sel: int,
                weight_mode: str = "kurtosis",
                ic_weights: Optional[List[float]] = None,
                detail_gamma: float = 0.0) -> np.ndarray:
    """
    Build weights for ICs 1..k_sel.
    - "custom": use provided ic_weights
    - "uniform": equal weights
    - "kurtosis": weight by absolute excess kurtosis of sources (default)
    Then apply detail emphasis: multiply j-th weight by (j+1)**detail_gamma.
    """
    wm = (weight_mode or "kurtosis").lower()
    if wm == "custom" and ic_weights is not None and len(ic_weights) >= k_sel:
        w = np.asarray(ic_weights[:k_sel], dtype=float)
    elif wm == "uniform":
        w = np.ones(k_sel, dtype=float)
    else:  # "kurtosis" default
        w = np.array([_kurtosis_abs_excess(S[:, j]) for j in range(k_sel)], dtype=float)
        if not np.isfinite(w).all() or float(w.sum()) <= 0:
            w = np.ones(k_sel, dtype=float)

    if detail_gamma > 0.0:
        w = w * (np.arange(1, k_sel + 1, dtype=float) ** float(detail_gamma))

    s = float(w.sum())
    return (w / s) if s > 0 else (np.ones(k_sel, dtype=float) / float(k_sel))


def _fit_ica_with_restarts(
    X: np.ndarray,
    V: int,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Fit ICA with restarts; pick the run with (1) no ConvergenceWarning preferred,
    then (2) highest sum(abs excess kurtosis) over the first k_eval sources.

    Returns
    -------
    S_best : [N, r] sources for best run
    A_best : [V, r] mixing for best run
    warned_best : bool, whether the chosen run warned
    """
    # solver controls
    ncomp_cap = int(config.get("ncomp_cap", 8))
    algo = config.get("algorithm", "parallel")
    fun = config.get("fun", "logcosh")
    alpha = float(config.get("alpha", 1.2))
    maxit = int(config.get("ica_max_iter", 1000))
    tol = float(config.get("ica_tol", 1e-3))
    restarts = int(config.get("restarts", 3))
    base_seed = config.get("ica_random_state", 0)
    k_cfg = config.get("k", 1)

    # number of components to fit must be >= desired k, and <= V and <= cap
    r_target = min(V, ncomp_cap)
    r_target = max(r_target, _select_k(k_cfg, V))

    best_S = None
    best_A = None
    best_warned = True
    best_score = -np.inf

    for i in range(max(1, restarts)):
        seed = None if base_seed is None else int(base_seed) + i
        ica = FastICA(
            n_components=r_target,
            algorithm=algo,
            fun=fun,
            fun_args={"alpha": alpha} if fun == "logcosh" else None,
            whiten="unit-variance",
            max_iter=maxit,
            tol=tol,
            random_state=seed,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            S_try = ica.fit_transform(X)  # [N, r]
            A_try = ica.mixing_          # [V, r]
            warned = any(issubclass(ws.category, ConvergenceWarning) for ws in w)

        r_avail = S_try.shape[1]
        k_eval = min(_select_k(k_cfg, r_avail), r_avail)
        score = float(np.sum([_kurtosis_abs_excess(S_try[:, j]) for j in range(k_eval)]))

        # prefer converged runs; tie-break by score
        prefer_tuple = (not warned, score)
        current_best_tuple = (not best_warned, best_score)
        if prefer_tuple > current_best_tuple:
            best_S = S_try
            best_A = A_try
            best_warned = warned
            best_score = score

    return best_S, best_A, best_warned


# ============================================================================
# Global ICA (multi-IC; default k=1)
# ============================================================================
def run_ica_standard(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running ica[standard] compounding.")
    B, V, C, D, H, W = views.shape
    assert C == 1, "Expected single-channel volumes (C=1)."

    # fusion controls
    k_cfg = config.get("k", 1)
    weight_mode = config.get("weight_mode", "kurtosis")
    ic_weights = config.get("ic_weights", None)
    detail_gamma = float(config.get("detail_gamma", 0.0))
    align_sign = bool(config.get("align_sign", True))
    add_global_mean = bool(config.get("add_global_mean", False))
    standardize = bool(config.get("standardize", True))

    device = views.device
    fused_vols = []

    for b in range(B):
        # X: [N, V] with N = D*H*W
        X = views[b, :, 0].reshape(V, -1).T.detach().cpu().numpy()
        X = _safe_nan_to_num(X)

        if not np.any(X):
            fused = torch.zeros((D, H, W), dtype=torch.float32, device=device)
            fused_vols.append(fused.unsqueeze(0))
            continue

        # optional per-view standardization (helps convergence a lot)
        if standardize:
            Xz, _, _ = _safe_standardize(X)
        else:
            Xz = X

        S, A, warned = _fit_ica_with_restarts(Xz, V, config)
        if warned:
            logger.debug("ica[standard]: best run raised ConvergenceWarning (accepted).")

        r = S.shape[1]

        # align signs to a view-space reference (pre-standardization mean over samples)
        if align_sign and A is not None:
            ref = (X if not standardize else X).mean(axis=0)  # same shape either way
            S, A_T = _align_signs_ica(S, A.T, ref)
            A = A_T.T

        k_sel = _select_k(k_cfg, r)
        w = _ic_weights(S[:, :k_sel], k_sel, weight_mode, ic_weights, detail_gamma)  # [k_sel]
        fused_flat = (S[:, :k_sel] @ w)  # [N]

        if add_global_mean:
            fused_flat = fused_flat + X.mean(axis=1)

        fused = torch.from_numpy(fused_flat).to(device=device, dtype=torch.float32).view(D, H, W)
        fused_vols.append(fused.unsqueeze(0))

    return torch.stack(fused_vols, dim=0).unsqueeze(1)  # [B, 1, D, H, W]


# ============================================================================
# Patchwise ICA (multi-IC; default adds patch mean)
# ============================================================================
def run_ica_patchwise(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running ica[patchwise] compounding.")

    B, V, C, D, H, W = views.shape
    patch_size = config.get("patch_size", (96, 96, 96))
    overlap = config.get("overlap", (48, 48, 48))

    # fusion controls
    k_cfg = config.get("k", 1)
    weight_mode = config.get("weight_mode", "kurtosis")
    ic_weights = config.get("ic_weights", None)
    detail_gamma = float(config.get("detail_gamma", 0.0))
    align_sign = bool(config.get("align_sign", True))
    add_patch_mean = bool(config.get("add_patch_mean", True))
    standardize = bool(config.get("standardize", True))

    patches = patchify(views, patch_size, overlap)
    logger.info(f"Total patches extracted: {len(patches)}")

    device = views.device
    fused_patches = []

    for patch in patches:
        # Accept (Bp, V, 1, pD, pH, pW) or (V, 1, pD, pH, pW)
        if patch.dim() == 6:
            pv = patch[0, :, 0]  # [V, pD, pH, pW]
        elif patch.dim() == 5:
            pv = patch[:, 0]     # [V, pD, pH, pW]
        else:
            raise ValueError(f"Unexpected patch dim {patch.dim()} for shape {tuple(patch.shape)}")

        pD, pH, pW = pv.shape[1:]
        X = pv.reshape(pv.shape[0], -1).T.detach().cpu().numpy()  # [N, V]
        X = _safe_nan_to_num(X)

        # short-circuit zero patches (all masked)
        if not np.any(X):
            fused_patch = torch.zeros((pD, pH, pW), dtype=torch.float32, device=device)
            fused_patches.append(fused_patch.unsqueeze(0).unsqueeze(0))
            continue

        if standardize:
            Xz, _, _ = _safe_standardize(X)
        else:
            Xz = X

        S, A, warned = _fit_ica_with_restarts(Xz, V, config)
        if warned:
            logger.debug("ica[patchwise]: best run raised ConvergenceWarning (accepted).")

        r = S.shape[1]

        if align_sign and A is not None:
            ref = (X if not standardize else X).mean(axis=0)
            S, A_T = _align_signs_ica(S, A.T, ref)
            A = A_T.T

        k_sel = _select_k(k_cfg, r)
        w = _ic_weights(S[:, :k_sel], k_sel, weight_mode, ic_weights, detail_gamma)
        fused_flat = (S[:, :k_sel] @ w)

        if add_patch_mean:
            fused_flat = fused_flat + X.mean(axis=1)

        fused_patch = torch.from_numpy(fused_flat).to(device=device, dtype=torch.float32).view(pD, pH, pW)
        fused_patches.append(fused_patch.unsqueeze(0).unsqueeze(0))  # [1,1,pD,pH,pW]

    # Try unpatchify with 6D output_shape first, then 5D fallback.
    output_shape6 = (B, 1, 1, D, H, W)  # (B, V=1, C=1, D, H, W)
    output_shape5 = (B, 1, D, H, W)
    try:
        fused_volume = unpatchify(
            fused_patches, patch_size, overlap, output_shape6,
            weight_method="gaussian", normalization_method="sum"
        )
        logger.info("ica[patchwise]: used 6D output_shape (B, V=1, C=1, D, H, W).")
        return fused_volume
    except Exception as e6:
        logger.warning(f"ica[patchwise]: 6D unpatchify failed ({e6}); trying 5D output_shape...")
        fused_volume = unpatchify(
            fused_patches, patch_size, overlap, output_shape5,
            weight_method="gaussian", normalization_method="sum"
        )
        logger.info("ica[patchwise]: used 5D output_shape (B, 1, D, H, W).")
        return fused_volume


# --------------------------------------------------------------
# Register
# --------------------------------------------------------------
METHOD_REGISTRY = {
    "ica[standard]": run_ica_standard,
    "ica[patchwise]": run_ica_patchwise,
}
