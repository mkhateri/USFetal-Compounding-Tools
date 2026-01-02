# methods/arithmetic.py

"""
arithmetic.py

Arithmetic-based compounding strategies for multi-view 3D ultrasound volumes (USFetal Toolbox).

Variants:
- mean[standard]    → average across views
- median[standard]  → median across views
- max[standard]     → max across views

Registered in: methods/registry.py
"""

import torch
from utils.logging_setup import get_logger

logger = get_logger("Arithmetic Compounding")


def run_mean_standard(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running mean[standard] compounding.")
    return views.mean(dim=1, keepdim=True)  # [B, 1, C, D, H, W]


def run_median_standard(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running median[standard] compounding.")
    return views.median(dim=1, keepdim=True).values  # Only keep the tensor (drop indices)


def run_max_standard(views: torch.Tensor, config: dict) -> torch.Tensor:
    logger.info("Running max[standard] compounding.")
    return views.max(dim=1, keepdim=True).values


# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------
METHOD_REGISTRY = {
    "mean[standard]": run_mean_standard,
    "median[standard]": run_median_standard,
    "max[standard]": run_max_standard,
}
