#!/usr/bin/env python
"""
main.py

Traditional USFetal Compounding Pipeline Runner.

This script loads multi-view fetal ultrasound volumes and applies
traditional image compounding (fusion) methods to generate a single
fused volume per sample.

Supported compounding methods include:
- PCA-based fusion (global and patch-wise)
- ICA-based fusion (global and patch-wise)
- Difference-of-Gaussians (DoG)
- Variational fusion methods

By default, all parameters are loaded from the configuration file
(`configs/config_traditional.py`). Command-line arguments, when provided,
override the corresponding configuration values.

For each sample, the pipeline performs the following steps:
1. Loads multi-view volumes (optionally applying a mask).
2. Applies the selected compounding methods.
3. Saves the fused volume(s) as NIfTI files.
4. Logs detailed execution information and summary results.

Usage examples:

Example 1: Run a single PCA (standard) compounding method
    python traditional/main.py --methods pca:standard

Example 2: Run multiple compounding methods
    python traditional/main.py --methods pca:standard ica:patchwise variational:dog

Example 3: Run PCA patch-wise compounding with custom paths and patch settings
    python traditional/main.py \
        --methods pca:patchwise \
        --data_parent ./data \
        --output_dir ./output \
        --patch_size 96 96 96 \
        --overlap 48 48 48
"""


import sys
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from utils.logging_setup import get_logger
from utils.io import load_multi_view
from utils.utils import save_nifti, save_middle_slice_as_png, convert_to_8bit, build_summary_table
from methods.registry import METHOD_REGISTRY

from configs.config_traditional import config


# ----------------------------------------------------------------------
# Logger Setup
# ----------------------------------------------------------------------
detail_logger = None
summary_logger = None

def setup_loggers(output_dir: Path):
    """Initialize loggers and return them."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    detail_log_path = log_dir / "compounding.log"
    summary_log_path = log_dir / "compounding_summary.log"

    detail = get_logger("compounding_details", log_file=str(detail_log_path))
    summary = get_logger("compounding_summary", log_file=str(summary_log_path))
    return detail, summary


# ----------------------------------------------------------------------
# Core Pipeline
# ----------------------------------------------------------------------
def run_pipeline(config: Dict):
    global detail_logger, summary_logger
    data_folder = Path(config["data_folder"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if detail_logger is None or summary_logger is None:
        detail_logger, summary_logger = setup_loggers(output_dir)

    sample_name = data_folder.name
    detail_logger.info(f"Processing sample: {sample_name}")
    detail_logger.info("Loading multi-view volume...")

    views, affine = load_multi_view(data_folder, apply_mask=config.get("apply_mask", True))
    detail_logger.info(f"Loaded shape => {views.shape}")

    if views.ndim == 5:
        views = views.unsqueeze(0)

    num_views = views.shape[1]

    # Fuse each method
    for base_method, variant in config["selected_methods"]:
        method_key = f"{base_method}[{variant}]"
        detail_logger.info(f"\nRunning: {method_key}")

        compounding_fn = METHOD_REGISTRY.get(method_key)
        if not compounding_fn:
            detail_logger.warning(f"Method not found: {method_key}")
            continue

        method_cfg = {**config, **config["methods"].get(base_method, {}).get(variant, {})}

        try:
            fused = compounding_fn(views, method_cfg)
            fused = squeeze_to_4d(fused)
        except Exception as e:
            detail_logger.error(f"Fusion failed: {method_key} => {e}")
            continue

        # Save output
        vol_np = fused.to(torch.float32).cpu().numpy()
        vol_3d = np.squeeze(vol_np, axis=0)
        vol_uint8 = convert_to_8bit(vol_3d)

        fused_name = f"{sample_name}_{base_method}_{variant}.nii.gz"
        # fused_png = f"mid_slice_{sample_name}_{base_method}_{variant}.png"

        # save_middle_slice_as_png(vol_3d, output_dir / fused_png)
        save_nifti(vol_uint8, fused_name, output_dir, affine)
        detail_logger.info(f"Saved => {fused_name}")


def squeeze_to_4d(t: torch.Tensor) -> torch.Tensor:
    """
    Squeeze out extra dimensions so that 't' ends up shape [1, D, H, W].
    """
    while t.dim() > 4:
        t = t.squeeze(0)
    return t

# ----------------------------------------------------------------------
# Process All Subjects
# ----------------------------------------------------------------------
def run_all_samples(config: Dict):
    global detail_logger, summary_logger
    data_parent = Path(config["data_parent"]).resolve()
    sample_dirs = sorted({d.resolve() for d in data_parent.iterdir() if d.is_dir()})

    output_dir = Path(config["output_dir"])
    if detail_logger is None or summary_logger is None:
        detail_logger, summary_logger = setup_loggers(output_dir)

    detail_logger.info(f"Found {len(sample_dirs)} samples in {data_parent}")
    for sample_dir in sample_dirs:
        config["data_folder"] = str(sample_dir)
        run_pipeline(config)


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def parse_methods(tokens):
    """Parse method tokens like: pca, pca:patchwise, variational:dog"""
    pairs = []
    for t in tokens:
        if ":" in t:
            m, v = t.split(":", 1)
            pairs.append((m.strip(), v.strip()))
        else:
            pairs.append((t.strip(), "standard"))
    return pairs

def validate_methods(cfg: dict, pairs):
    """Ensure user-specified methods exist in config."""
    valid = []
    for m, v in pairs:
        if m not in cfg["methods"]:
            raise ValueError(f"Unknown method '{m}'. Available: {list(cfg['methods'].keys())}")
        if v not in cfg["methods"][m]:
            raise ValueError(f"Unknown variant '{v}' for '{m}'. Available: {list(cfg['methods'][m].keys())}")
        valid.append((m, v))
    return valid

def override_cfg(cfg: dict,
                 data_parent=None, output_dir=None,
                 patch_size=None, overlap=None,
                 methods_parsed=None,
                 no_mask=False):
    """Shallow override of keys in the traditional config."""
    out = {**cfg}
    if data_parent is not None:
        out["data_parent"] = data_parent
    if output_dir is not None:
        out["output_dir"] = output_dir
    if patch_size is not None:
        out["patch_size"] = tuple(patch_size)
    if overlap is not None:
        out["overlap"] = tuple(overlap)
    if methods_parsed:
        out["selected_methods"] = methods_parsed
    if no_mask:
        out["apply_mask"] = False
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USFetal Compounding Runner")

    # methods
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Select specific methods, e.g. pca, pca:patchwise, variational:dog",
    )

    # paths & tiling
    parser.add_argument("--data_parent", type=str, default=None,
                        help="Parent folder containing subject_* directories")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs/logs")
    parser.add_argument("--patch_size", type=int, nargs=3, default=None,
                        metavar=("DX","DY","DZ"))
    parser.add_argument("--overlap", type=int, nargs=3, default=None,
                        metavar=("OX","OY","OZ"))


    # data loading
    parser.add_argument("--no_mask", action="store_true",
                        help="Disable mask application on load")

    args = parser.parse_args()

    # parse methods if provided
    methods_parsed = parse_methods(args.methods) if args.methods else None
    if methods_parsed:
        # validate requested methods against config
        methods_parsed = validate_methods(config, methods_parsed)

    # merge CLI overrides into config (keeping config defaults for anything not passed)
    cfg = override_cfg(
        config,
        data_parent=args.data_parent,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        methods_parsed=methods_parsed,
        no_mask=args.no_mask,
    )

    # optional: quick sanity print
    print("[Traditional] data_parent:", Path(cfg["data_parent"]).resolve())
    print("[Traditional] output_dir :", Path(cfg["output_dir"]).resolve())
    print("[Traditional] methods    :", cfg.get("selected_methods"))

    run_all_samples(cfg)





