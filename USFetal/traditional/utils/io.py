# USFetal_toolbox/io.py
"""
io.py

Handles loading NIfTI data and masks for multi-view fusion workflows.
Includes:
- load_nifti(): Loads a single NIfTI image.
- load_multi_view(): Loads multiple views (volumes) and optionally applies corresponding masks,
  then stacks them.

Expected folder structure (preferred):
    sample_001_0/
      volumes/
         0000.nii.gz
         0001.nii.gz
         ...
      masks/         
         0000.nii.gz         
         0001.nii.gz         
         ...
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import nibabel as nib
import torch
from utils.logging_setup import get_logger

# Setup shared logger
logger = get_logger(__name__)

def load_nifti(file_path):
    """
    Load a NIfTI file and return its data as a torch tensor along with the affine.
    """
    nii_img = nib.load(file_path)
    data = nii_img.get_fdata(dtype=np.float32)  # shape: [D, H, W]
    affine = nii_img.affine
    return torch.tensor(data), affine

def load_multi_view(folder_path, apply_mask=True):
    """
    Load all views from a folder and return:
      - A tensor of shape [V, 1, D, H, W]
      - The affine from the first volume.
      
    The function first checks for a "volumes" subfolder inside folder_path.
    For each volume file (e.g. 0000.nii.gz), if apply_mask is True it will:
      1. Look for a mask file in a "masks" subfolder (first as "0000.nii.gz", then as "0000_mask.nii.gz").
      2. If the "masks" subfolder does not exist, it looks in the same folder as the volume file.
    
    Parameters:
      folder_path (str or Path): Path to the sample folder.
      apply_mask (bool): Whether to apply masks if available. Default is True.
      
    Returns:
      views_tensor: A tensor of shape [V, 1, D, H, W]
      affine: The affine matrix from the first volume.
    """
    folder = Path(folder_path).resolve()
    volumes_folder = folder / "volumes"
    masks_folder = folder / "masks"

    logger.info(f"Volumes folder: {volumes_folder.resolve()}")
    if masks_folder.exists():
        mask_files = sorted(masks_folder.glob("*.nii.gz"))
        logger.info(f"Mask files in '{masks_folder}': {[f.name for f in mask_files]}")
    else:
        logger.info(f"No masks subfolder found in {folder}")

    # Get all volume files in the volumes folder, sorted by filename.
    volume_files = sorted(volumes_folder.glob("*.nii.gz"))
    if not volume_files:
        raise FileNotFoundError(f"No volume files found in {volumes_folder}")

    views = []
    affines = []

    for vol_file in volume_files:
        # Use splitting to get the base string (e.g., "0000")
        base_str = vol_file.name.split('.')[0]
        # Load the volume
        volume, affine = load_nifti(str(vol_file))
        volume = volume.unsqueeze(0)  # shape: [1, D, H, W]
        affines.append(affine)

        if apply_mask:
            mask_file = None
            if masks_folder.exists():
                mask_file = masks_folder / f"{base_str}.nii.gz"
                #logger.info(f"Checking for mask: {mask_file.resolve()}")
                if not mask_file.exists():
                    mask_file = masks_folder / f"{base_str}_mask.nii.gz"
                    #logger.info(f"Checking for mask with suffix: {mask_file.resolve()}")
            else:
                mask_file = vol_file.parent / f"{base_str}.nii.gz"
                if not mask_file.exists():
                    mask_file = vol_file.parent / f"{base_str}_mask.nii.gz"

            if mask_file.exists():
                mask, _ = load_nifti(str(mask_file))
                mask = mask.unsqueeze(0)  # shape: [1, D, H, W]
                logger.info(f"Found and applied mask for volume: {vol_file.name} -> {mask_file.name}")
                volume = volume * mask
            else:
                logger.info(f"No mask found for volume: {vol_file.name}")
        else:
            logger.info(f"Mask not applied for volume: {vol_file.name} (apply_mask=False)")

        views.append(volume)

    views_tensor = torch.stack(views, dim=0)
    return views_tensor, affines[0]

# Example usage:
if __name__ == "__main__":
    folder_path = "/research/work/mkhateri/USFetal/samples/sample_001_0"
    try:
        views, affine = load_multi_view(folder_path, apply_mask=True)
        print(f"Loaded views: {views.shape}, Affine:\n{affine}")
    except Exception as e:
        print(f"Error loading data: {e}")
