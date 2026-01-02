import os
import torch
import numpy as np
import nibabel as nib
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class InferenceDataLoader:
    """
    Loads multi-view ultrasound or MRI data for inference.
    """
    def __init__(self, data_config):
        self.data_dir = (
            Path(data_config["data_dir"])
            .expanduser()
            .resolve()
        )

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Inference data directory does not exist: {self.data_dir}"
            )

        self.resize_enabled = data_config.get("resize_enabled", False)
        self.resize_shape = data_config.get("resize_shape", [256, 256, 256])

        # dtype handling
        requested_dtype = data_config["dtype"]
        if requested_dtype == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = getattr(torch, requested_dtype)


    def get_sample_list(self):
        """Retrieve list of all sample folders."""
        return sorted([f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))])

    def load_nifti(self, file_path):
        """
        Load a `.nii.gz` file as a tensor and return its affine.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata(dtype=np.float32)
        affine = nii_img.affine  # <-- This is the original affine
        data = torch.tensor(data, dtype=self.dtype)
        if data.ndimension() == 3:
            data = data.unsqueeze(0)
        # (Optional) Resize if enabled…
        return data, affine


class InferenceDataset(Dataset):
    """
    Inference dataset for a SINGLE sample where data_dir itself is the ID folder.
    Expected structure:
      data_dir/
        ├── volumes/
        └── masks/
    """
    def __init__(self, loader: InferenceDataLoader):
        self.loader = loader

        self.sample_path = str(self.loader.data_dir)
        self.sample_name = Path(self.sample_path).name

        self.volumes_dir = os.path.join(self.sample_path, "volumes")
        self.masks_dir = os.path.join(self.sample_path, "masks")

        if not os.path.isdir(self.volumes_dir):
            raise ValueError(f"Missing 'volumes/' in {self.sample_path}")

        self.volume_files = sorted(glob.glob(os.path.join(self.volumes_dir, "*.nii.gz")))
        if len(self.volume_files) == 0:
            raise ValueError(f"No volume files found in {self.volumes_dir}")

    def __len__(self):
        # SINGLE sample inference
        return 1

    def __getitem__(self, idx):
        views, affines = [], []

        for vf in self.volume_files:
            v, affine = self.loader.load_nifti(vf)
            views.append(v)
            affines.append(affine)

        views = torch.stack(views, dim=0)  # [V, 1, D, H, W]

        masks = None
        if os.path.isdir(self.masks_dir):
            mask_tensors = []
            for vf in self.volume_files:
                fname = os.path.basename(vf)
                mask_path = os.path.join(self.masks_dir, fname)
                if os.path.exists(mask_path):
                    m, _ = self.loader.load_nifti(mask_path)
                    mask_tensors.append(m)

            if mask_tensors:
                masks = torch.stack(mask_tensors, dim=0)  # [V, 1, D, H, W]

        final_affine = affines[0]

        return {
            "views": views,
            "masks": masks,
            "filename": self.sample_name,
            "affine": final_affine
        }


def get_test_dataloader(data_config, batch_size=1):
    """Creates and returns inference DataLoader."""
    if "data_dir" not in data_config:
        raise KeyError("[ERROR] Missing 'data_dir' key inside 'data' config!")
    loader = InferenceDataLoader(data_config)
    dataset = InferenceDataset(loader)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=data_config.get("num_workers", 1))

