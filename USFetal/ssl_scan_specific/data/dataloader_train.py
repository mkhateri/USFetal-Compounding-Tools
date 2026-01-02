import torch
import numpy as np
import random
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import glob
import sys
import os

# Add the project root directory to sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_DIR)

# Import the custom transformation pipeline
from transforms.ultrasound_transforms import UltrasoundTransforms


class DataLoaderInitializer:
    """
    Initializes dataset components for multi-view ultrasound images in `.nii.gz` format.
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config["data"]["data_dir"]).expanduser()

        self.data_dir = (
            Path(config["data"]["data_dir"])
            .expanduser()
            .resolve()
        )

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}"
            )

        self.dtype = getattr(torch, config["data"]["dtype"])

    def get_sample_list(self):
        """Retrieve list of all sample folders."""
        return sorted([f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))])

    def load_nifti(self, file_path):
        """Load a `.nii.gz` file as a tensor with correct shape [C, D, H, W]."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")


        nii_img = nib.load(file_path)
        data = nii_img.get_fdata(dtype=np.float32)
        data = torch.tensor(data, dtype=self.dtype)

        # Ensure that data is [1, D, H, W] for single-channel images
        if data.ndimension() == 3:  # [D, H, W]
            data = data.unsqueeze(0)  # [1, D, H, W]
        return data

class UltrasoundDataset(Dataset):
    """Dataset for multi-view ultrasound image processing with augmentations."""
    def __init__(self, initializer: DataLoaderInitializer, transform=None, repeat_factor=10):

        self.init = initializer
        self.transform = transform
        self.repeat_factor = repeat_factor  # SSL virtual repetition

        # data_dir itself is the sample (single ID)
        self.sample_name = self.init.data_dir.name
        self.sample_path = str(self.init.data_dir)

        self.volumes_dir = os.path.join(self.sample_path, "volumes")
        self.masks_dir = os.path.join(self.sample_path, "masks")

        if not os.path.isdir(self.volumes_dir):
            raise ValueError(f"Missing 'volumes/' directory in {self.sample_path}")
        if not os.path.isdir(self.masks_dir):
            raise ValueError(f"Missing 'masks/' directory in {self.sample_path}")

        self.volume_files = sorted(glob.glob(os.path.join(self.volumes_dir, "*.nii.gz")))
        self.mask_files = sorted(glob.glob(os.path.join(self.masks_dir, "*.nii.gz")))

        if len(self.volume_files) == 0 or len(self.mask_files) == 0:
            raise ValueError(f"No NIfTI files found in {self.sample_path}")

        if len(self.volume_files) != len(self.mask_files):
            raise ValueError(
                f"Mismatched volumes ({len(self.volume_files)}) and masks ({len(self.mask_files)})"
            )

        # Ensure filenames match (strong safety check)
        vol_names = [os.path.basename(f) for f in self.volume_files]
        mask_names = [os.path.basename(f) for f in self.mask_files]
        if vol_names != mask_names:
            raise ValueError("Volume and mask filenames do not match exactly.")

    def __len__(self):
        # virtual dataset size for SSL
        return self.repeat_factor

    def __getitem__(self, idx):
        # deterministic randomness per virtual index
        seed = torch.initial_seed() + idx
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

        # Load all views
        views = torch.stack(
            [self.init.load_nifti(v) for v in self.volume_files],
            dim=0
        )  # [V, 1, D, H, W]

        masks = torch.stack(
            [self.init.load_nifti(m) for m in self.mask_files],
            dim=0
        )  # [V, 1, D, H, W]

        # Preserve original behavior
        max_views = views.shape[0]
        V = views.shape[0]

        if V < max_views:
            repeat_indices = torch.randint(0, V, (max_views,))
            views = views[repeat_indices]
            masks = masks[repeat_indices]
        elif V > max_views:
            selected_indices = torch.randperm(V)[:max_views]
            views = views[selected_indices]
            masks = masks[selected_indices]

        # Apply augmentations
        if self.transform:
            transformed = self.transform(
                {"views": views, "masks": masks, "sample_name": self.sample_name}
            )
            views, masks = transformed["views"], transformed["masks"]

        # Match previous tensor format
        views = views.squeeze(0)   # [V, D, H, W]
        masks = masks.squeeze(0)

        mean_views = views.mean(dim=0, keepdim=True)  # [1, D, H, W]

        return {
            "views": views,
            "masks": masks,
            "mean_views": mean_views,
            "sample_name": self.sample_name,
        }

def get_dataloaders(config):
    """Creates and returns train & validation DataLoaders using the SAME full dataset."""
    
    # Initialize DataLoaderInitializer with config
    initializer = DataLoaderInitializer(config)

    # Fetch transformation settings
    transform = UltrasoundTransforms(
        mode="train",
        crop_size=tuple(config["transforms"]["crop_size"]),  
        min_nonzero_ratio=config["transforms"]["min_nonzero_ratio"],  
        num_augmentations=config["transforms"]["num_augmentations"],  
        augment_list=config["transforms"]["augment_list"],
        random_crop=config["transforms"].get("random_crop", True)  
    )

    # Initialize the dataset with transformations
    dataset = UltrasoundDataset(initializer, transform=transform)

    total_samples = len(dataset)

    if total_samples == 0:
        raise ValueError("No samples found in the dataset directory! Check your dataset path.")

    train_dataset = dataset
    val_dataset = dataset

    batch_size = min(config["data"]["batch_size"], total_samples)

    dataloader_params = {
        "batch_size": batch_size,
        "num_workers": config["data"]["num_workers"],
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }

    # Return DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    valid_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

    return train_loader, valid_loader


