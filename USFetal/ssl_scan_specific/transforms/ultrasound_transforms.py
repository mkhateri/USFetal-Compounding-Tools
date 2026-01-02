import torch
import sys
import os
import logging
import monai.transforms as mt
import random
import numpy as np
from transforms.augment_registry import AUGMENTATION_REGISTRY  
from transforms.base_transforms import BaseTransforms  # Import Base Transforms
from utils.logger import setup_logger  

# Fix relative imports if running as a standalone script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logger = setup_logger("logs/ultrasound_transforms.log", "UltrasoundTransformsLogger")

def ensure_tensor(data):
    """Ensure the input data is a PyTorch Tensor, but ignore strings."""
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32)
    elif isinstance(data, torch.Tensor):
        return data.float()
    elif isinstance(data, str):  
        return data
    return data  

class UltrasoundTransforms:
    """
    Ultrasound Augmentation Pipeline.
    - **Delegates normalization & cropping** to `BaseTransforms`
    - **Registry-based augmentations** (from MONAI and Custom Registry) are applied afterward.
    - Spatial transformations (rotation, flip, affine) are **applied to both views and masks**.
    - Intensity transformations (noise, contrast, brightness) are **only applied to views**.
    """

    def __init__(self, mode="train", crop_size=(128, 128, 128), min_nonzero_ratio=0.9, 
                 num_augmentations=3, augment_list=None, augment_keys=("views", "masks"), 
                 exclude_key="mean_image", random_crop=True):
        self.mode = mode
        self.num_augmentations = num_augmentations  # Ensure `num_augmentations` is set
        self.augment_keys = augment_keys
        self.exclude_key = exclude_key

        # Delegate normalization & cropping to BaseTransforms
        self.base_transform = BaseTransforms(
            mode=self.mode, 
            crop_size=crop_size, 
            min_nonzero_ratio=min_nonzero_ratio,
            random_crop=random_crop
        )  

        # Use augmentations from config if provided
        available_augmentations = list(AUGMENTATION_REGISTRY.keys())
        self.augment_list = [aug for aug in augment_list if aug in available_augmentations] if augment_list else available_augmentations

        if len(self.augment_list) == 0:
            logger.warning(" No valid augmentations found. Using default MONAI transformations.")
            self.augment_list = ["RandGaussianNoise"]  # Only apply RandGaussianNoise

    def _build_transform_pipeline(self):
        """Retrieve and initialize augmentations from the registry."""
        monai_transforms = []

        selected_augmentations = (
            random.sample(self.augment_list, min(self.num_augmentations, len(self.augment_list)))  #  `self.num_augmentations` now exists
            if self.num_augmentations
            else self.augment_list
        )

        #logger.info(f"Applying Augmentations: {selected_augmentations}")  

        for name in selected_augmentations:
            if name in AUGMENTATION_REGISTRY:
                transform_func = AUGMENTATION_REGISTRY[name]
                try:
                    monai_transforms.append(transform_func(keys=["views", "masks"]))  #  Apply to both views and masks
                except Exception as e:
                    logger.error(f" Error initializing augmentation '{name}': {e}")

        return mt.Compose(monai_transforms)

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dictionary, got {type(data)}")

        data = {k: ensure_tensor(v) for k, v in data.items()}  

        # Normalize & Crop using BaseTransforms
        data = self.base_transform(data)  

        transform_pipeline = self._build_transform_pipeline()

        try:
            transformed_data = transform_pipeline({"views": data["views"], "masks": data["masks"]})

            data["views"] = transformed_data["views"]
            data["masks"] = transformed_data["masks"]
            return data

        except Exception as e:
            logger.error(f" Error during transformation: {e}")
            return data


# ========================== **TESTING CODE** ==========================
if __name__ == "__main__":
    CONFIG = {
        "data": {
            "data_dir": "/research/users/mkhateri/hms_sr/UltraSoundSR/ultrasound_data/",
            "batch_size": 4,
            "num_workers": 2,
            "train_split": 0.8,
            "dtype": "float32"
        },
        "training": {
            "seed": 42
        },
        "transforms": {
            "crop_size": [128, 128, 128],  # Pass crop_size
            "min_nonzero_ratio": 0.5,  # Pass min_nonzero_ratio
            "num_augmentations": 3,  # Randomly select num_augmentations from the augment_list
            "augment_list": ["RandGaussianNoise", "RandAnisotropicBlur", "RandBiasField", "RandCoarseDropout", "RandCoarseShuffle"],
            "random_crop": False  # Disable cropping if needed
        }
    }

    # Pass crop_size and min_nonzero_ratio
    transform = UltrasoundTransforms(
        mode="train",
        crop_size=CONFIG["transforms"]["crop_size"],
        min_nonzero_ratio=CONFIG["transforms"]["min_nonzero_ratio"],
        num_augmentations=CONFIG["transforms"]["num_augmentations"],
        augment_list=CONFIG["transforms"]["augment_list"],
        random_crop=CONFIG["transforms"]["random_crop"]  # Dynamic toggle
    )

    #  Correcting the shape of sample_data
    sample_data = {
        "views": torch.randn(1, 8, 1, 128, 128, 128),  # Added batch dim
        "masks": torch.randint(0, 2, (1, 8, 1, 128, 128, 128)),  # Added batch dim
        "mean_image": torch.randn(1, 1, 128, 128, 128),  # Ensuring correct dimensions
        "sample_name": "sample_001"
    }

    transformed_sample = transform(sample_data)

    logger.info("\nTransformed Data Summary:")
    for key, value in transformed_sample.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"{key}: {value.shape}, dtype: {value.dtype}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("Transformations applied successfully!")
