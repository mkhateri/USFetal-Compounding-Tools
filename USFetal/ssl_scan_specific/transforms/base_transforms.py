import torch
import monai.transforms as mt
import numpy as np
import random
import logging
import nibabel as nib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BaseTransforms:
    def __init__(self, mode="train", crop_size=(128, 128, 128), min_nonzero_ratio=0.8, 
                 num_augmentations=3, random_crop=True, max_retries=200):  
        self.mode = mode
        self.crop_size = crop_size
        self.min_nonzero_ratio = min_nonzero_ratio
        self.num_augmentations = num_augmentations
        self.random_crop = random_crop
        self.max_retries = max_retries

        # Define spatial augmentations
        self.spatial_augmentations = [
            mt.RandFlipd(keys=["views", "masks"], prob=0.2, spatial_axis=0),
            mt.RandFlipd(keys=["views", "masks"], prob=0.2, spatial_axis=1),
            mt.RandFlipd(keys=["views", "masks"], prob=0.2, spatial_axis=2),
        ]

    def ensure_tensor(self, data):
        """Ensure the input data is a PyTorch Tensor."""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            return data.float()
        return data



    def _normalize_image(self, image, mode="clip01"):
        """
        Normalizes each view in the image tensor independently using one of two modes:
        
        - "clip01": Clipping-based normalization (divides by 255 and clips to [0,1])
        - "minmax": Min-Max normalization (scales each view independently between 0 and 1)

        Args:
            image (torch.Tensor or np.ndarray): Input image of shape [V, C, D, H, W] or [B, V, C, D, H, W].
            mode (str): Normalization mode, either "clip01" (default) or "minmax".
        
        Returns:
            torch.Tensor: Normalized image with same shape as input.
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        if image.ndim not in [5, 6]:
            raise ValueError(f"Unexpected shape {image.shape}. Expected 5D (V, C, D, H, W) or 6D (B, V, C, D, H, W).")

        if mode == "clip01":
            normalized_image = image / 255.0
            return torch.clamp(normalized_image, 0, 1)

        elif mode == "minmax":
            # Compute min-max for **each view independently**
            min_val = image.amin(dim=(-3, -2, -1), keepdim=True)  # Min over spatial dimensions (D, H, W)
            max_val = image.amax(dim=(-3, -2, -1), keepdim=True)  # Max over spatial dimensions (D, H, W)

            # Avoid division by zero
            range_val = max_val - min_val
            range_val[range_val == 0] = 1e-6  # Prevent zero division

            normalized_image = (image - min_val) / range_val
            return torch.clamp(normalized_image, 0, 1)

        else:
            raise ValueError(f"Unsupported normalization mode: {mode}. Choose 'clip01' or 'minmax'.")


    def _random_crop_nonzero(self, views, masks):
        if not self.random_crop:
            return views, masks

        # Ensure batch dimension exists
        if views.ndim == 5:
            views = views.unsqueeze(0)
            masks = masks.unsqueeze(0)

        B, V, C, D, H, W = views.shape
        crop_d, crop_h, crop_w = self.crop_size

        # Max values for cropping
        d_max = D - crop_d
        h_max = H - crop_h
        w_max = W - crop_w

        retries = 0
        non_zero_ratio = 0
        while retries < self.max_retries:
            d = random.randint(0, d_max)
            h = random.randint(0, h_max)
            w = random.randint(0, w_max)

            cropped_views = views[:, :, :, d:d+crop_d, h:h+crop_h, w:w+crop_w]
            cropped_masks = masks[:, :, :, d:d+crop_d, h:h+crop_h, w:w+crop_w]

            non_zero_voxels = (cropped_masks > 0).float().sum()
            total_voxels = cropped_masks.numel()
            non_zero_ratio = non_zero_voxels / total_voxels

            if non_zero_ratio >= self.min_nonzero_ratio:
                break

            retries += 1

        if retries == self.max_retries:
            logger.warning(f"Max retries reached for cropping. Returning crop with non-zero ratio of {non_zero_ratio:.2f}.")
            
            # Fallback: Apply center crop
            center_d = (D - crop_d) // 2
            center_h = (H - crop_h) // 2
            center_w = (W - crop_w) // 2

            cropped_views = views[:, :, :, center_d:center_d+crop_d, center_h:center_h+crop_h, center_w:center_w+crop_w]
            cropped_masks = masks[:, :, :, center_d:center_d+crop_d, center_h:center_h+crop_h, center_w:center_w+crop_w]

            non_zero_voxels = (cropped_masks > 0).float().sum()
            total_voxels = cropped_masks.numel()
            non_zero_ratio = non_zero_voxels / total_voxels

        return cropped_views, cropped_masks

    def _build_transform_pipeline(self):
        if self.mode == "train":
            selected_spatial = random.sample(
                self.spatial_augmentations, min(self.num_augmentations, len(self.spatial_augmentations))
            )
            return mt.Compose(selected_spatial + [
                mt.EnsureTyped(keys=["views", "masks"], allow_missing_keys=True)
            ])
        else:
            return mt.Compose([
                mt.EnsureTyped(keys=["views", "masks"], allow_missing_keys=True)
            ])

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dictionary, got {type(data)}")

        data["views"] = self._normalize_image(data["views"])
        data["masks"] = self.ensure_tensor(data["masks"])

        data["views"], data["masks"] = self._random_crop_nonzero(data["views"], data["masks"])

        # Ensure correct shape for MONAI transforms
        if data["views"].ndim == 5:
            data["views"] = data["views"].unsqueeze(0)
        if data["masks"].ndim == 5:
            data["masks"] = data["masks"].unsqueeze(0)

        if data["views"].shape[2] != 1:
            data["views"] = data["views"].unsqueeze(2)
        if data["masks"].shape[2] != 1:
            data["masks"] = data["masks"].unsqueeze(2)

        B, V, C, D, H, W = data["views"].shape
        views_reshaped = data["views"].view(B * V, C, D, H, W)
        masks_reshaped = data["masks"].view(B * V, C, D, H, W)

        transform_pipeline = self._build_transform_pipeline()

        try:
            transformed_data = transform_pipeline({"views": views_reshaped, "masks": masks_reshaped})

            data["views"] = transformed_data["views"].view(B, V, C, D, H, W)
            data["masks"] = transformed_data["masks"].view(B, V, C, D, H, W)

            return data

        except Exception as e:
            logger.error(f"⚠️ Error during transformation: {e}")
            return data


# ========================== EXAMPLE USAGE ==========================
if __name__ == "__main__":
    transform = BaseTransforms(mode="train", crop_size=(128, 128, 128), min_nonzero_ratio=0.8, num_augmentations=3)
    
    sample_data = {
        "views": np.random.rand(8, 1, 256, 320, 320).astype(np.float32),  # Simulated views
        "masks": np.random.randint(0, 2, (8, 1, 256, 320, 320)).astype(np.float32)  # Simulated masks
    }

    transformed_data = transform(sample_data)
    
    print(f"Transformed Views Shape: {transformed_data['views'].shape}")
    print(f"Transformed Masks Shape: {transformed_data['masks'].shape}")
