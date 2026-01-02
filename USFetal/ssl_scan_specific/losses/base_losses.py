import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
import monai  # Import MONAI for SSIM version check
from monai.losses.ssim_loss import SSIMLoss as MonaiSSIMLoss  # Import MONAI’s SSIMLoss

# ----------------------------------------------------
# Base Loss Class (Parent for All Losses)
# ----------------------------------------------------
class BaseLoss(nn.Module):
    def forward(self, pred, target, mode="slicewise", axis=None):
        raise NotImplementedError("Each loss function must implement the forward method.")

# ----------------------------------------------------
# Unified Loss Computation
# ----------------------------------------------------
class UnifiedLossMixin:
    def _compute_loss(self, loss_fn, pred, target, mode, axis):
        """
        If 'volume', call loss_fn directly.
        If 'slicewise', process each slice along the selected axis (0,1,2) without merging views.
        """
        if mode == "volume":
            return loss_fn(pred, target)
        else:
            return self._compute_slicewise_loss(loss_fn, pred, target, axis)

    def _compute_slicewise_loss(self, loss_fn, pred, target, axis):
        """
        Computes slicewise loss:
        1) If input is [B,V,C,D,H,W], flatten it to [B*V,C,D,H,W].
        2) Slice along the requested axes (0=Depth, 1=Height, 2=Width).
        3) Compute loss for each slice and return mean loss.
        """
        # Flatten [B,V,C,D,H,W] → [B*V,C,D,H,W]
        if pred.dim() == 6:
            b, v, c, d, h, w = pred.shape
            pred = rearrange(pred, "b v c d h w -> (b v) c d h w")
            target = rearrange(target, "b v c d h w -> (b v) c d h w")

        # Select slicing axes (default: all three)
        axes = [0, 1, 2] if axis is None else [axis]

        # Compute loss per slice
        total_loss = sum(self._slice_loss(loss_fn, pred, target, ax) for ax in axes)
        return total_loss  # Average over all slicing axes


    def _slice_loss(self, loss_fn, pred, target, axis):
        """
        Handles slicing and ensures correct SSIM input shape.
        """
        #print(f"🔍 DEBUG: Input Shape - Pred: {pred.shape}, Target: {target.shape}, Axis: {axis}")

        if axis == 0:
            pred_slices   = rearrange(pred,   "b c d h w -> (b d) c h w")
            target_slices = rearrange(target, "b c d h w -> (b d) c h w")
        elif axis == 1:
            pred_slices   = rearrange(pred,   "b c d h w -> (b h) c d w")
            target_slices = rearrange(target, "b c d h w -> (b h) c d w")
        elif axis == 2:
            pred_slices   = rearrange(pred,   "b c d h w -> (b w) c d h")
            target_slices = rearrange(target, "b c d h w -> (b w) c d h")
        else:
            raise ValueError(f"Invalid axis: {axis}")

        #print(f"🔍 DEBUG: After Slicing - Pred: {pred_slices.shape}, Target: {target_slices.shape}")

        # If using SSIM, ensure it's in the expected 5D format
        if isinstance(loss_fn, SSIMLoss):
            # print("⚠ Converting to 5D for SSIM...")
            pred_slices = pred_slices.unsqueeze(2)  # [B', C, 1, H, W]
            target_slices = target_slices.unsqueeze(2)  # [B', C, 1, H, W]

        return loss_fn(pred_slices, target_slices)



# ----------------------------------------------------
# Mean Squared Error (MSE) Loss
# ----------------------------------------------------
class MSELoss(UnifiedLossMixin, BaseLoss):
    def forward(self, pred, target, mode="slicewise", axis=None):
        return self._compute_loss(F.mse_loss, pred, target, mode, axis)

# ----------------------------------------------------
# Mean Absolute Error (MAE) Loss
# ----------------------------------------------------
class MAELoss(UnifiedLossMixin, BaseLoss):
    def forward(self, pred, target, mode="slicewise", axis=None):
        return self._compute_loss(F.l1_loss, pred, target, mode, axis)

# ----------------------------------------------------
# Gradient-Based Loss
# ----------------------------------------------------
class GradientLoss(UnifiedLossMixin, BaseLoss):
    def forward(self, pred, target, mode="slicewise", axis=None):
        return self._compute_loss(self._gradient_loss, pred, target, mode, axis)

    def _gradient_loss(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)) / 2.0

# ----------------------------------------------------
# SSIM Loss using MONAI's SSIMMetric
# ----------------------------------------------------

# class SSIMLossMONAI(UnifiedLossMixin, BaseLoss):  # Define this first
#     def __init__(self, spatial_dims=2, win_size=11, data_range=1.0):
#         super().__init__()
#         self.loss_fn = MonaiSSIMLoss(spatial_dims=spatial_dims, win_size=win_size, data_range=data_range)

#     def forward(self, pred, target, mode="slicewise", axis=None):
#         return self._compute_loss(self.loss_fn, pred, target, mode, axis)

class SSIMLossMONAI(UnifiedLossMixin, BaseLoss):
    def __init__(self, win_size=11, data_range=1.0):
        """
        MONAI's SSIM loss adapted for both 2D (slicewise) and 3D (volume) computations.
        """
        super().__init__()
        # Create separate SSIM loss functions for slice-wise (2D) and volume-wise (3D)
        self.loss_fn_2d = MonaiSSIMLoss(spatial_dims=2, win_size=win_size, data_range=data_range)  # For slicewise
        self.loss_fn_3d = MonaiSSIMLoss(spatial_dims=3, win_size=win_size, data_range=data_range)  # For volumewise

    def forward(self, pred, target, mode="slicewise", axis=None):
        """
        Forward method dynamically selects the correct SSIM function:
        - Uses `self.loss_fn_2d` for "slicewise" mode.
        - Uses `self.loss_fn_3d` for "volume" mode.
        """
        if mode == "volume":
            return self._compute_volume_loss(pred, target)  # Call volume loss function
        return self._compute_loss(self.loss_fn_2d, pred, target, mode, axis)  # Use 2D SSIM for slices

    def _compute_volume_loss(self, pred, target):
        """
        Compute SSIM loss for 3D volumes.
        MONAI expects input shapes as **[B, C, D, H, W]**.
        """
        if pred.dim() == 5 and target.dim() == 5:  # Expected shape [B, C, D, H, W]
            return self.loss_fn_3d(pred, target)
        else:
            raise ValueError(f"Expected [B, C, D, H, W] but got {pred.shape} and {target.shape}")


# ----------------------------------------------------
# Final Loss Classes
# ----------------------------------------------------
class PerceptualLoss(UnifiedLossMixin, BaseLoss):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:8]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target, mode="slicewise", axis=None):
        if mode == "volume":
            raise NotImplementedError("Perceptual Loss is only supported in slicewise mode.")
        return self._compute_loss(self._perceptual_loss, pred, target, mode, axis)

    def _perceptual_loss(self, pred, target):
        device = pred.device
        self.feature_extractor = self.feature_extractor.to(device)
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.mse_loss(pred_features, target_features, reduction="mean")

# ----------------------------------------------------
# Intensity-Weighted Loss
# ----------------------------------------------------
class IntensityWeightedLoss(UnifiedLossMixin, BaseLoss):
    def __init__(self, base_loss="mse", weight_mode="log", alpha=1.0):
        """
        A loss function that gives more weight to high-intensity regions.
        
        Args:
            base_loss (str): The base loss function to use. Options: ["mse", "mae"].
            weight_mode (str): Weighting mode for intensities. Options: ["linear", "exp", "sqrt", "log"].
            alpha (float): Scaling factor for exponential weighting (only used if weight_mode="exp").
        """
        super().__init__()
        self.base_loss = base_loss
        self.weight_mode = weight_mode
        self.alpha = alpha

        # Define the base loss function dynamically
        if self.base_loss == "mse":
            self.loss_fn = F.mse_loss
        elif self.base_loss == "mae":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unsupported base loss: {self.base_loss}")

    def _compute_weight_map(self, target):
        """Compute weight map based on intensity values."""
        if self.weight_mode == "linear":
            weight_map = target  # Directly use intensity as weight
        elif self.weight_mode == "exp":
            weight_map = torch.exp(self.alpha * target)  # Exponential scaling
        elif self.weight_mode == "sqrt":
            weight_map = torch.sqrt(target + 1e-6)  # Square root scaling
        elif self.weight_mode == "log":
            weight_map = torch.log1p(target)  # Logarithmic weighting
        else:
            raise ValueError(f"Invalid weight mode: {self.weight_mode}")

        return weight_map / (weight_map.max() + 1e-6)  # Normalize between 0 and 1

    def forward(self, pred, target, mode="slicewise", axis=None):
        """Compute the intensity-weighted loss."""
        weight_map = self._compute_weight_map(target)

        # Compute base loss in the selected mode
        if mode == "volume":
            loss = self.loss_fn(pred, target, reduction="none")  # Per-pixel loss
        else:
            loss = self._compute_loss(self.loss_fn, pred, target, mode, axis)

        # Apply the intensity-based weight map
        weighted_loss = loss * weight_map
        return weighted_loss.mean()  # Reduce to scalar loss
# ----------------------------------------------------
# Apply mixin to losses
# ----------------------------------------------------
class MSELoss(MSELoss, UnifiedLossMixin): pass
class MAELoss(MAELoss, UnifiedLossMixin): pass
class SSIMLoss(SSIMLossMONAI, UnifiedLossMixin): pass
class GradientLoss(GradientLoss, UnifiedLossMixin): pass
class PerceptualLoss(PerceptualLoss, UnifiedLossMixin): pass
class IntensityWeightedLoss(IntensityWeightedLoss, UnifiedLossMixin): pass
