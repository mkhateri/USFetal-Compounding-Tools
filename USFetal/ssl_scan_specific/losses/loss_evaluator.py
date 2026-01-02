import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from .losses_registry import get_loss_function, REGISTERED_LOSSES
import numpy as np
class LossEvaluator:
    def __init__(self, config, device, amp_dtype):
        self.device = device
        self.amp_dtype = amp_dtype

        requested_losses = config["training"].get("loss", [])
        invalid_losses = [ln for ln in requested_losses if ln not in REGISTERED_LOSSES]
        if invalid_losses:
            raise ValueError(
                f"Unregistered losses: {invalid_losses}. "
                f"Available losses: {list(REGISTERED_LOSSES.keys())}"
            )

        self.loss_functions = {name: get_loss_function(name) for name in requested_losses}
        loss_weight_cfg = config["training"].get("loss_weights", {})
        self.loss_weight = {name: loss_weight_cfg.get(name, 1.0) for name in self.loss_functions}

        self.mean_consistency_weight = config["training"].get("mean_consistency_weight", 1.0)
        self.dog_consistency_weight = config["training"].get("dog_consistency_weight", 1.0)

        # DoG parameters
        self.sigma_list = config["training"].get("dog_sigma_list", [0.5, 1.0, 2.0])
        self.boost_base = config["training"].get("dog_boost_base", 1.0)
        self.boost_step = config["training"].get("dog_boost_step", 1.5)

    def compute_multiscale_3d_2d_dog(self, volume):
        """Compute Multi-Scale 3D + slice-wise 2D DoG."""
        volume_np = volume.to(dtype=torch.float32).detach().cpu().numpy()

        # 3D volumetric DoG
        dog_3d = np.zeros_like(volume_np)
        for i, (s1, s2) in enumerate(zip(self.sigma_list[:-1], self.sigma_list[1:])):
            g1 = gaussian_filter(volume_np, sigma=s1)
            g2 = gaussian_filter(volume_np, sigma=s2)
            factor = self.boost_base * (self.boost_step ** i)
            dog_3d += factor * (g1 - g2)


        dog_final = dog_3d 
        return torch.tensor(dog_final, dtype=torch.float32, device=volume.device)

    def compute_loss(self, outputs, inputs):
        loss_dict = {}

        fused_out = outputs["fused_out"].to(self.device, dtype=self.amp_dtype)  # [B,1,D,H,W]
        multi_views = inputs["views"].to(self.device, dtype=self.amp_dtype)      # [B,V,1,D,H,W]
        multi_views_masks = inputs["masks"].to(self.device, dtype=self.amp_dtype)

        B, V, C, D, H, W = multi_views.shape

        fused_out_repeated = fused_out.unsqueeze(1).expand(-1, V, -1, -1, -1, -1)

        masked_fused_out = fused_out_repeated * multi_views_masks
        masked_multi_views = multi_views * multi_views_masks

        valid_counts_per_view = multi_views_masks.sum(dim=[2, 3, 4, 5], keepdim=True).clamp(min=1e-6)

        mean_views = (masked_multi_views / valid_counts_per_view).sum(dim=1, keepdim=True)
        mean_fused = (masked_fused_out / valid_counts_per_view).sum(dim=1, keepdim=True)

        mean_consistency_loss = self.mean_consistency_weight * torch.mean((mean_fused - mean_views) ** 2)

        ### Multi-Scale DoG Consistency Loss
        dog_views = []
        for v in range(V):
            dog_view = self.compute_multiscale_3d_2d_dog(multi_views[:, v, 0])  # [B,D,H,W]
            dog_views.append(dog_view.unsqueeze(1))
        dog_views = torch.cat(dog_views, dim=1)
        dog_mean_views = dog_views.mean(dim=1)

        dog_fused = self.compute_multiscale_3d_2d_dog(fused_out.squeeze(1)).unsqueeze(1)

        #dog_consistency_loss = self.dog_consistency_weight * torch.mean((dog_fused - dog_mean_views) ** 2)
        dog_consistency_loss = self.dog_consistency_weight * torch.mean(torch.abs(dog_fused - dog_mean_views))

        ### --- Compute standard supervised losses ---
        total_slicewise_loss = torch.zeros([], device=self.device, dtype=self.amp_dtype)
        total_volume_loss = torch.zeros([], device=self.device, dtype=self.amp_dtype)

        for loss_name, loss_fn in self.loss_functions.items():
            weight = self.loss_weight[loss_name]

            if "SLICEWISE" in loss_name.upper():
                slice_loss_each = loss_fn(masked_fused_out, masked_multi_views, mode="slicewise", axis=None)
                slice_loss_each = (slice_loss_each * multi_views_masks).sum() / valid_counts_per_view.sum()
                combined_slice = weight * slice_loss_each
                loss_dict[loss_name] = combined_slice
                total_slicewise_loss += combined_slice

            elif "VOLUME" in loss_name.upper():
                volume_loss_sum = torch.zeros([], device=self.device, dtype=self.amp_dtype)
                for i in range(V):
                    single_view = masked_multi_views[:, i, :, :, :, :]
                    fused_volume = masked_fused_out[:, i, :, :, :, :]
                    vol_loss_i = loss_fn(fused_volume, single_view, mode="volume")
                    valid_counts_single = multi_views_masks[:, i, :, :, :, :].sum(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-6)
                    vol_loss_i = (vol_loss_i * multi_views_masks[:, i, :, :, :, :]).sum() / valid_counts_single.sum()
                    volume_loss_sum += vol_loss_i

                combined_vol = weight * volume_loss_sum
                loss_dict[loss_name] = combined_vol
                total_volume_loss += combined_vol

        # Add consistency terms
        loss_dict["MEAN_CONSISTENCY_LOSS"] = mean_consistency_loss
        loss_dict["DOG_CONSISTENCY_LOSS"] = dog_consistency_loss

        total_loss = total_slicewise_loss + total_volume_loss + mean_consistency_loss + dog_consistency_loss
        loss_dict["TOTAL_SLICEWISE_LOSS"] = total_slicewise_loss
        loss_dict["TOTAL_VOLUME_LOSS"] = total_volume_loss
        loss_dict["TOTAL_LOSS"] = total_loss

        return loss_dict, total_slicewise_loss, total_volume_loss

