import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

def visualize_mri_slices(inputs, outputs, targets, save_dir, epoch, view_name):
    """Save middle MRI slices for all three views (axial, coronal, sagittal)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure inputs are tensors and move to CPU
    inputs, outputs, targets = map(
        lambda x: torch.tensor(x).cpu() if not isinstance(x, torch.Tensor) else x.cpu(),
        [inputs, outputs, targets]
    )

    # Ensure tensors have at least 5 dimensions (B, C, W, H, D)
    def ensure_5d(tensor):
        if tensor.dim() == 3:  # Shape (W, H, D) → Add batch & channel dim
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 4:  # Shape (B, W, H, D) → Add channel dim
            return tensor.unsqueeze(1)
        return tensor

    inputs, outputs, targets = map(ensure_5d, [inputs, outputs, targets])

    num_samples = min(3, inputs.shape[0])  # Limit to 3 samples for visualization

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure consistency for single sample

    for i in range(num_samples):
        mid_slice = inputs.shape[-1] // 2  # Middle slice along depth (D)

        axes[i, 0].imshow(inputs[i, 0, :, :, mid_slice].float().numpy(), cmap="gray")
        axes[i, 0].set_title(f"LR Input ({view_name})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(outputs[i, 0, :, :, mid_slice].float().numpy(), cmap="gray")
        axes[i, 1].set_title(f"Super-Resolved Output ({view_name})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(targets[i, 0, :, :, mid_slice].float().numpy(), cmap="gray")
        axes[i, 2].set_title(f"Ground Truth ({view_name})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"epoch_{epoch}_{view_name}.png")
    plt.close()




def visualize_output_3d(
    lr_axial, lr_coronal, lr_sagittal, output_3d, targets, epoch, batch_idx, 
    save_dir="./outputs/visualizations/output_3d", slice_offsets=[-2, -1, 0, 1, 2]
):
    """Visualizes 3D LR inputs (Axial, Coronal, Sagittal), Super-Resolved (SR) output, and Ground Truth (GT) for multiple slices.
       Saves figures in a structured folder per epoch.
    """
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)  # Ensure directory exists

    # Get image dimensions
    depth, height, width = targets.shape[2], targets.shape[3], targets.shape[4]

    # Get middle indices
    slice_d = depth // 2  # Middle along Depth (Z)
    slice_h = height // 2  # Middle along Height (Y)
    slice_w = width // 2  # Middle along Width (X)

    # Iterate over slice offsets
    for offset in slice_offsets:
        slice_d_offset = max(0, min(depth - 1, slice_d + offset))
        slice_h_offset = max(0, min(height - 1, slice_h + offset))
        slice_w_offset = max(0, min(width - 1, slice_w + offset))

        # Extract slices for Depth View (Middle along Depth)
        lr_axial_d = lr_axial[0, 0, slice_d_offset, :, :].float().cpu().numpy()
        lr_coronal_d = lr_coronal[0, 0, slice_d_offset, :, :].float().cpu().numpy()
        lr_sagittal_d = lr_sagittal[0, 0, slice_d_offset, :, :].float().cpu().numpy()
        output_d = output_3d[0, 0, slice_d_offset, :, :].float().cpu().numpy()
        target_d = targets[0, 0, slice_d_offset, :, :].float().cpu().numpy()

        # Extract slices for Height View (Middle along Height)
        lr_axial_h = lr_axial[0, 0, :, slice_h_offset, :].float().cpu().numpy()
        lr_coronal_h = lr_coronal[0, 0, :, slice_h_offset, :].float().cpu().numpy()
        lr_sagittal_h = lr_sagittal[0, 0, :, slice_h_offset, :].float().cpu().numpy()
        output_h = output_3d[0, 0, :, slice_h_offset, :].float().cpu().numpy()
        target_h = targets[0, 0, :, slice_h_offset, :].float().cpu().numpy()

        # Extract slices for Width View (Middle along Width)
        lr_axial_w = lr_axial[0, 0, :, :, slice_w_offset].float().cpu().numpy()
        lr_coronal_w = lr_coronal[0, 0, :, :, slice_w_offset].float().cpu().numpy()
        lr_sagittal_w = lr_sagittal[0, 0, :, :, slice_w_offset].float().cpu().numpy()
        output_w = output_3d[0, 0, :, :, slice_w_offset].float().cpu().numpy()
        target_w = targets[0, 0, :, :, slice_w_offset].float().cpu().numpy()

        # Create figure with 3 rows (Depth, Height, Width)
        fig, axes = plt.subplots(3, 5, figsize=(20, 15))

        # Row 1: Slices along Depth
        axes[0, 0].imshow(lr_axial_d, cmap="gray")
        axes[0, 1].imshow(lr_coronal_d, cmap="gray")
        axes[0, 2].imshow(lr_sagittal_d, cmap="gray")
        axes[0, 3].imshow(output_d, cmap="gray")
        axes[0, 4].imshow(target_d, cmap="gray")

        # Row 2: Slices along Height
        axes[1, 0].imshow(lr_axial_h, cmap="gray")
        axes[1, 1].imshow(lr_coronal_h, cmap="gray")
        axes[1, 2].imshow(lr_sagittal_h, cmap="gray")
        axes[1, 3].imshow(output_h, cmap="gray")
        axes[1, 4].imshow(target_h, cmap="gray")

        # Row 3: Slices along Width
        axes[2, 0].imshow(lr_axial_w, cmap="gray")
        axes[2, 1].imshow(lr_coronal_w, cmap="gray")
        axes[2, 2].imshow(lr_sagittal_w, cmap="gray")
        axes[2, 3].imshow(output_w, cmap="gray")
        axes[2, 4].imshow(target_w, cmap="gray")

        # Titles for columns
        col_titles = ["LR Axial", "LR Coronal", "LR Sagittal", "Predicted SR", "Ground Truth"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title)

        # Row labels
        row_titles = [
            f"Depth Slice {offset:+d}",
            f"Height Slice {offset:+d}",
            f"Width Slice {offset:+d}"
        ]
        for ax, title in zip(axes[:, 0], row_titles):
            ax.set_ylabel(title, fontsize=12, rotation=90, labelpad=20)

        # Remove axis ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        # Save the figure
        plt.suptitle(f"3D Super-Resolution - Epoch {epoch} Batch {batch_idx} | Slice {offset:+d}")
        save_path = os.path.join(epoch_dir, f"output3D_epoch{epoch}_batch{batch_idx}_slice{offset:+d}.png")
        plt.savefig(save_path)
        plt.close()



def save_alignment_debug(inputs, targets, batch_idx, save_dir="./alignment_debug"):
    """Saves the same slice from axial, coronal, sagittal, and GT for debugging alignment."""
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    slice_idx = targets.shape[2] // 2  # Middle slice along depth

    # Extract the same slice from each view
    axial_slice = inputs["axial"][0, 0, slice_idx, :, :].float().cpu().numpy()
    coronal_slice = inputs["coronal"][0, 0, slice_idx, :, :].float().cpu().numpy()
    sagittal_slice = inputs["sagittal"][0, 0, slice_idx, :, :].float().cpu().numpy()
    gt_slice = targets[0, 0, slice_idx, :, :].float().cpu().numpy()  # GT in Axial orientation

    # Plot and save
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Axial", "Coronal", "Sagittal", "GT Iso"]
    slices = [axial_slice, coronal_slice, sagittal_slice, gt_slice]

    for ax, img, title in zip(axes, slices, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Alignment Debug - Batch {batch_idx}")
    plt.savefig(os.path.join(save_dir, f"alignment_batch_{batch_idx}.png"))
    plt.close()
