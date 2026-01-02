import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.cuda.amp import autocast

# For your DataLoader fix, ensure "bf16" maps to torch.bfloat16 in your dataloader_inference.py
from ssl_scan_specific.data.dataloader_inference import get_test_dataloader
from models.model_builder import ModelBuilder
from metrics import MetricsEvaluator
from utils.device_handler import get_device
from utils.logger import setup_logger
from ssl_scan_specific.utils.config_handler import set_config

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def patchify_multiview(volume_6d, patch_size, overlap):
    """Patchify a 6D tensor [B, V, C, D, H, W]."""
    B, V, C, D, H, W = volume_6d.shape
    pD, pH, pW = patch_size
    oD, oH, oW = overlap

    patches = []
    d_indices, h_indices, w_indices = [], [], []

    for d in range(0, D, pD - oD):
        d_start = min(d, D - pD)
        d_indices.append(d_start)
        for h in range(0, H, pH - oH):
            h_start = min(h, H - pH)
            h_indices.append(h_start)
            for w in range(0, W, pW - oW):
                w_start = min(w, W - pW)
                w_indices.append(w_start)

                patch = volume_6d[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
                patches.append(patch)

                #logging.debug(f"Patch extracted at d={d_start}, h={h_start}, w={w_start}, shape={patch.shape}")

    #logging.debug(f"Patchify: total patches extracted = {len(patches)}")
    return patches


def generate_weight_map(patch_size, overlap, method="gaussian", normalization_method="sum"):
    pD, pH, pW = patch_size
    oD, oH, oW = overlap

    grid = np.ones((pD, pH, pW), dtype=np.float32)

    if method == "gaussian":
        dz = np.exp(-0.5 * ((np.linspace(-1, 1, pD)) ** 2) / (0.5 ** 2))
        dy = np.exp(-0.5 * (np.linspace(-1, 1, pH) ** 2) / (0.5 ** 2))
        dx = np.exp(-0.5 * (np.linspace(-1, 1, pW) ** 2) / (0.5 ** 2))

        grid = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
        grid /= grid.max()
    else:
        grid = np.ones((pD, pH, pW), dtype=np.float32)

    if normalization_method == "sum":
        grid /= grid.sum()
    elif normalization_method == "max":
        grid /= grid.max()

    weight_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return weight_tensor


def unpatchify_multiview(patches, patch_size, overlap, output_shape, weight_method="gaussian", normalization_method="sum"):
    B, V, C, D, H, W = output_shape
    pD, pH, pW = patch_size
    oD, oH, oW = overlap

    if isinstance(patches, list):
        patches = torch.stack(patches, dim=0)

    weight_map = generate_weight_map(patch_size, overlap, method=weight_method, normalization_method=normalization_method).to(patches.device)

    recon = torch.zeros(output_shape, dtype=patches.dtype, device=patches.device)
    weight_acc = torch.zeros(output_shape, dtype=patches.dtype, device=patches.device)

    d_indices = [min(d, D - pD) for d in range(0, D, pD - oD)]
    h_indices = [min(h, H - pH) for h in range(0, H, pH - oH)]
    w_indices = [min(w, W - pW) for w in range(0, W, pW - oW)]

    idx = 0

    for d_start in d_indices:
        for h_start in h_indices:
            for w_start in w_indices:
                if idx >= patches.shape[0]:
                    break
                recon_slice = recon[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
                patch = patches[idx]
                recon[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW] += patch * weight_map
                weight_acc[:, :, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW] += weight_map
                idx += 1

    weight_acc = torch.where(weight_acc == 0, torch.ones_like(weight_acc), weight_acc)
    recon /= weight_acc

    # logger.debug(f"Unpatchify: Final reconstructed shape: {recon.shape}")
    return recon
########################################################
# HELPER FUNCTIONS (8-bit conversion, saving NIfTI, debug slice)
########################################################
def convert_to_8bit(volume):
    """
    Convert a float32 numpy array 'volume' to 8-bit using the 99th percentile.
    """
    p99 = np.percentile(volume, 99)
    if p99 > 0:
        volume = (volume / p99 * 255).clip(0, 255)
    return volume.astype(np.uint8)




def save_middle_slice_as_png(volume_3d, save_path):
    """
    Save the middle slice of a 3D volume [D, H, W] as a PNG.
    """
    d = volume_3d.shape[0]
    mid_slice = volume_3d[d // 2]
    plt.imsave(save_path, mid_slice, cmap="gray")
    print(f"[DEBUG] Middle slice saved => {save_path}")

def save_nifti(volume, filename, outdir, affine):
    """
    Save 'volume' as a NIfTI file.
    Expects 'volume' to be a 3D numpy array [D, H, W].
    Forces the affine to be a (4,4) numpy array of type float64.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Force affine to be a numpy array of type float64.
    affine = np.array(affine, dtype=np.float64)
    affine = np.squeeze(affine)
    if affine.ndim != 2 or affine.shape != (4, 4):
        print("[WARNING] Affine is not shape (4,4); using identity.")
        affine = np.eye(4, dtype=np.float64)
    
    # If volume is not 3D, squeeze extra dimensions.
    if volume.ndim != 3:
        volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected final volume to be 3D, but got shape {volume.shape}")
    
    nii = nib.Nifti1Image(volume, affine)
    out_path = outdir / f"{Path(filename).stem}_pred.nii.gz"
    nib.save(nii, str(out_path))
    print(f"[INFO] Saved => {Path(filename).stem}_pred.nii.gz")

########################################################
# MAIN INFERENCE CLASS
########################################################
class Inference:
    def __init__(self, config):
        # self.config = config
        self.config = set_config(config)

        self.device = get_device()
        self.amp_dtype = self._get_amp_dtype()

        self.output_dir = Path(config["inference"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.output_dir / "inference.log")

        self._infer_num_views()
        self.model = self._load_model()
        self.test_loader = get_test_dataloader(
            config["data"],
            batch_size=config["inference"].get("batch_size", 1)
        )

        self.patch_size = config["patching"].get("patch_size", [96, 96, 96])
        self.overlap = config["patching"].get("overlap", [48, 48, 48])
        self.metrics_evaluator = MetricsEvaluator(config["metrics"], device=self.device)

    def _get_amp_dtype(self):
        """
        Determine AMP dtype for inference.
        Falls back safely if training section is absent.
        """
        precision = (
            self.config
            .get("training", {})
            .get("amp_precision", self.config["data"].get("dtype", "fp32"))
            .lower()
        )

        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        else:
            return torch.float32


    def _load_model(self):
        model_path = Path(self.config["inference"]["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f" Model checkpoint not found at {model_path}")
        print(f"Loading Model from {model_path}")
        model = ModelBuilder(self.config["model"]).get_model().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        print(f"Checkpoint keys: {checkpoint.keys()}")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            print("Warning: No 'model_state_dict' found! Loading entire checkpoint.")
            model.load_state_dict(checkpoint, strict=False)
        return model

    def run(self):
        """
        Main inference:
         - For each batch [B,V,C,D,H,W]:
           - Patchify → process patches one by one → unpatchify.
         - Save a debug middle slice as PNG.
         - Convert final fused volume to 8-bit and save as NIfTI using the original affine.
        """
        self.model.eval()
        torch.cuda.empty_cache()

        debug_save_dir = self.output_dir / "debug_slices"
        debug_save_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in self.test_loader:
                views, masks, filenames, affines = self._extract_batch_data(batch)

                # Multiply by masks if provided
                #if masks is not None:
                views = views * masks

                # Process the volume via patch-based inference.
                # Expected fused output shape: [B, 1, C, D, H, W]
                recon_volume = self.process_one_patch_at_a_time(views)

                # Convert BF16 -> float32, move to CPU, convert to numpy.
                recon_fp32 = recon_volume.to(torch.float32).cpu()
                vol_np = recon_fp32.numpy()  # Expected shape: [B, 1, C, D, H, W]

                # For debugging: save the middle slice (from the fused output).
                # volume_3d = vol_np[0, 0, 0]  # => [D, H, W]
                # mid_slice_path = debug_save_dir / "middle_slice.png"
                # save_middle_slice_as_png(volume_3d, mid_slice_path)

                # Print volume stats before conversion.
                # print("[DEBUG] Volume shape before convert_to_8bit:", vol_np.shape)
                # print("[DEBUG] Volume stats → min:", vol_np.min(), "max:", vol_np.max(), "mean:", vol_np.mean())
                p99 = np.percentile(vol_np, 99)
                # print("[DEBUG] p99:", p99)

                # Squeeze extra dimensions to get a pure 3D volume.
                vol_squeezed = np.squeeze(vol_np, axis=(0, 1, 2))  # => [D, H, W]
                vol_uint8 = convert_to_8bit(vol_squeezed)

                filename = filenames[0] if filenames else "sample"
                affine = affines[0] if affines else np.eye(4)
                save_nifti(vol_uint8, filename, self.output_dir, affine)
                
                # Optionally compute metrics...
                # break  # Process only one batch; remove this to process all batches.

    def process_one_patch_at_a_time(self, volume_6d):
        """
        1) Patchify the 6D volume [B,V,C,D,H,W] into patches of shape [B,V,C,pD,pH,pW].
        2) Process each patch individually (to avoid OOM).
        3) Reconstruct (unpatchify) the fused volume.
           Here, output_shape is (B, 1, C, D, H, W) because the model fuses the views.
        """
        B, V, C, D, H, W = volume_6d.shape
        patches_list = patchify_multiview(volume_6d, self.patch_size, self.overlap)
        out_patches = []
        for i, patch in enumerate(patches_list):
            patch = patch.to(self.device, non_blocking=True)
            with autocast(dtype=self.amp_dtype):
                out_patch = self.model(patch)  # Expected output: [B, 1, C, pD, pH, pW]
            out_patches.append(out_patch.cpu())
        # Stack patch outputs: [num_patches, B, 1, C, pD, pH, pW]
        out_patches_tensor = torch.stack(out_patches, dim=0)
        recon_6d = unpatchify_multiview(out_patches_tensor, self.patch_size, self.overlap, (B, 1, C, D, H, W))
        # print("Final combined shape:", recon_6d.shape)
        return recon_6d

    def _extract_batch_data(self, batch):
        """
        Ensure views have shape [B,V,C,D,H,W]. If dimensions are missing, unsqueeze them.
        """
        views = batch["views"].to(self.device)
        if views.ndim == 5:  # [B,C,D,H,W] -> add V=1
            B, C, D, H, W = views.shape
            views = views.view(B, 1, C, D, H, W)
        elif views.ndim == 4:  # [C,D,H,W] -> add B=1,V=1
            views = views.unsqueeze(0).unsqueeze(0)
        elif views.ndim == 3:
            views = views.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        filenames = batch.get("filename", [])
        if isinstance(filenames, str):
            filenames = [filenames]

        aff_data = batch.get("affine", torch.eye(4))
        if not isinstance(aff_data, list):
            aff_list = [aff_data]
        else:
            aff_list = aff_data

        masks = batch.get("masks", None)
        if masks is not None:
            masks = masks.to(self.device)

        return views, masks, filenames, aff_list

    def _compute_and_log_metrics(self, out_uint8, masks, filename, debug_dir):
        if masks is not None:
            GT_np = convert_to_8bit(masks.cpu().numpy())
            GT_tensor = torch.tensor(GT_np, dtype=torch.float32, device=self.device) / 255.0
            out_tensor = torch.tensor(out_uint8, dtype=torch.float32, device=self.device) / 255.0
            metrics = self.metrics_evaluator.compute_metrics(out_tensor, GT_tensor)
            self.logger.info(f"Metrics for {filename}:\n{json.dumps(metrics, indent=4)}")
            debug_fig_save(GT_np, out_uint8, filename, debug_dir)
        else:
            self.logger.info(f"No mask => skipping metrics for {filename}.")

    def _infer_num_views(self):
        """
        Infer number of views (V) from the test dataloader
        and update config in-place.
        """
        test_loader = get_test_dataloader(
            self.config["data"],
            batch_size=1
        )

        batch = next(iter(test_loader))
        views = batch["views"]  # expected [B, V, C, D, H, W] or similar

        if views.ndim == 6:
            num_views = views.shape[1]
        elif views.ndim == 5:
            num_views = 1
        else:
            raise ValueError(f"Unexpected views shape: {views.shape}")

        self.config["model"]["num_views"] = num_views
        # self.logger.info(f"[Inference] Inferred num_views = {num_views}")


def debug_fig_save(gt, output_uint8, filename, savedir):
    """
    Save a center slice from GT and predicted output as a PNG for debugging.
    """
    slice_idx = gt.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gt[slice_idx], cmap="gray")
    ax[0].set_title("GT")
    ax[0].axis("off")
    ax[1].imshow(output_uint8[slice_idx], cmap="gray")
    ax[1].set_title("Pred")
    ax[1].axis("off")
    path = savedir / f"{Path(filename).stem}_debug.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Debug slice => {path}")


if __name__ == "__main__":
    import argparse
    import importlib.util
    from pathlib import Path

    # ============================================================
    # Helpers
    # ============================================================
    def resolve_path(p: str) -> str:
        """Expand user and convert to absolute path."""
        return str(Path(p).expanduser().resolve())

    # ============================================================
    # Default config path
    # ============================================================
    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "config_ssl_inference.py"
    )

    # ============================================================
    # Load config from .py
    # ============================================================
    def load_config_from_py(path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        spec = importlib.util.spec_from_file_location("config_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "config"):
            raise AttributeError(f"{path} must define a `config` dictionary")

        return module.config

    # ============================================================
    # Argument parser (explicit & human-friendly)
    # ============================================================
    parser = argparse.ArgumentParser(
        description="SSL Fetal Ultrasound Inference"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to inference config file",
    )

    # ---- Common inference options users actually change ----
    parser.add_argument("--data_dir", type=str, help="Input data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Checkpoint path")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--patch_size", type=int, nargs=3, help="Patch size: D H W")
    parser.add_argument("--overlap", type=int, nargs=3, help="Patch overlap: D H W")

    args = parser.parse_args()

    # ============================================================
    # Load base config (defaults)
    # ============================================================
    config = load_config_from_py(args.config)

    # ============================================================
    # Apply CLI values (ONLY if provided)
    # ============================================================
    if args.data_dir is not None:
        config["data"]["data_dir"] = resolve_path(args.data_dir)

    if args.output_dir is not None:
        config["inference"]["output_dir"] = resolve_path(args.output_dir)

    if args.model_path is not None:
        config["inference"]["model_path"] = resolve_path(args.model_path)

    if args.batch_size is not None:
        config["inference"]["batch_size"] = args.batch_size

    if args.patch_size is not None:
        config["patching"]["patch_size"] = list(args.patch_size)

    if args.overlap is not None:
        config["patching"]["overlap"] = list(args.overlap)

    # ============================================================
    # Run inference
    # ============================================================
    runner = Inference(config)
    runner.run()
