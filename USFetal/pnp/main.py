import os
import glob
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import nibabel as nib

from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util


# import the default config
try:
    from pnp.configs.config_pnp import config_pnp   # when run from repo root
except ImportError:
    from configs.config_pnp import config_pnp       # when CWD is pnp/


AXES = {"Z": 2, "Y": 1, "X": 0}

"""
This script:
  1) Loads multi-view NIfTI volumes for each subject from data_parent/<subject>/volumes/*.nii.gz
     and optionally masks from data_parent/<subject>/masks/*.nii.gz (or *_mask.nii.gz).
  2) Applies masks per view (if available), computes the masked mean in memory.
  3) Runs DPIR/DRUNet pipeline on the mean volume and saves the outputs per subject.

"""

# ------------------------ I/O HELPERS ------------------------

def _find_mask_for_volume(vol_file: Path, masks_folder: Path) -> Optional[Path]:

    """
    For a volume <name>.nii.gz, try masks/<name>.nii.gz then masks/<name>_mask.nii.gz.
    If masks_folder doesn't exist, try next to the volume file.
    """
    name = vol_file.name  # e.g., 0180_10.nii.gz
    base = name.replace(".nii.gz", "")
    candidates = []
    if masks_folder.exists():
        candidates = [
            masks_folder / f"{base}.nii.gz",
            masks_folder / f"{base}_mask.nii.gz",
        ]
    else:
        candidates = [
            vol_file.parent / f"{base}.nii.gz",
            vol_file.parent / f"{base}_mask.nii.gz",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_views_and_masks(subject_dir: str, apply_mask: bool = True,
                         binarize_mask: bool = True, mask_thr: float = 0.0):
    """
    Loads all views from <subject_dir>/volumes (sorted by name).
    If apply_mask=True, multiplies each view by its mask if found in <subject_dir>/masks or alongside the volume.

    Returns:
      views: np.ndarray [V, D, H, W]
      affine: np.ndarray (from first view)
      first_nii: nib.Nifti1Image (for header copying if needed)
      volume_files: list[Path]
    """
    subj = Path(subject_dir).resolve()
    vol_dir = subj / "volumes"
    mask_dir = subj / "masks"

    volume_files = sorted(vol_dir.glob("*.nii.gz"))
    if len(volume_files) == 0:
        raise FileNotFoundError(f"No volumes found in: {vol_dir}")

    views = []
    first_aff = None
    first_nii = None

    for i, vf in enumerate(volume_files):
        nii = nib.load(str(vf))
        vol = nii.get_fdata(dtype=np.float32)  # [D,H,W] or [X,Y,Z] (assume consistent)
        if i == 0:
            first_aff = nii.affine
            first_nii = nii

        if apply_mask:
            mf = _find_mask_for_volume(vf, mask_dir)
            if mf is not None:
                mnii = nib.load(str(mf))
                msk = mnii.get_fdata(dtype=np.float32)
                if binarize_mask:
                    msk = (msk > mask_thr).astype(np.float32)
                vol = vol * msk  # masked-view

        views.append(vol)

    views = np.stack(views, axis=0)  # [V, D, H, W]
    return views, first_aff, first_nii, volume_files

def masked_mean(views: np.ndarray) -> np.ndarray:
    """views: [V, D, H, W] -> mean: [D, H, W]"""
    return views.mean(axis=0)

# ------------------------ DPIR PROCESSOR ------------------------

def run_dpir_on_volume(nii_data: np.ndarray, affine: np.ndarray, subject_id: str,
                       output_dir: str, experiment_tag: str = 'sr_drunet_gray',
                       noise_level_img: int = 3, model_name: str = 'drunet_gray',
                       sf: int = 1, iter_num: int = 10,
                       ):
    """
    Runs the DPIR/DRUNet pipeline on an in-memory 3D volume (nii_data: [D,H,W]) and saves outputs under:
      output_dir/<experiment_tag>/<subject_id>/
    """

    # Parameters / kernel
    kernel_width_default_x1234 = [0.6, 0.9, 1.7, 2.2]
    noise_level_model = noise_level_img / 255.0
    kernel_width = kernel_width_default_x1234[sf-1]
    kernel_width = 0.5  # override if you want

    k = utils_deblur.fspecial('gaussian', 13, kernel_width)
    k = sr.shift_pixel(k, sf)
    k /= np.sum(k)

    x8 = True
    modelSigma1 = 49
    modelSigma2 = max(sf, noise_level_model * 255.)
    classical_degradation = True

    n_channels = 1 if 'gray' in model_name else 3
    model_zoo = 'pnp/model_zoo'

    # Experiment naming and output folder
    result_name = f"{experiment_tag}"
    E_path = os.path.join(output_dir, result_name, subject_id)
    util.mkdir(E_path)

    # Logger (fresh handlers per subject)
    logger_name = result_name
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        logger.handlers.clear()
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # Load model
    model_path = os.path.join(model_zoo, model_name + '.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    if 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512],
                    nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    elif 'ircnn' in model_name:
        from models.network_dncnn import IRCNN as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # logger.info(f'Model: {model_name}, noise_img: {noise_level_img:.3f}, noise_model: {noise_level_model:.3f}')
    # logger.info(f'Model path: {model_path}')

    # Prepare mask (if the mean is already masked this is fine; it will preserve brain support)
    nii_data = np.asarray(nii_data, dtype=np.float32)
    mask_data = (nii_data / 255.0 > 1e-4).astype(np.float32)

    temp_res = []

    # Process along Z/Y/X
    for axis_name, axis_index in AXES.items():
        # logger.info(f"\n--- Processing along {axis_name} axis ---")

        vol = np.moveaxis(nii_data, axis_index, -1)  # [H, W, D]
        H, W, D = vol.shape
        output_vol = np.zeros((H, W, D), dtype=np.float32)

        for z in range(D):
            # logger.info(f'Processing slice {z+1}/{D}')
            img_L = vol[:, :, z].astype(np.float32)

            # Pad to /8
            pad_h = (8 - img_L.shape[0] % 8) % 8
            pad_w = (8 - img_L.shape[1] % 8) % 8
            pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
            pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

            img_L = np.pad(img_L, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
            img_L = (img_L - img_L.min()) / (img_L.max() - img_L.min() + 1e-8)

            rhos, sigmas = pnp.get_rho_sigma(
                sigma=max(0.255/255., noise_level_model),
                iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1
            )
            rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

            x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
            if np.ndim(x) == 2:
                x = x[..., None]

            if classical_degradation:
                x = sr.shift_pixel(x, sf)
            x = util.single2tensor4(x).to(device)

            if img_L.ndim == 2:
                img_L = img_L[..., np.newaxis]

            img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
            [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

            for i in range(iter_num):
                tau = rhos[i].float().repeat(1, 1, 1, 1)
                x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

                if 'ircnn' in model_name:
                    current_idx = int(np.ceil(sigmas[i].cpu().numpy()*255./2.) - 1)
                    if current_idx != former_idx:
                        model.load_state_dict(model25[str(current_idx)], strict=True)
                        model.eval()
                        for _, v in model.named_parameters():
                            v.requires_grad = False
                        model = model.to(device)
                    former_idx = current_idx

                if x8:
                    x = util.augment_img_tensor4(x, i % 8)

                if 'drunet' in model_name:
                    x = torch.cat((x, sigmas[i].repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                    x = utils_model.test_mode(model, x, mode=2, refield=64, min_size=256, modulo=16)
                elif 'ircnn' in model_name:
                    x = model(x)

                if x8:
                    if i % 8 in (3, 5):
                        x = util.augment_img_tensor4(x, 8 - i % 8)
                    else:
                        x = util.augment_img_tensor4(x, i % 8)

            img_E = util.tensor2single(x)
            img_E = img_E[pad_top:img_E.shape[0]-pad_bottom, pad_left:img_E.shape[1]-pad_right]
            output_vol[:, :, z] = img_E

        # Restore axis and apply mask
        mask_moved = np.moveaxis(mask_data, axis_index, -1)
        output_vol = output_vol * mask_moved
        output_vol = np.moveaxis(output_vol, -1, axis_index)

        # out_img = nib.Nifti1Image(output_vol, affine=affine)
        # nib.save(out_img, os.path.join(E_path, f"{subject_id}_sr_{axis_name.lower()}.nii"))
        # logger.info(f"Saved compounded volume to {subject_id}_sr_{axis_name.lower()}.nii")
        temp_res.append(output_vol)

    # Mean of Z/Y/X outputs
    avg_vol = np.mean(np.stack(temp_res, axis=0), axis=0)
    avg_vol = avg_vol * mask_data
    avg_img = nib.Nifti1Image(avg_vol, affine=affine)
    mean_out_name = f"{subject_id}_pnp.nii.gz"
    nib.save(avg_img, os.path.join(E_path, mean_out_name))
    logger.info(f"Saved compounded volume to {mean_out_name}")

# ------------------------ DRIVER ------------------------
def override_cfg(cfg, **kwargs):
    """Shallow override for top-level keys + nested dpir/io/axes."""
    out = {**cfg}
    if "data_parent" in kwargs and kwargs["data_parent"]: out["data_parent"] = kwargs["data_parent"]
    if "output_dir" in kwargs and kwargs["output_dir"]: out["output_dir"] = kwargs["output_dir"]
    if "model_zoo" in kwargs and kwargs["model_zoo"]: out["model_zoo"] = kwargs["model_zoo"]
    if "experiment_tag" in kwargs and kwargs["experiment_tag"]: out["experiment_tag"] = kwargs["experiment_tag"]
    if "device" in kwargs and kwargs["device"] != "auto": out["device"] = kwargs["device"]

    if "axes" in kwargs and kwargs["axes"]:
        out["axes"] = kwargs["axes"]

    dpir = {**out["dpir"]}
    for k in ["model_name", "noise_level_img", "sf", "iter_num",
              "kernel_size", "kernel_sigma", "override_kernel_sigma"]:
        if k in kwargs and kwargs[k] is not None:
            dpir[k] = kwargs[k]
    out["dpir"] = dpir

    io = {**out["io"]}
    for k in ["apply_mask", "binarize_mask", "mask_thr"]:
        if k in kwargs and kwargs[k] is not None:
            io[k] = kwargs[k]
    out["io"] = io
    return out

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("USFetal PnP")
    # paths
    parser.add_argument("--data_parent", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_zoo", type=str, default=None)
    parser.add_argument("--experiment_tag", type=str, default=None)
    # device
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    # axes (comma-separated, e.g., Z,Y,X or just Z)
    parser.add_argument("--axes", type=str, default=None)
    # dpir
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--noise_level_img", type=int, default=None)
    parser.add_argument("--sf", type=int, default=None)
    parser.add_argument("--iter_num", type=int, default=None)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--kernel_sigma", type=float, default=None)
    parser.add_argument("--override_kernel_sigma", action="store_true")
    # io
    parser.add_argument("--no_mask", action="store_true", help="disable apply_mask")

    args = parser.parse_args()

    # parse axes
    axes = None
    if args.axes:
        axes = [a.strip().upper() for a in args.axes.split(",") if a.strip()]

    # build overrides
    overrides = dict(
        data_parent=args.data_parent,
        output_dir=args.output_dir,
        model_zoo=args.model_zoo,
        experiment_tag=args.experiment_tag,
        device=args.device,
        axes=axes,
        model_name=args.model_name,
        noise_level_img=args.noise_level_img,
        sf=args.sf,
        iter_num=args.iter_num,
        kernel_size=args.kernel_size,
        kernel_sigma=args.kernel_sigma,
        override_kernel_sigma=args.override_kernel_sigma if args.override_kernel_sigma else None,
        apply_mask=(False if args.no_mask else None),
    )

    # merge config + overrides
    cfg = override_cfg(config_pnp, **overrides)

    data_parent    = Path(cfg["data_parent"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    model_zoo    = Path(cfg["model_zoo"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # device info (optional)
    dev_choice = cfg.get("device", "auto")
    if dev_choice == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_choice)
    print(f"[PnP] Using device: {device}")
    if device.type == "cuda":
        print(f"[PnP] GPU: {torch.cuda.get_device_name(0)}")

    subjects = sorted([p for p in data_parent.iterdir() if p.is_dir()])
    print(f"Found {len(subjects)} subjects to process under: {data_parent}")

    for subj in subjects:
        subject_id = subj.name
        print(f"\nSubject: {subject_id}")

        views, affine, _, _ = load_views_and_masks(
            str(subj),
            apply_mask=cfg["io"]["apply_mask"],
            binarize_mask=cfg["io"]["binarize_mask"],
            mask_thr=cfg["io"]["mask_thr"]
        )
        mean_vol = masked_mean(views)

        dp = cfg["dpir"]
        run_dpir_on_volume(
            mean_vol, affine, subject_id, str(output_dir),
            experiment_tag=cfg["experiment_tag"],
            noise_level_img=int(dp["noise_level_img"]),
            model_name=str(dp["model_name"]),
            sf=int(dp["sf"]),
            iter_num=int(dp["iter_num"]),
            # keep your other defaults inside run_dpir_on_volume
        )
