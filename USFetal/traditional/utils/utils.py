import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from utils.logging_setup import get_logger
from typing import Dict, List
from typing import Dict, List, Union

logger = get_logger(__name__)

def convert_to_8bit(volume):
    """
    Normalize float32 volume to 0–255 using 99th percentile scaling.
    """
    p99 = np.percentile(volume, 99)
    if p99 > 0:
        volume = np.clip(volume / p99 * 255, 0, 255)
    return volume.astype(np.uint8)

def save_middle_slice_as_png(volume_3d, save_path):
    """
    Save central axial slice of 3D volume.
    """
    mid = volume_3d.shape[0] // 2
    plt.imsave(save_path, volume_3d[mid], cmap="gray")
    logger.info(f"Saved middle slice: {save_path}")

def save_nifti(volume, filename, output_dir, affine):
    """
    Save volume as NIfTI using given affine.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if affine.shape != (4, 4):
        logger.warning("Invalid affine shape; using identity.")
        affine = np.eye(4)
    
    img = nib.Nifti1Image(volume, affine)
    out_path = output_dir / filename
    nib.save(img, out_path)
    logger.info(f"Saved NIfTI: {out_path}")




def build_centered_table(all_eval_scores: Dict[str, Dict[str, float]], metric_keys: List[str]) -> List[str]:
    """
    Create a list of lines representing a single-row-per-method table
    with the method column left-aligned, and the metric columns center-aligned.

    Returns: list of strings (each string is one line), so you can:
      - Print them to console
      - Also write them to a file
    """
    # Adjust widths for your taste
    method_col_width = 30
    metric_col_width = 9

    # For the header, "Method" is left-aligned, metrics are center.
    #  :<30   => left aligned in 30 chars
    #  :^9    => centered in 9 chars
    header_format = (f"{{:<{method_col_width}}}" + 
                     "".join([f"{{:^{metric_col_width}}}" for _ in metric_keys]))

    # We’ll use the same format for data rows (Method left, metrics center)
    row_format = header_format

    lines = []
    # Header row
    header_cols = ["Method"] + [m.upper() for m in metric_keys]
    lines.append(header_format.format(*header_cols))

    # Data rows
    for method_key, scores in all_eval_scores.items():
        # Format each metric value as e.g. 0.6235 etc.
        row_values = [f"{scores.get(m, 0.0):.4f}" for m in metric_keys]
        lines.append(row_format.format(method_key, *row_values))

    return lines



def build_summary_table(all_eval_scores: Dict[str, Dict[str, Union[List[float], float]]],
                        metric_keys: List[str]) -> List[str]:
    """
    Build a summary table with each row representing a fused method (skip input views)
    and each column representing the average value of each metric.
    For pairwise metrics (stored as lists), we use the last element (the average).
    For single-volume metrics, we use the value directly.
    
    Returns a list of strings (each string is one row).
    """
    method_col_width = 30
    metric_col_width = 10

    # Only include fused methods (skip Views)
    methods = [key for key in all_eval_scores.keys() if not key.lower().startswith("view")]

    # Header row
    header = f"{'Method':<{method_col_width}}" + "".join(
        f"{metric.upper():^{metric_col_width}}" for metric in metric_keys
    )
    lines = [header]

    for method in methods:
        row = f"{method:<{method_col_width}}"
        for metric in metric_keys:
            val = all_eval_scores[method].get(metric, None)
            if isinstance(val, list):
                avg_val = val[-1] if val else 0.0
            else:
                avg_val = val if val is not None else 0.0
            row += f"{avg_val:^{metric_col_width}.4f}"
        lines.append(row)

    return lines

