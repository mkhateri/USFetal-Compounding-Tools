from pathlib import Path
import pandas as pd
import torch
import nibabel as nib
import numpy as np
import logging
import argparse

from evaluation.evaluator import evaluate_fusion_all_metrics, SINGLE_VOLUME_METRICS
from evaluation.metric_utils import (
    print_pairwise_metric_tables,
    print_single_volume_metric_table_horizontal,
    build_summary_table
)


def get_logger(name="iq_logger", log_file=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


logger = get_logger("IQAssess")


def load_nifti_tensor(file_path):
    img = nib.load(str(file_path))
    data = img.get_fdata(dtype=np.float32)
    return torch.tensor(data), img.affine


def load_sample(sample_id, data_root, results_root, apply_mask=True):
    data_root = Path(data_root)
    results_root = Path(results_root)
    sample_path = data_root / sample_id
    volumes_dir = sample_path / "volumes"
    masks_dir = sample_path / "masks"
    results_sample_dir = results_root / sample_id

    info = {
        "sample_id": sample_id,
        "volumes_found": 0,
        "volume_files": [],
        "masks_applied": [],
        "results": {},
        "warnings": [],
    }

    if not volumes_dir.exists():
        info["warnings"].append(f"Volumes folder missing: {volumes_dir}")
        return None, None, None, info

    volume_files = sorted(volumes_dir.glob("*.nii.gz"))
    if not volume_files:
        info["warnings"].append(f"No volume files found in: {volumes_dir}")
        return None, None, None, info

    views = []
    affines = []
    info["volumes_found"] = len(volume_files)

    for vol_file in volume_files:
        base = vol_file.name.replace(".nii.gz", "")
        vol, affine = load_nifti_tensor(vol_file)
        vol = vol.unsqueeze(0)
        affines.append(affine)

        mask_applied = False
        if apply_mask:
            candidates = [
                masks_dir / f"{base}.nii.gz",
                masks_dir / f"{base}_mask.nii.gz"
            ]
            mask_file = next((f for f in candidates if f.exists()), None)
            if mask_file:
                mask, _ = load_nifti_tensor(mask_file)
                mask = mask.unsqueeze(0)
                vol *= mask
                mask_applied = True
                logger.info(f"[{sample_id}] Applied mask: {mask_file.name}")
            else:
                logger.warning(f"[{sample_id}] No mask found for: {vol_file.name}")

        info["volume_files"].append(vol_file.name)
        info["masks_applied"].append(mask_applied)
        views.append(vol)

    views_tensor = torch.stack(views, dim=0)
    logger.info(f"[{sample_id}] Loaded {len(views)} volumes. Shape: {views_tensor.shape}")

    if not results_sample_dir.exists():
        info["warnings"].append(f" Results folder not found: {results_sample_dir}")
        return views_tensor, affines[0], [], info

    result_files = sorted(results_sample_dir.glob(f"{sample_id}_*.nii.gz"))
    for res_file in result_files:
        name = res_file.stem
        parts = name.split("_", 1)
        if len(parts) != 2:
            info["warnings"].append(f" Skipping malformed result: {res_file.name}")
            continue
        method_name = parts[1]
        try:
            result_tensor, _ = load_nifti_tensor(res_file)
            result_tensor = result_tensor.unsqueeze(0)
            info["results"][method_name] = {
                "exists": True,
                "shape": list(result_tensor.shape),
                "path": str(res_file)
            }
            logger.info(f"[{sample_id}] Loaded result: {method_name}")
        except Exception as e:
            info["results"][method_name] = {
                "exists": False,
                "error": str(e)
            }
            info["warnings"].append(f" Failed to load {res_file.name}: {e}")

    return views_tensor, affines[0], result_files, info


def run_iq_assessment(sample_id, data_root, results_root, metric_keys, csv_accumulator=None):
    views, affine, result_files, info = load_sample(
        sample_id=sample_id,
        data_root=data_root,
        results_root=results_root,
        apply_mask=True
    )

    if views is None:
        logger.error(f" Sample {sample_id} failed to load. Skipping.")
        return

    all_scores = {}
    for method, meta in info["results"].items():
        try:
            result_tensor, _ = load_nifti_tensor(meta["path"])
            result_tensor = result_tensor.unsqueeze(0)
            result_metrics = evaluate_fusion_all_metrics(result_tensor, views, metric_keys)
            all_scores[method] = result_metrics
        except Exception as e:
            logger.error(f" Evaluation failed for method {method}: {e}")

    if not all_scores:
        logger.warning(f" No results found/evaluated for sample {sample_id}.")
        return

    num_views = views.shape[0]
    pairwise_metrics = [m for m in metric_keys if m.lower() not in SINGLE_VOLUME_METRICS]
    single_metrics = [m for m in metric_keys if m.lower() in SINGLE_VOLUME_METRICS]

    print_pairwise_metric_tables(all_scores, pairwise_metrics, num_views)
    print_single_volume_metric_table_horizontal(all_scores, single_metrics)

    summary_lines = build_summary_table(all_scores, metric_keys, num_views)
    summary_path = Path(results_root) / f"{sample_id}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f" Summary saved to: {summary_path}")

    if csv_accumulator is not None:
        for method, scores in all_scores.items():
            row = {"sample_id": sample_id, "method": method}
            for metric, vals in scores.items():
                row[metric] = float(vals[-1]) if isinstance(vals, list) else float(vals)
            csv_accumulator.append(row)


# -------------------------------
# Helpers to discover all samples
# -------------------------------
def discover_samples(data_root: Path):
    data_root = Path(data_root)
    def has_volumes(p: Path):
        vdir = p / "volumes"
        return p.is_dir() and vdir.is_dir() and any(vdir.glob("*.nii.gz"))
    return sorted(p.name for p in data_root.iterdir() if has_volumes(p))


# -------------------------------------------------------------------
# Run script over ALL samples
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IQ assessment for fusion results")

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing input samples (with volumes/ and masks/)"
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default="./output_shared",
        help="Root directory containing fusion results"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()

    metric_keys = ["ssim", "psnr", "cc", "entropy", "mi"]
    results_root.mkdir(parents=True, exist_ok=True)

    # Discover and run every sample under data_root
    sample_ids = discover_samples(data_root)
    logger.info(f" Discovered {len(sample_ids)} samples: {sample_ids}")

    rows = []
    for sid in sample_ids:
        logger.info(f"\n=== Evaluating sample {sid} ===")
        run_iq_assessment(
            sample_id=sid,
            data_root=data_root,
            results_root=results_root,
            metric_keys=metric_keys,
            csv_accumulator=rows
        )

    # Save combined CSV and pretty tables
    if rows:
        from tabulate import tabulate

        df = pd.DataFrame(rows)

        # Save full numeric CSV (keep precision)
        csv_path = results_root / "summary_all_samples.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n Overall CSV saved to: {csv_path}")

        # Pretty tables (4 significant figures in text rendering)
        # Per-sample pretty tables:
        for sid in sorted(set(df["sample_id"])):
            df_current = df[df["sample_id"] == sid].drop(columns=["sample_id"])
            tex_table = tabulate(df_current, headers="keys", tablefmt="latex", showindex=False, floatfmt=".4g")
            txt_table = tabulate(df_current, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4g")
            (results_root / f"{sid}_summary.tex").write_text(tex_table)
            (results_root / f"{sid}_summary.txt").write_text(txt_table)
            print(f" LaTeX table saved to: {sid}_summary.tex")
            print(f" Pretty text table saved to: {sid}_summary.txt")

        # Also an all-samples pretty table
        text_all = tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4g")
        (results_root / "summary_all_samples.txt").write_text(text_all)
        print(" Pretty text table for ALL samples saved to: summary_all_samples.txt")

        # ---------------------------------------------------------
        # Mean ± Std summary across subjects (grouped by method)
        # ---------------------------------------------------------
        # Identify metric columns (exclude identifiers)
        metrics_cols = [c for c in df.columns if c not in ("sample_id", "method")]

        # 1) CSV with separate mean and std columns per metric
        agg = df.groupby("method")[metrics_cols].agg(["mean", "std"])
        # flatten MultiIndex columns: e.g., ('ssim','mean')->'ssim_mean'
        agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
        csv_meanstd_path = results_root / "summary_all_samples_mean_std.csv"
        agg.to_csv(csv_meanstd_path)
        print(f" Mean/Std CSV saved to: {csv_meanstd_path}")

        # 2) Pretty tables with "mean ± std"
        def fmt_mu_sigma(mu, sd):
            return f"{mu:.3g} ± {sd:.3g}"

        pretty = pd.DataFrame(index=agg.index)
        for m in metrics_cols:
            mu_col = f"{m}_mean"
            sd_col = f"{m}_std"
            pretty[m] = [
                fmt_mu_sigma(agg.loc[idx, mu_col], agg.loc[idx, sd_col])
                for idx in agg.index
            ]

        tex_pretty = tabulate(pretty, headers="keys", tablefmt="latex", showindex=True)
        txt_pretty = tabulate(pretty, headers="keys", tablefmt="fancy_grid", showindex=True)

        tex_meanstd_path = results_root / "summary_all_samples_meanstd.tex"
        txt_meanstd_path = results_root / "summary_all_samples_meanstd.txt"
        tex_meanstd_path.write_text(tex_pretty)
        txt_meanstd_path.write_text(txt_pretty)
        print(f" LaTeX mean±std table saved to: {tex_meanstd_path.name}")
        print(f" Pretty text mean±std table saved to: {txt_meanstd_path.name}")

    else:
        logger.warning(" No rows to save. Did any samples evaluate successfully?")

