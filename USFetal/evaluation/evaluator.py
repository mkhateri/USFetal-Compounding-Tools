# metrics/evaluator.py

import torch
from typing import Dict, List, Union
from evaluation.registry import METRIC_REGISTRY
from evaluation.logging_setup import get_logger

logger = get_logger("Evaluator")

# Which metric keys are single-volume only (like "entropy")?
SINGLE_VOLUME_METRICS = {"entropy"}  


def squeeze_to_4d(t: torch.Tensor) -> torch.Tensor:
    """
    Squeeze out extra dimensions so that 't' ends up shape [1, D, H, W].
    """
    while t.dim() > 4:
        t = t.squeeze(0)
    return t


def evaluate_fusion_per_view(
    fused: torch.Tensor,
    views: torch.Tensor,
    pairwise_keys: List[str],
    log=logger
) -> Dict[str, List[float]]:
    """
    Evaluate 'fused' vs each view in 'views' for the given *pairwise* metrics,
    returning {metric: [score_v0, ..., score_vN-1, average]}.

    fused: shape [1, D, H, W]
    views: shape [V, 1, D, H, W] or [V, D, H, W]
    """
    num_views = views.shape[0]
    results: Dict[str, List[float]] = {}

    for metric in pairwise_keys:
        fn = METRIC_REGISTRY.get(metric)
        if fn is None:
            log.warning(f"Pairwise metric '{metric}' not found in registry.")
            results[metric] = [0.0]*(num_views+1)
            continue

        per_view_scores = []
        for v in range(num_views):
            view_tensor = squeeze_to_4d(views[v])
            try:
                # Ensure both fused and view_tensor are on the same device
                view_tensor = view_tensor.to(fused.device)
                score = fn(fused, view_tensor)
                per_view_scores.append(float(score))
            except Exception as e:
                log.error(f"Metric '{metric}' failed on view {v}: {e}")
                per_view_scores.append(0.0)

        avg_score = sum(per_view_scores)/num_views if num_views else 0.0
        results[metric] = per_view_scores + [avg_score]

    return results


def evaluate_single_volume_metrics(
    volume: torch.Tensor,
    single_keys: List[str],
    log=logger
) -> Dict[str, float]:
    """
    Compute single-volume metrics (e.g. "entropy") on 'volume'.
    Returns {metricName: float_value}.
    """
    results: Dict[str, float] = {}
    for metric in single_keys:
        fn = METRIC_REGISTRY.get(metric)
        if fn is None:
            log.warning(f"Single-volume metric '{metric}' not found in registry.")
            results[metric] = 0.0
            continue

        try:
            val = fn(volume.to(volume.device), None)
            results[metric] = float(val)
        except Exception as e:
            log.error(f"Metric '{metric}' failed: {e}")
            results[metric] = 0.0
    return results


def evaluate_fusion_all_metrics(
    fused: torch.Tensor,
    views: torch.Tensor,
    metric_keys: List[str],
    log=logger
) -> Dict[str, Union[List[float], float]]:
    """
    Evaluate both pairwise metrics (fused vs each view) and single-volume metrics on 'fused'.

    For pairwise metrics, we return a list of length (num_views+1).
    For single-volume metrics, we return a single float.

    Example returned dict:
        {
          "ssim": [0.64, 0.66, ..., 0.62],
          "cc": [0.87, 0.85, ..., 0.77],
          "entropy": 2.945,
          "MI": [...],
          ...
        }
    """
    # Split keys
    pairwise_keys = [k for k in metric_keys if k.lower() not in SINGLE_VOLUME_METRICS]
    single_keys   = [k for k in metric_keys if k.lower() in SINGLE_VOLUME_METRICS]

    pairwise_results = {}
    if pairwise_keys:
        pairwise_results = evaluate_fusion_per_view(fused, views, pairwise_keys, log)

    single_results = {}
    if single_keys:
        single_results = evaluate_single_volume_metrics(fused, single_keys, log)

    return {**pairwise_results, **single_results}


# ---------------------------------------------------------------------
# PRINTING FUNCTIONS
# ---------------------------------------------------------------------

def print_pairwise_metric_tables(
    all_eval_scores: Dict[str, Dict[str, Union[List[float], float]]],
    pairwise_metrics: List[str],
    num_views: int,
    log=logger
):
    """
    Print a table for each pairwise metric. 
    Show *only fused methods* in the rows (skip any "ViewN" rows),
    columns are v0..v(N-1), plus an 'Avg' column.
    """
    if not pairwise_metrics:
        return

    method_col_width = 25
    score_col_width = 8

    for metric in pairwise_metrics:
        heading = f"\nFinal Table for {metric.upper()}:\n"
        print(heading)
        log.info(heading.rstrip())

        header_fmt = (
            f"{{:<{method_col_width}}}" +
            "".join([f"{{:^{score_col_width}}}" for _ in range(num_views)]) +
            f"{{:^{score_col_width}}}"
        )
        header_cols = ["Method"] + [f"v{i}" for i in range(num_views)] + ["Avg"]
        header_line = header_fmt.format(*header_cols)
        print(header_line)
        log.info(header_line)

        for method_key, metric_dict in all_eval_scores.items():
            if method_key.startswith("View"):
                continue

            raw_val = metric_dict.get(metric, None)
            if not isinstance(raw_val, list) or len(raw_val) < (num_views+1):
                raw_val = [0.0]*(num_views+1)
            str_scores = [f"{v:.4f}" for v in raw_val]
            row_line = header_fmt.format(method_key, *str_scores)
            print(row_line)
            log.info(row_line)

        # print()
        # log.info("")


def print_single_volume_metric_table_horizontal(
    all_eval_scores: Dict[str, Dict[str, Union[List[float], float]]],
    single_metrics: List[str],
    log=logger
):
    """
    Print single-volume metrics horizontally:
      - Rows = each metric (e.g., ENTROPY)
      - Columns = each 'ViewN' and fused method key from all_eval_scores

    Example layout (if single_metrics = ["entropy"]):

    Final Table for Single-Volume Metrics (Horizontal):

                               View0       View1       View2       ...  mean[standard]  mean[patchwise]
    ENTROPY                    2.2373      2.4292      2.0896      ...  2.9450          2.9450
    """
    if not single_metrics:
        return  # nothing to print

    # Get column headers from the keys of all_eval_scores.
    columns = list(all_eval_scores.keys())
    # Calculate a suitable column width: at least 10, or as wide as the longest column name plus 2 spaces.
    col_width = max(10, max(len(col) for col in columns)) + 2

    heading = "\nFinal Table for Single-Volume Metrics (Horizontal):\n"
    print(heading)
    log.info(heading.rstrip())

    # Build header row: an empty left cell, then each column header.
    header_str = f"{'':{col_width}}"
    for col in columns:
        header_str += f"{col:>{col_width}}"
    print(header_str)
    log.info(header_str)

    # Build a row for each single-volume metric.
    for metric in single_metrics:
        row_str = f"{metric.upper():<{col_width}}"
        for col in columns:
            # Get the single-volume metric value (if stored; if it's a list, it's pairwise and we ignore it)
            val = all_eval_scores[col].get(metric, None) if col in all_eval_scores else None
            if isinstance(val, list):
                val = None
            if val is None:
                row_str += f"{'--':>{col_width}}"
            else:
                row_str += f"{val:>{col_width}.4f}"
        print(row_str)
        log.info(row_str)

    # print()
    # log.info("")
