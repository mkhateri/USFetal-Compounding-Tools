# metrics_eval/metric_utils.py

import logging
from typing import Dict, List, Union

logger = logging.getLogger("MetricUtils")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def print_pairwise_metric_tables(
    all_eval_scores: Dict[str, Dict[str, Union[List[float], float]]],
    pairwise_metrics: List[str],
    num_views: int,
    log=logger
):
    """
    Print a separate table for each pairwise metric.
    Columns: method | v0 v1 v2 ... Avg
    """
    if not pairwise_metrics:
        return

    method_col_width = 25
    score_col_width = 8

    for metric in pairwise_metrics:
        heading = f"\n Final Table for {metric.upper()}:"
        print(heading)
        log.info(heading)

        # Header
        header_fmt = (
            f"{{:<{method_col_width}}}" +
            "".join([f"{{:^{score_col_width}}}" for _ in range(num_views)]) +
            f"{{:^{score_col_width}}}"
        )
        header_cols = ["Method"] + [f"v{i}" for i in range(num_views)] + ["Avg"]
        header_line = header_fmt.format(*header_cols)
        print(header_line)
        log.info(header_line)

        # Each method row
        for method_key, metric_dict in all_eval_scores.items():
            if method_key.startswith("View"):
                continue

            values = metric_dict.get(metric, None)
            if not isinstance(values, list) or len(values) < (num_views + 1):
                values = [0.0] * (num_views + 1)
            str_scores = [f"{v:.4f}" for v in values]
            row_line = header_fmt.format(method_key, *str_scores)
            print(row_line)
            log.info(row_line)


def print_single_volume_metric_table_horizontal(
    all_eval_scores: Dict[str, Dict[str, Union[List[float], float]]],
    single_metrics: List[str],
    log=logger
):
    """
    Print single-volume metrics horizontally (1 row per metric).
    Columns: method names
    """
    if not single_metrics:
        return

    method_names = list(all_eval_scores.keys())
    col_width = max(10, max(len(name) for name in method_names)) + 2

    heading = "\n Final Table for Single-Volume Metrics (Horizontal):"
    print(heading)
    log.info(heading)

    # Header row
    header = f"{'Metric':<{col_width}}" + "".join([f"{name:>{col_width}}" for name in method_names])
    print(header)
    log.info(header)

    # Rows per metric
    for metric in single_metrics:
        row = f"{metric.upper():<{col_width}}"
        for name in method_names:
            val = all_eval_scores.get(name, {}).get(metric)
            if isinstance(val, list):
                val = None
            row += f"{val:>{col_width}.4f}" if val is not None else f"{'--':>{col_width}}"
        print(row)
        log.info(row)


def build_summary_table(
    all_eval_scores: Dict[str, Dict[str, Union[List[float], float]]],
    metric_keys: List[str],
    num_views: int = None
) -> List[str]:
    """
    Build a summary table (one line per method) in text format.

    Returns:
        list of lines (strings) for printing or saving.
    """
    lines = []
    method_names = list(all_eval_scores.keys())

    # Table header
    header = ["Method"]
    for m in metric_keys:
        if isinstance(all_eval_scores[method_names[0]].get(m), list):
            header.append(f"{m.upper()}_AVG")
        else:
            header.append(m.upper())
    lines.append("\t".join(header))

    # Each row
    for method in method_names:
        row = [method]
        for m in metric_keys:
            val = all_eval_scores[method].get(m)
            if isinstance(val, list):
                val = val[-1]  # average from fused vs views
            row.append(f"{val:.4f}" if val is not None else "--")
        lines.append("\t".join(row))

    return lines
