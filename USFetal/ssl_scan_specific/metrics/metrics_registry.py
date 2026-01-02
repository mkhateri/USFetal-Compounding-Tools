from .base_metrics import DiceCoefficient, DiceMetric, IoUMetric, SSIMMetricWrapper, PSNRMetricWrapper

# Registry of available metrics
REGISTERED_METRICS = {
    "SSIM": SSIMMetricWrapper,
    "PSNR": PSNRMetricWrapper,
    "DiceCoefficient": DiceCoefficient,
    "DiceMetric": DiceMetric,
    "IoU": IoUMetric,
    }

def get_metric_function(name, device="cuda", **kwargs):
    """
    Retrieve a metric function by name.

    Args:
        name (str): Name of the metric function.
        device (str): Device to place the metric on (for PyTorch-based metrics).
        **kwargs: Optional parameters for custom metrics.

    Returns:
        Instantiated metric function/class.
    """
    if name not in REGISTERED_METRICS:
        raise ValueError(f"Metric '{name}' is not registered in REGISTERED_METRICS.")

    metric_class = REGISTERED_METRICS[name]

    # Check if the metric class accepts a 'device' argument before passing it
    if name in ["PSNR"]:  # Add other metrics if they require a device argument
        metric = metric_class(device=device, **kwargs)
    else:
        metric = metric_class(**kwargs)  # Do not pass device if not needed

    # Move to device if it's a torch-based metric
    if hasattr(metric, "to"):  
        metric.to(device)

    return metric
