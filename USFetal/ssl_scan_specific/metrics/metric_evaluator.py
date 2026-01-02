import torch
from .metrics_registry import get_metric_function

class MetricsEvaluator:
    def __init__(self, config, device="cuda"):
        """
        Initialize selected metrics from the config.

        Args:
            config (dict): Configuration dictionary.
            device (str): Device to run metrics on.
        """
        self.device = device

        # Ensure `metrics` is extracted as a list safely
        if isinstance(config, dict):
            metrics_list = config.get("metrics", [])
        elif isinstance(config, list):
            metrics_list = config  # If config is mistakenly a list, use it directly
        else:
            raise ValueError(f"Expected `config` to be a dictionary or list, but got {type(config)}")

        if not isinstance(metrics_list, list):
            raise ValueError(f"Expected 'metrics' to be a list, but got {type(metrics_list)}")

        self.selected_metrics = {}
        for name in metrics_list:
            metric = get_metric_function(name)
            
            # Only move to device if the metric supports `.to(device)`
            if hasattr(metric, "to"):
                metric.to(device)  # Ensure the metric is moved to the correct device

            self.selected_metrics[name] = metric

    def compute_metrics(self, outputs, targets):
        """
        Computes metrics for the given outputs and targets.

        Args:
            outputs (tensor): The predicted values (output from model).
            targets (tensor): The ground truth values.

        Returns:
            dict: A dictionary containing metric names and their corresponding values.
        """
        results = {}

        # Move tensors to the correct device
        outputs = outputs.to(self.device)  # No need for `.items()` here since outputs is a tensor
        targets = targets.to(self.device)

        for name, metric in self.selected_metrics.items():
            # Compute metric
            metric_value = metric(outputs, targets)
            if isinstance(metric_value, torch.Tensor):  # Ensure it's a tensor before calling .mean()
                results[name] = metric_value.mean().item()
            else:
                results[name] = float(metric_value)  # Directly store float if it's already a scalar

        return results
