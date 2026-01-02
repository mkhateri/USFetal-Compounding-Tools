# metrics/registry.py

"""
Dynamic metric registry for the USFetal Toolbox.

Allows registering metric functions using @register_metric("name").
Used by the evaluator to compute selected metrics dynamically.
"""

METRIC_REGISTRY = {}

def register_metric(name: str):
    """
    Decorator to register a metric function under a given name.

    Example:
    @register_metric("psnr")
    def psnr_3d(...): ...
    """
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator
