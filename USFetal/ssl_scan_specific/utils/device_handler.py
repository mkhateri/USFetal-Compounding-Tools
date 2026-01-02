import torch

def get_device():
    """Return the best available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_gpus():
    """Return the number of available GPUs."""
    return torch.cuda.device_count()

def wrap_model_for_multi_gpu(model, num_gpus):
    """Wrap model in DataParallel for multi-GPU training if necessary."""
    if num_gpus > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    return model
