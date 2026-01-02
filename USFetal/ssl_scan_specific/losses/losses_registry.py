from .base_losses import MSELoss, MAELoss, GradientLoss, SSIMLoss

REGISTERED_LOSSES = {
    "MSE_VOLUME": MSELoss,
    "MAE_VOLUME": MAELoss,
    "SSIM_VOLUME":    SSIMLoss,  
    "GRADIENT_VOLUME": GradientLoss,
}

def get_loss_function(name, **kwargs):
    name = name.upper()
    if name not in REGISTERED_LOSSES:
        raise ValueError(f"Loss function '{name}' is not registered. Available: {list(REGISTERED_LOSSES.keys())}")
    return REGISTERED_LOSSES[name](**kwargs)





