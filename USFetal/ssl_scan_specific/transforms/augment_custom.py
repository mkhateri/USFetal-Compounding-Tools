
import torch
import random

# Custom Augmentation Registry 
CUSTOM_AUGMENTATION_REGISTRY = {
    "GaussianNoise": lambda keys: lambda data: apply_gaussian_noise(data, keys, prob=0.2)
}

# Custom Gaussian Noise Function with Probability
def apply_gaussian_noise(data, keys, mean=0.0, std_range=(0.005, 0.002), prob=0.5):
    """
    Applies Gaussian noise with a probability and a random standard deviation within a given range.
    
    Args:
        data (dict): Dictionary containing the MRI data.
        keys (list): List of keys to apply noise on.
        mean (float): Mean of the Gaussian noise (default: 0.0).
        std_range (tuple): Range of standard deviation (default: (0.005, 0.02)).
        prob (float): Probability of applying the noise (default: 0.5).
    
    Returns:
        dict: Updated data dictionary with added noise.
    """
    if random.random() < prob:  # Apply noise only with probability `prob`
        std = random.uniform(std_range[0], std_range[1])  # Sample std within range
        for key in keys:
            if key in data and key != "GT_iso":  # Exclude ground truth
                data[key] += torch.randn_like(data[key]) * std + mean

    return data
