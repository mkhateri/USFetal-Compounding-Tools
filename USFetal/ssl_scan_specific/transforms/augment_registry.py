from .augment_monai import MONAI_AUGMENTATION_REGISTRY
from .augment_custom import CUSTOM_AUGMENTATION_REGISTRY  

# Create a unified augmentation registry
AUGMENTATION_REGISTRY = {}

# Register MONAI augmentations
AUGMENTATION_REGISTRY.update(MONAI_AUGMENTATION_REGISTRY)

# Register custom augmentations
AUGMENTATION_REGISTRY.update(CUSTOM_AUGMENTATION_REGISTRY)

def get_augmentation(name):
    """Fetch an augmentation by name from the registry."""
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Augmentation '{name}' not found in registry!")
    return AUGMENTATION_REGISTRY[name]
