import monai.transforms as mt
import random

MONAI_AUGMENTATION_REGISTRY = {

    # Gaussian noise to simulate sensor imperfections and scanner noise
    "RandGaussianNoise": lambda keys=("views",): mt.RandGaussianNoised(
        keys=list(keys),  # Ensure keys is a list
        prob=0.2,
        mean=0,
        std=random.uniform(0.001, 0.005),
        allow_missing_keys=True,
    ), 

    # Simulates MRI-specific Rician noise caused by magnitude reconstruction  
    "RandRicianNoise": lambda keys=("views",): mt.RandRicianNoised(
        keys=list(keys),
        prob=0.2,
        mean=0.0,
        std=random.uniform(0.0001, 0.0005),
        allow_missing_keys=True,
    ),

    # Slightly scales intensity values to simulate scanner calibration differences  
    "RandScaleIntensity": lambda keys=("views",): mt.RandScaleIntensityd(
        keys=list(keys),
        factors=(0.995, 1.005),
        prob=0.1,
        allow_missing_keys=True,
    ),

    # Simulates RF coil inhomogeneity (bias field artifact) seen in MRI  
    "RandBiasField": lambda keys=("views",): mt.RandBiasFieldd(
        keys=list(keys),
        coeff_range=(0.0, 0.0015),
        prob=0.2,
        allow_missing_keys=True,
    ),

    # Simulates inter-scanner intensity variations in multi-site MRI datasets  
    "RandHistogramShift": lambda keys=("views",): mt.RandHistogramShiftd(
        keys=list(keys),
        num_control_points=random.randint(3, 8),
        prob=0.2,
        allow_missing_keys=True,
    ),

    # Simulates scanner defocus and minor motion blur 
    "RandGaussianBlur": lambda keys=("views",): mt.RandGaussianSmoothd(
        keys=list(keys),
        sigma_x=(0.05, 0.5),
        sigma_y=(0.05, 0.5),
        sigma_z=(0.05, 0.5),
        prob=0.2,
        allow_missing_keys=True,
    ),


    # Mimics missing slices or dropout artifacts seen in MRI reconstruction
    "RandCoarseDropout": lambda keys=("views",): mt.RandCoarseDropoutd(
        keys=list(keys),
        holes=1, 
        max_holes=3,
        spatial_size=(5, 5, 5),
        fill_value=0,  # Set dropped regions to zero
        prob=0.2,
        allow_missing_keys=True,
    ),

    # Structured Noise (Shuffle Instead of Zeroing Out)
    "RandCoarseShuffle": lambda keys=("views",): mt.RandCoarseDropoutd(
        keys=list(keys),
        holes=2, 
        max_holes=4,
        spatial_size=(3, 3),
        fill_value=None,
        prob=0.2,
        allow_missing_keys=True,
    ),
}
