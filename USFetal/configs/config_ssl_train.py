config = {
    "data": {
        "data_dir": "./data/SUBJECT_ID/",  # set to the subject folder (replace SUBJECT_ID with your subject's folder name) 
        "batch_size": 6,
        "num_workers": 1,
        "mode": "train",
        "dtype": "float32",
        "train_split": 0.7,
        "format": "nifti"  # "nifti"
    },

    "transforms": {
        "crop_size": [72, 72, 72], #crop size for training
        "min_nonzero_ratio": 0.25, #minimum non-zero ratio in the crop
        "num_augmentations": 1, #**randomly select num_augmentations from the augment_list** 
        "augment_list": ["RandGaussianNoise",
                            #"RandCoarseDropout",
                            #"RandGaussianBlur",
                            ],  

        "random_crop": True,
    },

    "training": {
        "output_dir": "./outputs_ssl_scan_specific/", 
        "num_epochs": 25, 
        "early_stopping_patience": 1000, 
        "stagnation_threshold": 100, 
        "loss": ["MAE_VOLUME", "SSIM_VOLUME"],
        "loss_weights": {"MAE_VOLUME": 100.0, "SSIM_VOLUME":0.02 },  
        "mean_consistency_weight": 0.0, 
        "dog_consistency_weight": 10.0, 
        "use_uncertainty_weighting": False, 
        "optimizer": "adamw", 
        "lr": 1e-4, 
        "weight_decay": 1e-3, 
        "scheduler_step": 25, 
        "scheduler_gamma": 0.9, 
        "gradient_clipping": 1.0, 
        "checkpoint_interval": 5, 
        "resume_training": False, 
        "resume_checkpoint": "./outputs/checkpoints/checkpoint_epoch_25.pth", 
        "num_gpus":1, 
        "use_amp": True,  # Enable AMP (Automatic Mixed Precision) 
        "amp_precision": "bf16",  # Can be "fp16", "bf16", or "fp32" 
        "seed": 42, 
        "upsampling": "trilinear", 
    },

    "model": {
        "name": "UltrasoundSR_DoGFusion",
        "in_channels": 1,
        "out_channels": 1,
        # "num_views": 3,

        "encoder_features": [64, 128, 256],

        "dog": {
                "sigma1": 0.5,
                "sigma2": 2.5,
                "boost_strength": 1.5
                },

        "use_attention": True
    },


    "metrics": ["PSNR", "SSIM"],
    "visualization": {
        "log_interval": 1,
        "save_scale_factor": 255,
        "num_samples": 3
    },

    "logging": {
        "use_tensorboard": True,  
        "use_wandb": False,
        "wandb_project": "MRI-SuperRes",
        "wandb_run_name": "experiment_run"
    },
}

