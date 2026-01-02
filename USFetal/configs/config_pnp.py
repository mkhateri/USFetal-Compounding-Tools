# pnp/configs/config_pnp.py

config_pnp = {
    "data_parent": "./data",          # parent folder with subject_01, subject_02, ...
    "output_dir": "./output_pnp",          # where to write results
    "model_zoo": "./model_zoo",            # contains drunet_gray.pth, ircnn_gray.pth from KAIR Toolbox (https://github.com/cszn/KAIR)
    "experiment_tag": "USFetal_pnp",       # tag for the experiment

    "io": {
        "apply_mask": True,
        "binarize_mask": True,
        "mask_thr": 0.0,
    },

    # DPIR/denoiser params
    "dpir": {
        "model_name": "drunet_gray",
        "noise_level_img": 3,   # image noise in [0. .255] scale
        "sf": 1,                # scale factor
        "iter_num": 10,
        "x8": True,
        "classical_degradation": True,
        "kernel_size": 13,
        "kernel_sigma": 0.5,
        "override_kernel_sigma": True,
        "modelSigma1": 49, # default 49
    },

    # runtime
    "device": "cuda"  # "cpu" | "cuda"
}



