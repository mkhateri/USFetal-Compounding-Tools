config = {          
    "data_parent": "./data/",      # Parent directory containing sample folders.
    "output_dir": "./output_traditional/",  # output directory
    "patch_size": (96, 96, 96), # size of patches for patchwise methods
    "overlap": (48, 48, 48),  # overlap between patches for patchwise methods
    "methods": {
        "mean": {
            "standard": {}
        },
        "median": {
            "standard": {}
        },
        "max": {
            "standard": {}
        },
        # methods configurations PCA, ICA, DoG, Variational methods
        "pca": {
            "standard": {
                "k": "auto",
                "evr_thresh": 0.95,
                "min_k": 1,
                "max_k": 8,
                "ncomp_cap": 16,
                "weight_mode": "evr",
                "detail_gamma": 0.5,
                "align_sign": True,
                "add_global_mean": False
            },
            "patchwise": {
                "patch_size": (48,48,48),
                "overlap": (24,24,24),
                "k": "auto",
                "evr_thresh": 0.95,
                "min_k": 1,
                "max_k": 8,
                "ncomp_cap": 16,
                "weight_mode": "evr",
                "detail_gamma": 0.5,
                "align_sign": True,
                "add_patch_mean": True
            }
        },
        "ica": {
            "standard": {
                "k": 2,
                "weight_mode": "kurtosis",     # "kurtosis" | "uniform" | "custom"
                # "ic_weights": [0.7, 0.3],    # only if weight_mode == "custom"
                "detail_gamma": 0.0,

                "ncomp_cap": 8,
                "standardize": True,
                "ica_max_iter": 1000,
                "ica_tol": 1e-3,
                "algorithm": "parallel",       # or "deflation" if convergence is tough
                "fun": "logcosh",
                "alpha": 1.2,
                "restarts": 3,
                "ica_random_state": 0,

                "align_sign": True,
                "add_global_mean": False
            },

            "patchwise": {
                "patch_size": (96, 96, 96),
                "overlap": (48, 48, 48),

                "k": 2,
                "weight_mode": "kurtosis",
                # "ic_weights": [0.7, 0.3],
                "detail_gamma": 0.0,

                "ncomp_cap": 8,
                "standardize": True,
                "ica_max_iter": 1000,
                "ica_tol": 1e-3,
                "algorithm": "parallel",
                "fun": "logcosh",
                "alpha": 1.2,
                "restarts": 3,
                "ica_random_state": 0,

                "align_sign": True,
                "add_patch_mean": True
            }
        },
        "dog": {
            "standard": {
                "sigma1": 0.5,
                "sigma2": 2.0
            },

            "multiscale": {
            "sigma_scales": [0.5, 1.0, 2.0, 4.0],
            "rule_views": "mean",
            "rule_scales": "sum",
            "band_gain": 3.0,
            "add_mean": True
            },

        },

        "variational": {
            "dog": {
                "alpha": 1.0,             # TV weight
                "beta": 200.0,            # Strong DoG consistency
                "gamma": 0.01,            # Fidelity weight
                "sigma1": 0.5,
                "sigma2": 2.0,
                "lr": 0.01,
                "steps": 20
            },
            "gradient": {
                "alpha": 1.0,
                "beta": 200.0,            # Strong gradient consistency
                "gamma": 0.01,
                "lr": 0.01,
                "steps": 20
            }
        },
        
    },
    #comment out methods not being used
    "selected_methods": [
        ("pca", "standard"),            # PCA global
        # ("pca", "patchwise"),         # PCA patchwise
        # ("ica", "standard"),          # ICA global
        # ("ica", "patchwise"),         # ICA patchwise
        # ("dog", "standard"),          # DoG standard
        ("dog", "multiscale"),          # DoG multiscale 
        ("variational", "dog"),                    # Variational with DoG
        # ("variational", "gradient"),             # Variational with gradient
        ],
    "apply_mask": True, # Whether to apply a mask when loading data
}
