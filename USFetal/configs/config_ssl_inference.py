
config = {
    "inference": {
        "model_name": "UltrasoundSR_DoGFusion",
        "model_path": "./outputs_ssl_scan_specific/checkpoints/best_checkpoint.pth",
        "output_dir": "./outputs_ssl_scan_specific/inference_results/",
        "batch_size": 1
    },
    "data": {
        "data_dir": "./data/SUBJECT_ID/",  # set to the subject folder (replace SUBJECT_ID with your subject's folder name)
        "dtype": "bf16",  # This should be fixed in your DataLoader (maps to torch.bfloat16)
        "num_workers": 1
    },

    "patching": {
        "patch_size": [72, 72, 72], #patch size for inference
        "overlap": [48, 48, 48] #overlap size for inference
    },
    "metrics": ["PSNR", "SSIM"],
    
    # model parameters (must match the trained model)so it is loaded from the saved experiment

}