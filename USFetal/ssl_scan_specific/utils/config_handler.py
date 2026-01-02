import json
from pathlib import Path
import copy


def setup_experiment_directory(config):
    """Set up directories for saving logs, outputs, and checkpoints."""
    output_dir = Path(config["training"]["output_dir"])
    logs_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, logs_dir

def save_config(config, output_dir):
    """Save config file to output directory."""
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config initialized. Saved at: {config_path}")


def load_config(config_path, default_config="config/default.yaml"):
    """Load experiment configuration and merge with default settings."""
    with open(default_config, "r") as f:
        default_cfg = yaml.safe_load(f)

    with open(config_path, "r") as f:
        experiment_cfg = yaml.safe_load(f)

    # Merge configs, experiment settings override default
    default_cfg.update(experiment_cfg)
    return default_cfg





def set_config(config: dict) -> dict:
    """
    Finalize inference config by recovering model definition
    from the training experiment.
    """
    import copy
    import json
    from pathlib import Path

    cfg = copy.deepcopy(config)

    # Required for inference
    for k in ["inference", "data"]:
        if k not in cfg:
            raise KeyError(f"Missing required config section: '{k}'")

    # Locate training experiment from checkpoint path
    model_path = Path(cfg["inference"]["model_path"]).expanduser().resolve()
    exp_dir = model_path.parents[1]   # outputs_xxx/

    train_cfg_path = exp_dir / "config.json"
    if not train_cfg_path.exists():
        raise FileNotFoundError(
            f"Training config not found at {train_cfg_path}"
        )

    # Load training config
    with open(train_cfg_path, "r") as f:
        train_cfg = json.load(f)

    if "model" not in train_cfg:
        raise KeyError("Training config does not contain 'model' section")

    # SOURCE OF TRUTH
    cfg["model"] = train_cfg["model"]

    # Optional metadata
    cfg.setdefault("meta", {})
    cfg["meta"]["training_experiment"] = str(exp_dir)

    return cfg
