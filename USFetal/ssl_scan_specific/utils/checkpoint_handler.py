import torch
from pathlib import Path
import logging

# Setup Logger (Avoid Multiple Initializations)
logger = logging.getLogger("TrainingLogger")
if not logger.hasHandlers():  # Prevent duplicate handlers
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss, output_dir, logger, best=False, interval=False):
    """Save a model checkpoint with optimizer, scheduler, and AMP scaler states."""
    try:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,  
            "best_loss": best_loss,
        }

        # Save last checkpoint (overwrite)
        torch.save(checkpoint, checkpoint_dir / "last_checkpoint.pth")

        # Save best checkpoint (overwrite)
        if best:
            torch.save(checkpoint, checkpoint_dir / "best_checkpoint.pth")
            logger.info(f"Best model updated at Epoch {epoch} | Best Loss: {best_loss:.4f}")

        # Save interval checkpoints without overwriting
        if interval:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Interval checkpoint saved at: {checkpoint_path}")

    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path):
    """Load model checkpoint and handle single-GPU and multi-GPU cases."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

        state_dict = checkpoint["model_state_dict"]

        # Fix: Handle loading between DataParallel and non-DataParallel models
        model_state_dict = model.state_dict()
        new_state_dict = {}

        for key in state_dict:
            new_key = key
            if key.startswith("module.") and not any(k.startswith("module.") for k in model_state_dict):
                new_key = key.replace("module.", "")  # Remove "module." prefix for single-GPU models
            elif not key.startswith("module.") and any(k.startswith("module.") for k in model_state_dict):
                new_key = f"module.{key}"  # Add "module." prefix for multi-GPU models

            new_state_dict[new_key] = state_dict[key]

        # Load the corrected state dict
        model.load_state_dict(new_state_dict, strict=False)  # Allow missing keys

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore AMP scaler state if using mixed precisionis
        if "scaler_state_dict" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            #logger.info("AMP scaler state restored.")

        epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        logger.info(f"Checkpoint loaded: Epoch {epoch}, Best Loss: {best_loss:.4f}")
        return epoch, best_loss

    except FileNotFoundError:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 1, float("inf")

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 1, float("inf")  # Default to epoch 1 and reset best_loss if error occurs
