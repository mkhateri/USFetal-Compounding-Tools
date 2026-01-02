import argparse
import importlib.util
import torch
import json
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.parallel
from pathlib import Path
import wandb
import os

import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for matplotlib

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


from torch.utils.tensorboard import SummaryWriter
from ssl_scan_specific.data.dataloader_train import get_dataloaders
from ssl_scan_specific.models.model_builder import ModelBuilder
from ssl_scan_specific.losses import LossEvaluator
from ssl_scan_specific.metrics import MetricsEvaluator

from ssl_scan_specific.utils.logger import setup_logger
from ssl_scan_specific.utils.checkpoint_handler import save_checkpoint, load_checkpoint
from ssl_scan_specific.utils.device_handler import get_device, get_num_gpus, wrap_model_for_multi_gpu
from ssl_scan_specific.utils.config_handler import setup_experiment_directory, save_config

from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler  
torch.backends.cudnn.benchmark = True # Boosts speed for consistent batch sizes
torch.backends.cudnn.enabled = True # Ensure fast convolutions


class Trainer:
    def __init__(self, config, print_config=False):
        """Initialize the trainer with configurations, logging, model setup, and training components."""
        self.config = config

        # Setup directories and save config
        self._setup_config(print_config)

        # Setup logger
        self._setup_logging()

        # Setup environment
        self._setup_device()

        # Setup training components
        self._setup_mixed_precision()
        self._setup_model()
        self._setup_loss_optimizer_scheduler()
        self._setup_dataloaders()

        # Best Validation Loss Tracking
        self.best_loss = float("inf")

        # Resume Training If Needed
        self.start_epoch = 1
        self._resume_checkpoint()

        # Initialize Adaptive Uncertainty Weighting
        self._initialize_uncertainty_weighting()

    def _setup_config(self, print_config):
        """Validate and set up required directories, then save config."""
        self.output_dir, logs_dir = setup_experiment_directory(self.config)
        save_config(self.config, self.output_dir)

        # Print configuration if requested
        self._print_config(print_config)


    def _print_config(self, print_config):
        """Prints the training configuration if enabled."""
        if print_config:
            print("\nTraining Configuration:\n", json.dumps(self.config, indent=4))


    def _setup_device(self):
        """Set up device (CPU/GPU) and number of GPUs."""
        self.device = get_device()
        self.num_gpus = get_num_gpus()

        if self.num_gpus > 1:
            self.logger.info(f"Multi-GPU Training | GPUs Available: {self.num_gpus}")
        elif self.num_gpus == 1:
            self.logger.info("Single-GPU Training")
        else:
            self.logger.info("Training is running on CPU (No GPU detected).")


    def _setup_logging(self):
        """Initialize unified logging for standard logger, TensorBoard & wandb."""
        log_cfg = self.config.get("logging", {})

        self.logged_epochs = set()

        # Standard Logger
        self.logger = setup_logger(self.output_dir / "training.log")

        # TensorBoard Logger
        self.use_tb = log_cfg.get("use_tensorboard", True)
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs") if self.use_tb else None

        # Weights & Biases Logger
        self.use_wandb = log_cfg.get("use_wandb", False)
        if self.use_wandb:
            wandb.require("core")  
            wandb.init(
                project=log_cfg.get("wandb_project", "MRI-SuperRes"),
                config=self.config,
                name=log_cfg.get("wandb_run_name", "experiment_run"),
                dir=str(self.output_dir)
            )
            self.logger.info("wandb initialized.")

    def _initialize_uncertainty_weighting(self):
        """
        Initializes adaptive uncertainty weighting for loss functions.

        If enabled in the config, this function creates a learnable uncertainty parameter 
        for each loss term and integrates them into the optimizer.

        This method follows the uncertainty-based loss balancing from Kendall et al. (CVPR 2018).
        """
        self.use_uncertainty_weighting = self.config["training"].get("use_uncertainty_weighting", False)

        if self.use_uncertainty_weighting:
            self.loss_uncertainties = {}
            loss_names = self.config["training"]["loss"]
            preset_weights = self.config["training"]["loss_weights"]  # Example: {"MAE_SLICEWISE": 1.0, ...}

            for loss_name in loss_names:
                # Initialize s_i such that (1/2) * exp(-s_i) = preset weight
                init_val = -torch.log(torch.tensor(2.0 * preset_weights[loss_name], device=self.device))
                self.loss_uncertainties[loss_name] = torch.nn.Parameter(init_val)

            # Add uncertainty parameters to optimizer for training
            self.optimizer.add_param_group({"params": list(self.loss_uncertainties.values())})
            self.logger.info("Adaptive Uncertainty Weighting Initialized for Multiple Losses.")


    def _log_metrics(self, metrics, epoch, phase="Train"):
        """
        Log each loss component and metric to TensorBoard, wandb, and the console.
        """
        # Convert BFloat16 to float for safe logging
        safe_metrics = {key: float(value) if value is not None else 0.0 for key, value in metrics.items()}

        # Log slice-wise and volume-wise losses
        slice_loss_keys = [key for key in safe_metrics if "SLICEWISE" in key]
        volume_loss_keys = [key for key in safe_metrics if "VOLUME" in key]

        for key, value in safe_metrics.items():
            if "SLICEWISE" in key or "VOLUME" in key:
                self.logger.info(f"{key}: {value:.4f}")  # Log every individual loss component

        # Log aggregated mean losses
        if slice_loss_keys and volume_loss_keys:
            mean_slice_loss = sum(safe_metrics[key] for key in slice_loss_keys) / len(slice_loss_keys)
            mean_volume_loss = sum(safe_metrics[key] for key in volume_loss_keys) / len(volume_loss_keys)
            self.logger.info(f"Epoch {epoch} [{phase}] → Mean Slice-Wise Loss: {mean_slice_loss:.4f} | Mean Volume-Wise Loss: {mean_volume_loss:.4f}")

        if self.use_tb:
            for key, value in safe_metrics.items():
                self.writer.add_scalar(f"{phase}/{key}", value, epoch)

        if self.use_wandb:
            wandb_log_dict = {f"{phase}/{key}": value for key, value in safe_metrics.items()}
            wandb_log_dict["epoch"] = epoch  # Ensure epoch is logged
            wandb.log(wandb_log_dict)



    def _infer_num_views(self):
        """
        Dynamically infer number of views (V) from the dataset
        and store it directly in the model config.
        """
        train_loader, _ = get_dataloaders(self.config)
        batch = next(iter(train_loader))

        views = batch["views"]              # [B, V, D, H, W]
        num_views = views.shape[1]


        self.config["model"]["num_views"] = num_views
        # self.logger.info(f"Inferred num_views = {num_views}")



    def _setup_model(self):
        """Initialize the model and handle multi-GPU support."""

        # Infer num_views BEFORE model construction
        self._infer_num_views()

        # Build model
        self.model = ModelBuilder(self.config["model"]).get_model().to(self.device)

        self.model = wrap_model_for_multi_gpu(self.model, self.num_gpus)


        # overwrite saved config with final version
        save_config(self.config, self.output_dir)

        if self.num_gpus > 1:
            self.logger.info(f"Using {self.num_gpus} GPUs for training.")

        batch_size = self.config["data"]["batch_size"]
        if batch_size % self.num_gpus != 0:
            raise ValueError(
                f"Batch size ({batch_size}) is not divisible by the number of GPUs ({self.num_gpus})."
            )



    def _setup_loss_optimizer_scheduler(self):
        """Initialize loss function, optimizer, and scheduler."""
        self.loss_evaluator = LossEvaluator(self.config, self.device, self.amp_dtype)
        self.metric_evaluator = MetricsEvaluator(self.config, device=self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"].get("weight_decay", 1e-2),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config["training"]["scheduler_step"],
            gamma=self.config["training"]["scheduler_gamma"]
        )

    def _setup_dataloaders(self):
        """Initialize data loaders."""
        self.train_loader, self.valid_loader = get_dataloaders(self.config)

    def _setup_mixed_precision(self):
        """Setup AMP for mixed precision training if enabled."""
        self.use_amp = self.config["training"].get("use_amp", True)
        precision_mode = self.config["training"].get("amp_precision", "fp16").lower()

        amp_dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }

        if precision_mode not in amp_dtype_map:
            raise ValueError(f"Unsupported amp_precision: {precision_mode}. Choose 'fp32', 'fp16', or 'bf16'.")

        self.amp_dtype = amp_dtype_map[precision_mode]
        self.scaler = GradScaler() if self.use_amp else None  # Only use GradScaler if AMP is enabled
        
        precision_str = "full precision (FP32)" if not self.use_amp else f"AMP with {precision_mode.upper()} precision"
        self.logger.info(f"Using {precision_str}.")


    def _resume_checkpoint(self):
        """Check if training should resume and handle checkpoint loading."""
        resume_training = self.config["training"].get("resume_training", False)
        checkpoint_path = Path(self.config["training"].get("resume_checkpoint", self.output_dir / "checkpoints/best_checkpoint.pth"))

        # Log whether training should resume or start from scratch
        if resume_training:
            self.logger.info("Resume Training: Enabled")
            
            # Check if the checkpoint exists
            if checkpoint_path.exists():
                self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
                
                self.start_epoch, self.best_loss = load_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler, checkpoint_path)

                #Log Learning Rate After Resume
                for param_group in self.optimizer.param_groups:
                    self.logger.info(f"Learning Rate After Resume: {param_group['lr']:.8f}")

                if self.num_gpus > 1 and not isinstance(self.model, torch.nn.DataParallel):
                    self.logger.info("Wrapping model in DataParallel for multi-GPU training.")
                    self.model = torch.nn.DataParallel(self.model)

                self.logger.info(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.4f}.")

            else:
                self.logger.error("ERROR: Resume training is enabled, but no checkpoint was found! Stopping training.")
                exit(1)  # STOP training if checkpoint is missing

        else:
            self.logger.info("Resume Training: Disabled - Starting from Scratch")
            self.start_epoch = 1
            self.best_loss = float("inf")



    def compute_total_loss(self, outputs, inputs):
        """
        Computes slice-wise, volume-wise, and total loss, applying uncertainty weighting if enabled.
        """
        # Compute slice-wise & volume-wise losses
        loss_dict, total_slicewise_loss, total_volume_loss = self.loss_evaluator.compute_loss(outputs, inputs)

        # Include mean consistency loss (stored inside loss_dict)
        mean_consistency_loss = loss_dict.get("MEAN_CONSISTENCY_LOSS", torch.tensor(0.0, device=self.device))

        #print(f"[DEBUG] Slice Loss: {total_slicewise_loss.item()}, Volume Loss: {total_volume_loss.item()}, Mean Consistency Loss: {mean_consistency_loss.item()}")

        # Base total loss (without uncertainty weighting)
        base_total_loss = total_slicewise_loss + total_volume_loss + mean_consistency_loss  

        if self.use_uncertainty_weighting:
            total_loss = torch.tensor(0.0, device=self.device, dtype=self.amp_dtype)
            for key, val in loss_dict.items():
                if key in self.loss_uncertainties:
                    s = self.loss_uncertainties[key]
                    exp_neg_s = torch.exp(-s).detach().clamp(min=1e-3, max=10.0)
                    weighted_loss = 0.5 * exp_neg_s * val + 0.5 * s
                    total_loss += weighted_loss
                elif isinstance(val, torch.Tensor):
                    total_loss += val
        else:
            total_loss = base_total_loss

        # Ensure proper logging of TOTAL_LOSS
        loss_dict["TOTAL_LOSS"] = total_loss

        #print(f"[DEBUG] Final TOTAL_LOSS: {total_loss.item()}")

        return loss_dict, total_loss


    def _log_to_wandb_and_tensorboard(self, log_dict, epoch, phase, console_log=True):
        """
        Log ONLY volume-based losses to console, TensorBoard, and wandb.
        """

        # ---- Extract volume losses only ----
        volume_losses = {
            key: value
            for key, value in log_dict.items()
            if "VOLUME" in key.upper() and "TOTAL" not in key.upper()
        }

        # Optional: mean consistency (if you want to keep it)
        mean_consistency_loss = log_dict.get("MEAN_CONSISTENCY_LOSS", 0.0)

        # ---- Compute total loss (volume only) ----
        total_volume_loss = sum(volume_losses.values())

        # ---- Console logging ----
        self.logger.info(
            f"Epoch {epoch} [{phase}] → "
            f"Volume Loss Sum: {total_volume_loss:.4f} | "
            f"Total {phase} Loss: {total_volume_loss:.4f}"
        )

        for key, value in volume_losses.items():
            loss_name = key.replace("_VOLUME", "")
            self.logger.info(f"    ├── {loss_name}: Volume {value:.4f}")

        # ---- TensorBoard logging ----
        if self.use_tb:
            self.writer.add_scalar(f"{phase}/VolumeLoss", total_volume_loss, epoch)
            self.writer.add_scalar(f"{phase}/TotalLoss", total_volume_loss, epoch)

            for key, value in volume_losses.items():
                self.writer.add_scalar(f"{phase}/{key}", value, epoch)

            if mean_consistency_loss:
                self.writer.add_scalar(
                    f"{phase}/MeanConsistencyLoss",
                    mean_consistency_loss,
                    epoch
                )

        # ---- wandb logging ----
        if self.use_wandb:
            wandb_log_dict = {
                f"{phase}/VolumeLoss": total_volume_loss,
                f"{phase}/TotalLoss": total_volume_loss,
                "epoch": epoch,
            }

            for key, value in volume_losses.items():
                wandb_log_dict[f"{phase}/{key}"] = value

            if mean_consistency_loss:
                wandb_log_dict[f"{phase}/MeanConsistencyLoss"] = mean_consistency_loss

            wandb.log(wandb_log_dict)


    def train_one_epoch(self, epoch):
        """Trains the model for one epoch with mixed precision."""
        torch.cuda.empty_cache()
        self.model.train()

        total_loss_val = 0.0  # For accumulating final loss over batches
        running_loss_dict = {}  # For accumulating sub-losses from loss_dict
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Extract inputs
            multi_view_image = batch["views"].to(self.device, non_blocking=True)
            multi_view_mask = batch["masks"].to(self.device, non_blocking=True)

            model_inputs = multi_view_image * multi_view_mask
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.set_grad_enabled(True):
                with autocast(dtype=self.amp_dtype) if self.use_amp else nullcontext():
                    fused_out = self.model(model_inputs)

            outputs = {"fused_out": fused_out}
            inputs = {"views": multi_view_image, "masks": multi_view_mask}

            # Compute total loss
            loss_dict, final_loss = self.compute_total_loss(outputs, inputs)

            # Check for NaN or Inf losses
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                self.logger.warning(f" NaN/Inf detected in loss at batch {batch_idx}. Skipping update.")
                continue

            #self.logger.info(f"[DEBUG] Batch {batch_idx}: TOTAL_LOSS = {final_loss.item():.4f}")

            # Backpropagation with gradient debugging
            if self.use_amp and self.scaler:
                self.scaler.scale(final_loss).backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["training"]["gradient_clipping"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["training"]["gradient_clipping"])
                self.optimizer.step()

            # Accumulate final loss
            total_loss_val += final_loss.item()

            # Accumulate each sub-loss
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    running_loss_dict[key] = running_loss_dict.get(key, 0.0) + value.item()

        # Compute average loss per epoch
        avg_loss_dict = {k: v / num_batches for k, v in running_loss_dict.items()}
        avg_total_loss = total_loss_val / num_batches

        #self.logger.info(f"[DEBUG] Epoch {epoch} → Final TOTAL_LOSS: {avg_total_loss:.4f}")

        # Log metrics to TensorBoard & WandB
        self._log_to_wandb_and_tensorboard(avg_loss_dict, epoch, phase="Train", console_log=True)

        # Save visualizations for model output
        visualize_epoch_outputs(self.model, batch, self.device, epoch, save_path=str(self.output_dir / "visualizations"))

        return avg_total_loss  # Returns average loss for the epoch


    def train(self, num_epochs):
        """
        Train model while saving checkpoints and tracking best training loss.
        Uses self.best_loss as the ONLY source of truth (resume-safe).
        """
        patience_counter = 0
        patience = self.config["training"]["early_stopping_patience"]
        checkpoint_interval = self.config["training"].get("checkpoint_interval", 5)

        prev_lr = self.optimizer.param_groups[0]["lr"]

        for epoch in range(self.start_epoch, num_epochs + 1):

            # -------------------------
            # Train one epoch
            # -------------------------
            train_loss = self.train_one_epoch(epoch)

            self.logger.info(
                f"[DEBUG] Epoch {epoch}: train_loss = {train_loss:.6f}, "
                f"best_loss = {self.best_loss:.6f}"
            )

            # -------------------------
            # Interval checkpoint
            # -------------------------
            if epoch % checkpoint_interval == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    self.best_loss,
                    self.output_dir,
                    self.logger,
                    best=False,
                    interval=True,
                )
                self.logger.info(f"Checkpoint saved at epoch {epoch} (interval).")

            # -------------------------
            # Best model checkpoint
            # -------------------------
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                patience_counter = 0

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    self.best_loss,
                    self.output_dir,
                    self.logger,
                    best=True,
                    interval=False,
                )

                # self.logger.info(
                #     f"Best model updated at Epoch {epoch} | Best Loss: {self.best_loss:.4f}"
                # )

                if self.use_tb:
                    self.writer.add_scalar("Loss/Best_Training_Loss", self.best_loss, epoch)

                if self.use_wandb:
                    wandb.log(
                        {"Loss/Best_Training_Loss": self.best_loss, "epoch": epoch}
                    )

            else:
                patience_counter += 1

            # -------------------------
            # LR scheduler
            # -------------------------
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                self.logger.info(
                    f"Learning Rate Updated: {prev_lr:.8f} → {new_lr:.8f}"
                )
                prev_lr = new_lr

            # -------------------------
            # Early stopping
            # -------------------------
            if patience_counter >= patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}."
                )
                break




def visualize_epoch_outputs(model, batch, device, epoch, save_path):
    """
    Visualize a middle slice from:
      - Each full view (original data, not masked),
      - Overlaid with the boundary of the mask,
      - The mean of all (masked) views,
      - The fused output produced by the model.
    """
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    multi_view_image = batch["views"].to(device)   # [B, V, 1, D, H, W]
    multi_view_mask = batch["masks"].to(device)    # [B, V, 1, D, H, W]
    masked_views = multi_view_image * multi_view_mask

    raw_views_example = multi_view_image[0]        # [V, 1, D, H, W]
    masks_example = multi_view_mask[0]             # [V, 1, D, H, W]

    with torch.no_grad():
        fused_output = model(masked_views)         # [B, 1, D, H, W]
    fused_example = fused_output[0]                # [1, D, H, W]

    mean_example = masked_views[0].mean(dim=0)     # [1, D, H, W]

    _, D, H, W = fused_example.shape
    slice_idx = D // 2

    fused_slice = fused_example[0, slice_idx].cpu().numpy()
    mean_slice = mean_example[0, slice_idx].cpu().numpy()

    num_views = raw_views_example.shape[0]
    view_slices = [raw_views_example[v, 0, slice_idx].cpu().numpy() for v in range(num_views)]
    mask_slices = [masks_example[v, 0, slice_idx].cpu().numpy() for v in range(num_views)]

    # --- Grid layout ---
    total_images = num_views + 2  # views + mean + fused
    num_cols = 4
    num_rows = int(np.ceil(total_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    # --- Plot views with masks ---
    for i in range(num_views):
        ax = axs[i]
        ax.imshow(view_slices[i], cmap="gray")
        overlay_mask_boundary(ax, mask_slices[i], color="red")
        ax.set_title(f"View {i+1}")
        ax.axis("off")

    # --- Plot mean ---
    if len(axs) > num_views:
        axs[num_views].imshow(mean_slice, cmap="gray")
        axs[num_views].set_title("Mean")
        axs[num_views].axis("off")

    # --- Plot fused ---
    if len(axs) > num_views + 1:
        axs[num_views + 1].imshow(fused_slice, cmap="gray")
        axs[num_views + 1].set_title("Fused")
        axs[num_views + 1].axis("off")

    # --- Hide any unused axes ---
    for ax in axs[num_views + 2:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/epoch_{epoch:04d}.png")
    plt.close(fig)

    model.train()


def overlay_mask_boundary(ax, mask_slice, color="red"):
    """
    Draws boundary around a binary mask.
    """
    padded_mask = np.pad(mask_slice, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_mask, 0.5)
    for contour in contours:
        contour -= 1  # undo padding shift
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=color)


if __name__ == "__main__":
    # ============================================================
    # Default config path
    # ============================================================
    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "config_ssl_train.py"
    )

    # ============================================================
    # Load config from .py
    # ============================================================
    def load_config_from_py(path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        spec = importlib.util.spec_from_file_location("config_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "config"):
            raise AttributeError(f"{path} must define a `config` dictionary")

        return module.config

    # ============================================================
    # Argument parser (explicit & human-friendly)
    # ============================================================
    parser = argparse.ArgumentParser(
        description="SSL Fetal Ultrasound Training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to training config file",
    )

    # ---- Common things users actually change ----
    parser.add_argument("--data_dir", type=str, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for resume",
    )

    args = parser.parse_args()

    # ============================================================
    # Load base config (defaults)
    # ============================================================
    config = load_config_from_py(args.config)

    # ============================================================
    # Apply CLI values (ONLY if provided)
    # ============================================================
    if args.data_dir is not None:
        config["data"]["data_dir"] = str(
            Path(args.data_dir).expanduser().resolve()
        )

    if args.output_dir is not None:
        config["training"]["output_dir"] = str(
            Path(args.output_dir).expanduser().resolve()
        )

    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs

    if args.lr is not None:
        config["training"]["lr"] = args.lr

    if args.resume:
        config["training"]["resume_training"] = True

    if args.checkpoint is not None:
        config["training"]["resume_checkpoint"] = str(
            Path(args.checkpoint).expanduser().resolve()
        )

    # ============================================================
    # Train
    # ============================================================
    trainer = Trainer(config, print_config=True)
    trainer.train(config["training"]["num_epochs"])

